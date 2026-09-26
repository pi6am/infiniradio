"""Public site: player page, now-playing fragments, voting API, HLS files."""

from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .. import library
from ..config import runtime_get
from ..db import DB
from ..util import fmt_duration, human_bytes, iso, parse_iso, utcnow, wilson_score

log = logging.getLogger(__name__)
router = APIRouter()

LISTENER_COOKIE = "ir_listener"
VOTE_WINDOW_S = 60
VOTE_MAX = 30


# ------------------------------------------------------------------ helpers

def ensure_listener(request: Request) -> tuple[str, str | None]:
    """Return (listener_id, cookie_value_to_set | None)."""
    lid = request.cookies.get(LISTENER_COOKIE)
    if lid:
        return lid, None
    new = uuid.uuid4().hex
    return new, new


def _set_cookie(response: Response, value: str) -> None:
    response.set_cookie(LISTENER_COOKIE, value, max_age=365 * 24 * 3600,
                        httponly=True, samesite="lax")


def _prompt_names(db: DB) -> dict[int, str]:
    return {p["id"]: p["name"] for p in db.query("SELECT id, name FROM prompts")}


def track_public(row: dict, prompts: dict[int, str], my: dict[int, int]) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "prompt": prompts.get(row["prompt_id"], "custom"),
        "music_key": row["music_key"],
        "tempo": row["tempo"],
        "duration_s": row["duration_s"],
        "duration_fmt": fmt_duration(row["duration_s"] or 0),
        "vote_up": row["vote_up"],
        "vote_down": row["vote_down"],
        "score": round(wilson_score(row["vote_up"], row["vote_down"]), 3),
        "protected": bool(row["protected"]),
        "is_variation": row["parent_track_id"] is not None,
        "my_vote": my.get(row["id"], 0),
    }


def now_playing(db: DB, cfg, listener_id: str, at_ms: float | None = None,
                encoder=None) -> dict:
    """Current + recent tracks.

    at_ms: the caller's position on the HLS EXT-X-PROGRAM-DATE-TIME timeline
    (epoch ms, server clock domain). When given and the encoder exposes the
    timeline, tracks are matched by exact program offset — frame-accurate
    voting regardless of player buffering. Otherwise we fall back to wall
    clock against scheduled airtimes (approximate).
    """
    prompts = _prompt_names(db)
    my = library.my_votes(db, listener_id)

    mode = "clock"
    if (at_ms is not None and encoder is not None
            and getattr(encoder, "started_wall", None) is not None):
        try:
            # started_wall is naive UTC; timestamp() would read it as local
            # time, so attach UTC explicitly before converting to epoch.
            from datetime import timezone as _tz
            start_epoch = encoder.started_wall.replace(tzinfo=_tz.utc).timestamp()
            prog_at = at_ms / 1000.0 - start_epoch
            if abs(prog_at - encoder.program_seconds()) < 6 * 3600:
                mode = "program"
        except (TypeError, ValueError, OSError):
            pass

    if mode == "program":
        rows = db.query(
            "SELECT p.started_at AS airtime, p.prog_offset AS prog_offset, t.*"
            " FROM play_log p JOIN tracks t ON t.id = p.track_id"
            " WHERE p.epoch = ? AND p.prog_offset IS NOT NULL"
            " AND p.prog_offset <= ?"
            " ORDER BY p.prog_offset DESC LIMIT 10",
            (encoder.epoch, prog_at),
        )
        current = None
        if rows:
            r = rows[0]
            if r["prog_offset"] + (r["duration_s"] or 1.0) > prog_at:
                current = track_public(r, prompts, my)
                current["airtime"] = r["airtime"]
    else:
        t = iso(utcnow())
        rows = db.query(
            "SELECT p.started_at AS airtime, t.* FROM play_log p"
            " JOIN tracks t ON t.id = p.track_id"
            " WHERE p.started_at <= ? ORDER BY p.started_at DESC LIMIT 10",
            (t,),
        )
        current = None
        if rows:
            r = rows[0]
            end = parse_iso(r["airtime"]) + timedelta(seconds=r["duration_s"] or 1.0)
            if end >= parse_iso(t):
                current = track_public(r, prompts, my)
                current["airtime"] = r["airtime"]

    recent = [track_public(r, prompts, my) | {"airtime": r["airtime"]} for r in rows]
    return {
        "server_time": iso(utcnow()),
        "latency_hint": float(cfg.get("stream.latency_hint", 20.0)),
        "sync": mode,
        "current": current,
        "recent": recent,
    }


class _VoteLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        start, count = self._buckets.get(key, (now, 0))
        if now - start > VOTE_WINDOW_S:
            start, count = now, 0
        count += 1
        self._buckets[key] = (start, count)
        return count <= VOTE_MAX


_limiter = _VoteLimiter()


def _apply_vote(rt, request: Request, track_id: int, value: int,
                listener: tuple[str, str | None]) -> dict | None:
    if value not in (-1, 0, 1):
        return None
    lid, _ = listener
    if not _limiter.allow(lid):
        return None
    row = rt.db.one("SELECT id FROM tracks WHERE id = ? AND status != 'deleted'", (track_id,))
    if not row:
        return None
    return library.apply_vote(rt.db, rt.cfg, track_id, lid, value)


# ------------------------------------------------------------------ pages

@router.get("/", response_class=HTMLResponse)
async def player(request: Request):
    rt = request.app.state.rt
    listener = ensure_listener(request)
    ctx = {
        "request": request,
        "np": now_playing(rt.db, rt.cfg, listener[0], encoder=_encoder(rt)),
        "station": station_info(rt),
    }
    resp = request.app.state.templates.TemplateResponse(request, "player.html", ctx)
    if listener[1]:
        _set_cookie(resp, listener[1])
    return resp


def _encoder(rt):
    return rt.feeder.encoder if rt.feeder else None


def station_info(rt) -> dict:
    cfg = rt.cfg
    stats = rt.db.one(
        "SELECT COALESCE(SUM(status = 'ready'), 0) AS ready,"
        " COALESCE(SUM(protected = 1 AND status = 'ready'), 0) AS protected,"
        " COALESCE(SUM(play_count), 0) AS plays FROM tracks"
    ) or {"ready": 0, "protected": 0, "plays": 0}
    usage = library.disk_usage(rt.db)
    quota = int(runtime_get(cfg, rt.db, "library.quota_gb") * (1 << 30))
    return {
        "library_size": stats["ready"],
        "protected": stats["protected"],
        "total_plays": stats["plays"],
        "disk": f"{human_bytes(usage)} / {human_bytes(quota)}",
    }


# ------------------------------------------------------------------ fragments

@router.get("/nowplaying-card", response_class=HTMLResponse)
async def nowplaying_card(request: Request, at: float | None = None):
    rt = request.app.state.rt
    listener = ensure_listener(request)
    resp = request.app.state.templates.TemplateResponse(
        request, "_nowplaying.html",
        {"request": request,
         "np": now_playing(rt.db, rt.cfg, listener[0], at_ms=at, encoder=_encoder(rt))},
    )
    if listener[1]:
        _set_cookie(resp, listener[1])
    return resp


@router.post("/vote/{track_id}/{value}", response_class=HTMLResponse)
async def vote_fragment(request: Request, track_id: int, value: int,
                        at: str = Form("")):
    rt = request.app.state.rt
    listener = ensure_listener(request)
    _apply_vote(rt, request, track_id, value, listener)
    resp = request.app.state.templates.TemplateResponse(
        request, "_nowplaying.html",
        {"request": request,
         "np": now_playing(rt.db, rt.cfg, listener[0], at_ms=_parse_at(at),
                           encoder=_encoder(rt))},
    )
    if listener[1]:
        _set_cookie(resp, listener[1])
    return resp


def _parse_at(raw: str | None) -> float | None:
    """Defensive parse of a client-reported timeline position.

    htmx form-encodes js-null as the string "null"; never let that (or any
    other junk) 422 the vote endpoint — fall back to clock mode instead.
    """
    if not raw:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


# ------------------------------------------------------------------ JSON API

@router.get("/api/nowplaying")
async def api_nowplaying(request: Request, at: float | None = None):
    rt = request.app.state.rt
    listener = ensure_listener(request)
    data = now_playing(rt.db, rt.cfg, listener[0], at_ms=at, encoder=_encoder(rt))
    resp = JSONResponse(data)
    if listener[1]:
        _set_cookie(resp, listener[1])
    return resp


@router.post("/api/vote")
async def api_vote(request: Request):
    import json as _json
    rt = request.app.state.rt
    listener = ensure_listener(request)
    try:
        body = _json.loads((await request.body()) or b"{}")
        track_id = int(body["track_id"])
        value = int(body["value"])
    except (ValueError, KeyError):
        return JSONResponse({"error": "expected {track_id, value}"}, status_code=400)
    row = _apply_vote(rt, request, track_id, value, listener)
    if row is None:
        return JSONResponse({"error": "invalid vote"}, status_code=400)
    resp = JSONResponse({
        "track_id": row["id"], "vote_up": row["vote_up"], "vote_down": row["vote_down"],
        "protected": bool(row["protected"]),
    })
    if listener[1]:
        _set_cookie(resp, listener[1])
    return resp


@router.get("/api/track/{track_id}")
async def api_track(request: Request, track_id: int):
    rt = request.app.state.rt
    row = rt.db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    if not row or row["status"] == "deleted":
        return JSONResponse({"error": "not found"}, status_code=404)
    listener = ensure_listener(request)
    return JSONResponse(track_public(row, _prompt_names(rt.db),
                                     library.my_votes(rt.db, listener[0])))


# ------------------------------------------------------------------ HLS files

import re as _re

_PDT_TAIL = _re.compile(r"([+-]\d{2})(\d{2})$")


def _fix_pdt_offsets(text: str) -> str:
    """Normalize #EXT-X-PROGRAM-DATE-TIME offsets to ±HH:MM.

    ffmpeg's strftime emits `-0700` (no colon), which the ECMAScript
    Date Time String Format does not allow; Firefox's Date.parse returns
    NaN for it (Chrome is lenient), breaking hls.js timeline sync there.
    """
    out = []
    for line in text.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            line = _PDT_TAIL.sub(r"\1:\2", line)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


@router.get("/live/{fname:path}")
async def live(request: Request, fname: str):
    rt = request.app.state.rt
    hls_dir: Path = rt.cfg.path("library.data_dir") / "hls"
    try:
        p = (hls_dir / fname).resolve()
        p.relative_to(hls_dir.resolve())
    except (ValueError, OSError):
        return Response(status_code=404)
    if not p.is_file():
        return Response(status_code=404)
    if p.suffix == ".m3u8":
        # Playlists must never be cached, and their PDT offsets normalized.
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(_fix_pdt_offsets(p.read_text(encoding="utf-8")),
                                 headers={"Cache-Control": "no-store"})
    # Segments are immutable for as long as they exist in the window.
    return FileResponse(p, headers={"Cache-Control": "public, max-age=180"})
