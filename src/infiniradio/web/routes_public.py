"""Public site: player page (JSON schedule + client-side rendering),
voting API, and HLS file serving.

The player polls /api/schedule once per song change (computed locally from
the PDT timeline boundaries in the response) and updates the DOM itself;
vote clicks are optimistic and reconciled via /api/vote. No server-rendered
fragments are swapped into the page.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from datetime import timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from .. import library
from ..config import runtime_get
from ..db import DB
from ..util import (
    epoch_ms,
    fmt_duration,
    iso,
    parse_iso,
    utcnow,
    wilson_score,
)

log = logging.getLogger(__name__)
router = APIRouter()

LISTENER_COOKIE = "ir_listener"
VOTE_WINDOW_S = 60
VOTE_MAX = 30          # votes per listener id per window
IP_VOTE_MAX = 120      # votes per client IP per window — blunts scripted
                       # ballot stuffing by rotating freshly-minted ids


# ------------------------------------------------------------------ helpers

def ensure_listener(request: Request) -> tuple[str, str | None]:
    """Return (listener_id, cookie_value_to_set | None)."""
    lid = request.cookies.get(LISTENER_COOKIE)
    if lid:
        return lid, None
    new = uuid.uuid4().hex
    return new, new


def _set_cookie(response: Response, value: str, request: Request) -> None:
    # Secure only when the request itself arrived over HTTPS (behind a
    # TLS-terminating proxy such as cloudflared); plain-HTTP LAN access
    # keeps working.
    response.set_cookie(LISTENER_COOKIE, value, max_age=365 * 24 * 3600,
                        httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")


def _prompt_names(db: DB) -> dict[int, str]:
    return {p["id"]: p["name"] for p in db.query("SELECT id, name FROM prompts")}


_LYRICS_HDR = re.compile(r"^\s*\[[^\]]*\]\s*$")


def lyrics_public(text: str | None) -> str | None:
    """Lyrics for display — library.authored_lyrics (the one implementation
    of the strip-header/None-when-instrumental rule; the web layer keeps
    this alias because the player payloads and the admin pages speak it)."""
    return library.authored_lyrics(text)


def track_public(row: dict, prompts: dict[int, str], my: dict[int, int]) -> dict:
    # Subtitle: workflow tracks show the seed title that themed the lyrics
    # ("The Shape of a Dream · Gauzy and Stated"); everything else shows the
    # prompt name, and workflow-less customs stay "custom".
    subtitle = row.get("theme_title") or prompts.get(row["prompt_id"]) \
        or row.get("workflow_id") or "custom"
    return {
        "id": row["id"],
        "title": row["title"],
        "prompt": subtitle,
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


def _encoder(rt):
    return rt.feeder.encoder if rt.feeder else None


def _parse_at(raw: str | None) -> float | None:
    """Defensive parse of a client-reported timeline position (epoch ms).

    Junk values ("null", "nan", ...) must never 422 an endpoint — they just
    mean "no timeline information, use wall clock".
    """
    if not raw:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


# ------------------------------------------------------------- schedule core

def _schedule_rows(db: DB, cfg, listener_id: str, at_ms: float | None,
                   encoder) -> tuple[str, float | None, list[dict]]:
    """(mode, prog_at, rows) — the aired-tracks timeline, newest first.

    Program mode matches tracks by exact program offset on the HLS PDT
    timeline; if that yields nothing (e.g. an epoch the client has not
    re-synced from yet) we self-heal to clock mode instead of returning an
    empty timeline — the panels must never silently blank out.
    """
    mode, prog_at = "clock", None
    if (at_ms is not None and encoder is not None
            and getattr(encoder, "started_wall", None) is not None):
        try:
            # started_wall is naive UTC; timestamp() would read it as local
            # time, so attach UTC explicitly before converting to epoch.
            start_epoch = encoder.started_wall.replace(tzinfo=timezone.utc).timestamp()
            prog_at = at_ms / 1000.0 - start_epoch
            if abs(prog_at - encoder.program_seconds()) < 6 * 3600:
                mode = "program"
        except (TypeError, ValueError, OSError):
            prog_at = None

    rows: list[dict] = []
    if mode == "program":
        rows = db.query(
            "SELECT p.started_at AS airtime, p.prog_offset AS prog_offset,"
            " p.played_s AS played_s, t.*"
            " FROM play_log p JOIN tracks t ON t.id = p.track_id"
            " WHERE p.epoch = ? AND p.prog_offset IS NOT NULL"
            " AND p.prog_offset <= ? AND t.status != 'deleted'"
            " ORDER BY p.prog_offset DESC LIMIT 11",
            (encoder.epoch, prog_at),
        )
        if not rows:
            mode = "clock"
    if mode == "clock":
        rows = db.query(
            "SELECT p.started_at AS airtime, p.prog_offset AS prog_offset,"
            " p.played_s AS played_s, t.*"
            " FROM play_log p JOIN tracks t ON t.id = p.track_id"
            " WHERE p.started_at <= ? AND t.status != 'deleted'"
            " ORDER BY p.started_at DESC LIMIT 11",
            (iso(utcnow()),),
        )
    return mode, prog_at, rows


def _window_s(r: dict) -> float:
    """How long an airing occupies on the timeline: its remainder when a
    restart resumed it mid-song (played_s), else the track's duration."""
    return r.get("played_s") or r["duration_s"] or 1.0


def _pick_current(mode: str, prog_at: float | None, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    r = rows[0]
    if mode == "program":
        if r["prog_offset"] + _window_s(r) > prog_at:
            return r
    else:
        end = parse_iso(r["airtime"]) + timedelta(seconds=_window_s(r))
        if end >= utcnow():
            return r
    return None


def now_playing(db: DB, cfg, listener_id: str, at_ms: float | None = None,
                encoder=None) -> dict:
    """Compact now-playing summary (used by the admin dashboard)."""
    mode, prog_at, rows = _schedule_rows(db, cfg, listener_id, at_ms, encoder)
    prompts = _prompt_names(db)
    my = library.my_votes(db, listener_id)
    cur = _pick_current(mode, prog_at, rows)
    current = track_public(cur, prompts, my) | {"airtime": cur["airtime"]} if cur else None
    recent = [track_public(r, prompts, my) | {"airtime": r["airtime"]} for r in rows[:10]]
    return {
        "server_time": iso(utcnow()),
        "latency_hint": float(cfg.get("stream.latency_hint", 20.0)),
        "sync": mode,
        "current": current,
        "recent": recent,
    }


def _schedule_payload(db: DB, cfg, listener_id: str, at_ms: float | None,
                      encoder) -> dict:
    """Full schedule for the player client, including each track's
    boundaries on the PDT timeline so song changes are computed locally."""
    mode, prog_at, rows = _schedule_rows(db, cfg, listener_id, at_ms, encoder)
    prompts = _prompt_names(db)
    my = library.my_votes(db, listener_id)
    have_tl = (mode == "program" and encoder is not None
               and getattr(encoder, "started_wall", None) is not None)

    def entry(r: dict) -> dict:
        d = track_public(r, prompts, my) | {"airtime": r["airtime"]}
        if have_tl and r.get("prog_offset") is not None:
            start_ms = (encoder.started_wall.replace(tzinfo=timezone.utc).timestamp()
                        + r["prog_offset"]) * 1000.0
            d["start_pdt_ms"] = round(start_ms)
            d["end_pdt_ms"] = round(start_ms + _window_s(r) * 1000.0)
        else:
            d["start_pdt_ms"] = d["end_pdt_ms"] = None
        return d

    cur = _pick_current(mode, prog_at, rows)
    current = entry(cur) if cur else None
    if current is not None:
        # Lyrics ride along on the current entry only — the player folds
        # them out on each song change; the recent rows stay lean. Art
        # likewise: only the On Air card and the media session need it.
        current["lyrics"] = lyrics_public(cur.get("lyrics_snapshot"))
        current["art"] = art_links(db, cfg, cur["id"])
    return {
        "server_time": iso(utcnow()),
        "server_time_ms": epoch_ms(),
        "sync": mode,
        "epoch": encoder.epoch if encoder is not None else None,
        "playback_paused": bool(db.get_setting("playback_paused")),
        "current": current,
        "recent": [entry(r) for r in rows[:10]],
    }


# ------------------------------------------------------------------ pages

def _site_base(request: Request) -> str:
    """Absolute base URL for og:url/og:image: site.base_url when configured
    (canonical, works even for crawlers arriving via a redirect or wrong
    scheme), else derived from the request (dev/LAN)."""
    rt = request.app.state.rt
    configured = str(rt.cfg.get("site.base_url", "") or "").strip().rstrip("/")
    return configured or str(request.base_url).rstrip("/")


@router.get("/", response_class=HTMLResponse)
async def player(request: Request):
    listener = ensure_listener(request)
    resp = request.app.state.templates.TemplateResponse(
        request, "player.html", {"request": request, "site_base": _site_base(request)})
    if listener[1]:
        _set_cookie(resp, listener[1], request)
    return resp


# ------------------------------------------------------------------ voting

class _VoteLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, key: str, limit: int = VOTE_MAX) -> bool:
        now = time.monotonic()
        start, count = self._buckets.get(key, (now, 0))
        if now - start > VOTE_WINDOW_S:
            start, count = now, 0
        count += 1
        self._buckets[key] = (start, count)
        # Bound memory: id-minting attackers grow the bucket dict forever.
        if len(self._buckets) > 10_000:
            self._buckets = {k: (s, c) for k, (s, c) in self._buckets.items()
                             if now - s <= VOTE_WINDOW_S}
        return count <= limit


_limiter = _VoteLimiter()


def _apply_vote(rt, request: Request, track_id: int, value: int,
                listener: tuple[str, str | None]) -> dict | None:
    if value not in (-1, 0, 1):
        return None
    lid, _ = listener
    if not _limiter.allow(lid):
        return None
    ip = request.client.host if request.client else "?"
    if not _limiter.allow(f"ip:{ip}", limit=IP_VOTE_MAX):
        return None
    row = rt.db.one("SELECT id FROM tracks WHERE id = ? AND status != 'deleted'", (track_id,))
    if not row:
        return None
    return library.apply_vote(rt.db, rt.cfg, track_id, lid, value)


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
        "my_vote": value, "protected": bool(row["protected"]),
        "score": round(wilson_score(row["vote_up"], row["vote_down"]), 3),
    })
    if listener[1]:
        _set_cookie(resp, listener[1], request)
    return resp


# ------------------------------------------------------------------ JSON API

@router.get("/api/schedule")
async def api_schedule(request: Request, at: str | None = None):
    """The aired-tracks timeline for the player client."""
    rt = request.app.state.rt
    listener = ensure_listener(request)
    data = _schedule_payload(rt.db, rt.cfg, listener[0], _parse_at(at), _encoder(rt))
    resp = JSONResponse(data)
    if listener[1]:
        _set_cookie(resp, listener[1], request)
    return resp


@router.get("/api/track/{track_id}/tally")
async def api_tally(request: Request, track_id: int):
    """Tiny per-track vote tally for the slow in-song refresh."""
    rt = request.app.state.rt
    listener = ensure_listener(request)
    row = rt.db.one(
        "SELECT id, vote_up, vote_down, protected FROM tracks"
        " WHERE id = ? AND status != 'deleted'", (track_id,))
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    my = rt.db.one("SELECT value FROM votes WHERE track_id = ? AND listener_id = ?",
                   (track_id, listener[0]))
    resp = JSONResponse({
        "track_id": track_id, "vote_up": row["vote_up"], "vote_down": row["vote_down"],
        "my_vote": my["value"] if my else 0, "protected": bool(row["protected"]),
    })
    if listener[1]:
        _set_cookie(resp, listener[1], request)
    return resp


@router.get("/api/nowplaying")
async def api_nowplaying(request: Request, at: str | None = None):
    rt = request.app.state.rt
    listener = ensure_listener(request)
    data = now_playing(rt.db, rt.cfg, listener[0], at_ms=_parse_at(at),
                       encoder=_encoder(rt))
    resp = JSONResponse(data)
    if listener[1]:
        _set_cookie(resp, listener[1], request)
    return resp


@router.get("/api/track/{track_id}")
async def api_track(request: Request, track_id: int):
    rt = request.app.state.rt
    row = rt.db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    if not row or row["status"] == "deleted":
        return JSONResponse({"error": "not found"}, status_code=404)
    listener = ensure_listener(request)
    # Art links ride here too (not just on the schedule's current entry):
    # the Discord bot's /art command and per-card art button fetch by id.
    return JSONResponse(track_public(row, _prompt_names(rt.db),
                                     library.my_votes(rt.db, listener[0]))
                        | {"lyrics": lyrics_public(row["lyrics_snapshot"]),
                           "art": art_links(rt.db, rt.cfg, track_id)})


# ------------------------------------------------------------------ track art

# URL names map onto disk files with fixed shapes; anything else is a 404
# (the ids are ints and the stem/ext are checked, so no traversal).
ART_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "webp": "image/webp"}


def art_links_for(cfg, track_id: int, art_id: int) -> dict | None:
    """Immutable /art URLs for one stored image, or None when its files
    are gone. Each image owns its URLs forever (the art_id is in the path,
    the files never change), so the immutable Cache-Control — and
    Cloudflare's edge, which caches .png/.jpg by default — can never serve
    a stale pick. Card = the largest thumbnail, icon = the smallest (fall
    back to the master when a size is missing); the px values let the
    player fill Media Session artwork sizes without guessing."""
    files = library.art_file_map(cfg, track_id, art_id)
    if "full" not in files:
        return None
    sizes = sorted((int(k) for k in files if k != "full"))

    def url(key: str) -> str:
        return f"/art/{track_id}/{art_id}/{key}{files[key].suffix}"

    card_key = str(sizes[-1]) if sizes else "full"
    icon_key = str(sizes[0]) if sizes else card_key
    return {
        "master": url("full"),
        "card": url(card_key), "card_px": int(card_key) if card_key != "full" else None,
        "icon": url(icon_key), "icon_px": int(icon_key) if icon_key != "full" else None,
    }


def art_links(db, cfg, track_id: int) -> dict | None:
    """URLs for the track's CURRENT pick (players and previews)."""
    row = library.current_art(db, track_id)
    if row is None:
        return None
    return art_links_for(cfg, track_id, row["id"])


def _art_file(cfg, track_id: int, art_id: int, name: str) -> Path | None:
    stem, dot, ext = name.rpartition(".")
    ext = ext.lower()
    if dot and stem and ext in ART_MEDIA:
        if stem == "full":
            want = f"{art_id}.png"
        elif stem.isdigit():
            want = f"{art_id}_{stem}.{ext}"
        else:
            return None
        p = library.art_dir(cfg, track_id) / want
        return p if p.is_file() else None
    return None


@router.get("/art/{track_id}/{art_id}/{name}")
@router.head("/art/{track_id}/{art_id}/{name}")
async def art_file(request: Request, track_id: int, art_id: int, name: str):
    """Track artwork: one immutable URL per stored image (see
    art_links_for). Long max-age + ETag so browsers and Cloudflare's edge
    cache it; picking a different current image changes the URL the
    schedule hands out, never the bytes behind an old one."""
    p = _art_file(request.app.state.rt.cfg, track_id, art_id, name)
    if p is None:
        return Response(status_code=404)
    return FileResponse(p, media_type=ART_MEDIA[p.suffix.lstrip(".").lower()],
                        headers={
                            "Cache-Control": "public, max-age=31536000, immutable",
                            "ETag": f'"{track_id}-{art_id}-{name}"',
                        })


# ------------------------------------------------------------------ HLS files

_PDT_TAIL = re.compile(r"([+-]\d{2})(\d{2})$")


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
    if rt.telemetry is not None:
        rt.telemetry.heartbeat(request)   # presence = fetched an HLS file
    if p.suffix == ".m3u8":
        # Playlists must never be cached, and their PDT offsets normalized.
        return PlainTextResponse(_fix_pdt_offsets(p.read_text(encoding="utf-8")),
                                 headers={"Cache-Control": "no-store"})
    # Segments are immutable for as long as they exist in the window.
    return FileResponse(p, headers={"Cache-Control": "public, max-age=180"})


# ------------------------------------------------------------------ icons

_STATIC = Path(__file__).resolve().parent / "static"


def _root_icon(fname: str, media_type: str) -> FileResponse:
    # Browsers and legacy iOS request these two from the site root regardless
    # of any <link> tags; everything else is served under /static/.
    return FileResponse(_STATIC / fname, media_type=media_type,
                        headers={"Cache-Control": "public, max-age=86400"})


@router.get("/favicon.ico")
async def favicon_ico():
    return _root_icon("favicon.ico", "image/x-icon")


@router.get("/apple-touch-icon.png")
async def apple_touch_icon():
    return _root_icon("apple-touch-icon.png", "image/png")
