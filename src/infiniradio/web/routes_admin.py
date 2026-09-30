"""Admin panel: dashboard, library, prompts, jobs, settings. Server-rendered
Jinja2 + htmx-free plain forms (full page submissions) to keep it simple."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse

from .. import dj, janitor, library
from ..config import RUNTIME_SETTINGS, runtime_get
from ..util import fmt_duration, human_bytes, iso, utcnow, wilson_score
from ..web.auth import SESSION_COOKIE, LoginThrottle, Sessions, verify_password, stored_password_hash
from ..web.routes_public import now_playing

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin")

_throttle = LoginThrottle()


# ------------------------------------------------------------------ helpers

def _is_admin(request: Request) -> bool:
    sessions: Sessions = request.app.state.sessions
    return sessions.validate(request.cookies.get(SESSION_COOKIE))


def _csrf(request: Request) -> str:
    sessions: Sessions = request.app.state.sessions
    return sessions.csrf_token(request.cookies.get(SESSION_COOKIE))


def _ctx(request: Request, **extra) -> dict:
    return {"request": request, "csrf": _csrf(request), **extra}


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/admin/login", status_code=303)


def _csrf_ok(request: Request, submitted: str) -> bool:
    sessions: Sessions = request.app.state.sessions
    return sessions.check_csrf(request.cookies.get(SESSION_COOKIE), submitted)


def _track_with_meta(db, row: dict) -> dict:
    prompt = db.one("SELECT name FROM prompts WHERE id = ?", (row["prompt_id"],)) \
        if row["prompt_id"] else None
    style = db.one("SELECT name FROM styles WHERE id = ?", (row["style_id"],)) \
        if row.get("style_id") else None
    out = dict(row)
    out["prompt_name"] = prompt["name"] if prompt else (
        "custom" if row["style_text"] else "?")
    out["style_name"] = style["name"] if style else ""
    # Display subtitle (library/aired tables): theme title for workflow
    # tracks, else prompt name, else the workflow slug.
    out["subtitle"] = row.get("theme_title") or out["prompt_name"] \
        or row.get("workflow_id") or "?"
    out["tags"] = library.track_tags(db, row["id"])
    out["score"] = round(wilson_score(row["vote_up"], row["vote_down"]), 3)
    out["duration_fmt"] = fmt_duration(row["duration_s"] or 0)
    out["size_fmt"] = human_bytes(row["size_bytes"])
    out["listens"] = row["listener_count"] or 0
    return out


# ------------------------------------------------------------------ auth

@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    if _is_admin(request):
        return RedirectResponse("/admin", status_code=303)
    return request.app.state.templates.TemplateResponse(
        request, "admin/login.html", {"request": request, "error": None})


@router.post("/login")
async def login_submit(request: Request, username: str = Form(""), password: str = Form("")):
    rt = request.app.state.rt
    key = f"{request.client.host if request.client else '?'}:{username}"
    if _throttle.locked(key):
        return request.app.state.templates.TemplateResponse(
            request, "admin/login.html",
            {"request": request,
             "error": f"Too many attempts; try again in {_throttle.seconds_left(key)}s"},
            status_code=429)

    expected_user = str(rt.cfg.get("admin.username", "admin"))
    stored = stored_password_hash(rt.db, rt.cfg)
    ok = bool(stored) and username == expected_user and verify_password(password, stored)
    if not ok:
        _throttle.record_failure(key)
        return request.app.state.templates.TemplateResponse(
            request, "admin/login.html",
            {"request": request, "error": "Invalid credentials"}, status_code=401)
    _throttle.reset(key)
    sessions: Sessions = request.app.state.sessions
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(SESSION_COOKIE, sessions.issue(), max_age=7 * 24 * 3600,
                    httponly=True, samesite="lax",
                    secure=request.url.scheme == "https")
    return resp


@router.post("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE, secure=request.url.scheme == "https")
    return resp


# ------------------------------------------------------------------ dashboard

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    db, cfg = rt.db, rt.cfg

    quota = janitor.quota_state(db, cfg)
    stats = db.one(
        "SELECT COALESCE(SUM(status = 'ready'), 0) AS ready,"
        " COALESCE(SUM(status = 'generating'), 0) AS generating,"
        " COALESCE(SUM(status = 'failed'), 0) AS failed,"
        " COALESCE(SUM(status = 'deleted'), 0) AS deleted,"
        " COALESCE(SUM(status = 'trashed'), 0) AS trashed,"
        " COALESCE(SUM(hold = 1 AND status = 'ready'), 0) AS held,"
        " COALESCE(SUM(protected = 1 AND status = 'ready'), 0) AS protected,"
        " COALESCE(SUM(CASE WHEN status = 'ready' THEN duration_s ELSE 0 END), 0.0)"
        " AS playtime_s FROM tracks"
    )
    gen_status = await rt.generator.status() if rt.generator else {"backend": "off", "ok": None}
    llm_status = await rt.llm.status() if rt.llm else {"backend": "off", "ok": None}
    jobs = db.query("SELECT j.*, p.name AS prompt_name, st.name AS style_name, pt.title AS parent_title"
                    " FROM generation_jobs j"
                    " LEFT JOIN prompts p ON p.id = j.prompt_id"
                    " LEFT JOIN styles st ON st.id = j.style_id"
                    " LEFT JOIN tracks pt ON pt.id = j.parent_track_id"
                    " ORDER BY j.id DESC LIMIT 12")
    for j in jobs:
        j["age"] = j["created_at"]

    feeder_track = None
    if rt.feeder and rt.feeder.current_track_id:
        row = db.one("SELECT * FROM tracks WHERE id = ?", (rt.feeder.current_track_id,))
        if row:
            feeder_track = row["title"]

    gen_paused, gen_reason = janitor.generation_paused(db, cfg)

    np = now_playing(db, cfg, "admin")
    queue = db.query(
        "SELECT q.id AS qid, q.queued_at, t.id AS track_id, t.title, t.duration_s,"
        " t.theme_title, t.workflow_id, p.name AS prompt_name FROM play_queue q"
        " JOIN tracks t ON t.id = q.track_id"
        " LEFT JOIN prompts p ON p.id = t.prompt_id ORDER BY q.id"
    )
    return request.app.state.templates.TemplateResponse(
        request, "admin/dashboard.html",
        _ctx(request,
             np=np, quota=quota, stats=stats, gen=gen_status, llm=llm_status, jobs=jobs,
             feeder_track=feeder_track, queue=queue,
             gen_manual_paused=bool(db.get_setting("generation_paused")),
             gen_paused=gen_paused, gen_reason=gen_reason,
             playback_paused=bool(db.get_setting("playback_paused")),
             has_generator=rt.generator is not None,
             listeners_last_minute=(rt.telemetry.last_minute_count
                                    if rt.telemetry is not None else 0),
             queued_s=round(rt.feeder.encoder.queued_seconds(), 1) if rt.feeder else 0,
             uptime_s=int((utcnow() - rt.started).total_seconds()),
             usage_fmt=human_bytes(quota["usage"]), quota_fmt=human_bytes(quota["quota"])))


# ------------------------------------------------------------------ listeners

# scale -> (window, bucket seconds, bucket label format) — all UTC. Buckets
# align to unix-epoch boundaries, so hour/6-hour/day divisions land on :00.
LISTENER_SCALES: dict[str, tuple[timedelta, int, str]] = {
    "1h":  (timedelta(hours=1),        60,     "%H:%M"),
    "1d":  (timedelta(hours=24),       5 * 60, "%H:%M"),
    "7d":  (timedelta(hours=24 * 7),   3600,   "%m-%d %H:%M"),
    "30d": (timedelta(hours=24 * 30),  6 * 3600, "%m-%d %H:%M"),
    "1yr": (timedelta(hours=24 * 365), 24 * 3600, "%m-%d"),
}

# SVG plot geometry (viewBox units)
_PLOT_W, _PLOT_H = 960.0, 180.0
_PAD_L, _PAD_T, _PAD_B = 34.0, 8.0, 22.0


def _listener_buckets(db, window: timedelta, bucket_s: int, label_fmt: str) -> list[dict]:
    """Aggregated buckets for the chart, oldest first.

    The plotted value is the time-average over the WHOLE bucket (sum of
    listener-minutes / minutes in the bucket): minutes without a row are
    minutes with zero listeners, and pretending otherwise would inflate
    gappy history. At the 1h scale buckets are single minutes, i.e. raw.
    """
    since = (utcnow() - window).strftime("%Y-%m-%dT%H:%M")
    rows = db.query(
        "SELECT (CAST(strftime('%s', minute) AS INTEGER) / ?) AS b,"
        " SUM(listeners) AS total, MAX(listeners) AS peak"
        " FROM listener_minutes WHERE minute >= ? GROUP BY b ORDER BY b",
        (bucket_s, since))
    minutes = max(1, bucket_s // 60)
    out = []
    for r in rows:
        start = datetime.fromtimestamp(r["b"] * bucket_s, tz=timezone.utc)
        nxt = datetime.fromtimestamp((r["b"] + 1) * bucket_s, tz=timezone.utc)
        out.append({
            "label": start.strftime(label_fmt),
            "value": r["total"] / minutes,          # average over the bucket
            "peak": r["peak"],
            "tip": (f"{start.strftime('%m-%d %H:%M')}–{nxt.strftime('%H:%M')} UTC"
                    f" · avg {r['total'] / minutes:.1f} · peak {r['peak']}"),
        })
    return out


def _layout(buckets: list[dict]) -> tuple[list[dict], float, list[dict]]:
    """Add pixel geometry to buckets; returns (buckets, y_max, y_ticks)."""
    n = len(buckets)
    if not n:
        return [], 0.0, []
    step = _PLOT_W / n
    bar_w = max(1.0, step * 0.8)
    vmax = max(b["value"] for b in buckets) or 1.0
    # nice ceiling: 1/2/5 x 10^k
    mag = 10 ** math.floor(math.log10(vmax)) if vmax > 0 else 1
    y_max = 10 * mag
    for mult in (1, 2, 5, 10):
        y_max = mult * mag
        if vmax <= y_max:
            break
    for i, b in enumerate(buckets):
        b["x"] = round(_PAD_L + i * step + (step - bar_w) / 2, 1)
        b["h"] = round(b["value"] / y_max * _PLOT_H, 1)
        b["y"] = round(_PAD_T + _PLOT_H - b["h"], 1)
        b["w"] = round(bar_w, 1)
    ticks = []
    for frac in (1.0, 0.5):
        ticks.append({"y": round(_PAD_T + _PLOT_H * (1 - frac), 1),
                      "v": _fmt_count(y_max * frac)})
    return buckets, y_max, ticks


def _fmt_count(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


@router.get("/listeners", response_class=HTMLResponse)
async def listeners_page(request: Request, scale: str = "7d"):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    if scale not in LISTENER_SCALES:
        scale = "7d"
    window, bucket_s, label_fmt = LISTENER_SCALES[scale]
    buckets, y_max, ticks = _layout(
        _listener_buckets(rt.db, window, bucket_s, label_fmt))
    # a handful of x-axis labels, evenly spaced
    xlabels = []
    if buckets:
        stride = max(1, len(buckets) // 7)
        for i in range(0, len(buckets), stride):
            xlabels.append({"x": round(buckets[i]["x"], 1),
                            "text": buckets[i]["label"]})
    stats = {
        "peak": max((b["peak"] for b in buckets), default=0),
        # average audience during minutes someone was actually listening
        "avg": (sum(b["value"] for b in buckets) / len(buckets)) if buckets else 0.0,
        "buckets": len(buckets),
    }
    return request.app.state.templates.TemplateResponse(
        request, "admin/listeners.html",
        _ctx(request, scale=scale, scales=list(LISTENER_SCALES),
             buckets=buckets, y_max=_fmt_count(y_max), y_ticks=ticks,
             xlabels=xlabels, stats=stats,
             now_listeners=(rt.telemetry.last_minute_count
                            if rt.telemetry is not None else 0)))


# ------------------------------------------------------------------ library

@router.get("/library", response_class=HTMLResponse)
async def library_page(request: Request, status: str = "ready"):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    rows = rt.db.query(
        "SELECT * FROM tracks WHERE status = ? ORDER BY id DESC LIMIT 500", (status,)
    )
    tracks = [_track_with_meta(rt.db, r) for r in rows]
    return request.app.state.templates.TemplateResponse(
        request, "admin/library.html",
        _ctx(request, tracks=tracks, status=status))


@router.get("/aired", response_class=HTMLResponse)
async def aired_page(request: Request, n: int = 100):
    """Airing history: every play_log entry, newest first.

    Unlike the library (one row per track), this is one row per airing —
    a track played twice appears twice, exactly as it went out on air.
    """
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    n = max(10, min(n, 500))
    rows = rt.db.query(
        "SELECT p.started_at AS aired_at, t.* FROM play_log p"
        " JOIN tracks t ON t.id = p.track_id"
        " ORDER BY p.id DESC LIMIT ?", (n,))
    aired = []
    for r in rows:
        entry = _track_with_meta(rt.db, r)
        entry["aired_at"] = r["aired_at"]
        aired.append(entry)
    return request.app.state.templates.TemplateResponse(
        request, "admin/aired.html", _ctx(request, aired=aired, n=n))


@router.get("/track/{track_id}", response_class=HTMLResponse)
async def track_page(request: Request, track_id: int):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    row = rt.db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    if not row:
        return _login_redirect()
    db = rt.db
    track = _track_with_meta(db, row)
    parent = db.one("SELECT * FROM tracks WHERE id = ?", (row["parent_track_id"],)) \
        if row["parent_track_id"] else None
    children = db.query("SELECT * FROM tracks WHERE parent_track_id = ?", (track_id,))
    votes = db.query("SELECT * FROM votes WHERE track_id = ? ORDER BY updated_at DESC LIMIT 50",
                     (track_id,))
    plays = db.query("SELECT started_at FROM play_log WHERE track_id = ?"
                     " ORDER BY id DESC LIMIT 20", (track_id,))
    prompts = db.query("SELECT id, name FROM prompts WHERE enabled = 1 ORDER BY name")
    gen_job = db.one("SELECT id, status FROM generation_jobs WHERE track_id = ?"
                     " ORDER BY id DESC LIMIT 1", (track_id,))
    return request.app.state.templates.TemplateResponse(
        request, "admin/track.html",
        _ctx(request, t=track, parent=parent,
             children=[_track_with_meta(db, c) for c in children],
             votes=votes, plays=plays, prompts=prompts, gen_job=gen_job))


@router.get("/track/{track_id}/audio")
async def track_audio(request: Request, track_id: int):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    row = rt.db.one("SELECT playout_path FROM tracks WHERE id = ?", (track_id,))
    if not row or not row["playout_path"]:
        return PlainTextResponse("no audio", status_code=404)
    p: Path = rt.cfg.path("library.data_dir") / "music" / row["playout_path"]
    if not p.is_file():
        return PlainTextResponse("no audio", status_code=404)
    return FileResponse(p, media_type="audio/mp4")


@router.post("/track/{track_id}/protect")
async def track_protect(request: Request, track_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT protected FROM tracks WHERE id = ?", (track_id,))
    if row:
        rt.db.execute("UPDATE tracks SET protected = ? WHERE id = ?",
                      (0 if row["protected"] else 1, track_id))
    return RedirectResponse(f"/admin/track/{track_id}", status_code=303)


@router.post("/track/{track_id}/queue_next")
async def track_queue_next(request: Request, track_id: int, csrf: str = Form("")):
    """Manually override the DJ: air this track next."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT id FROM tracks WHERE id = ? AND status = 'ready'", (track_id,))
    if not row:
        return PlainTextResponse("track is not ready", status_code=400)
    rt.db.execute("INSERT INTO play_queue (track_id, queued_at) VALUES (?, ?)",
                  (track_id, iso(utcnow())))
    return RedirectResponse(_back(request, "/admin/library"), status_code=303)


@router.post("/queue/{qid}/remove")
async def queue_remove(request: Request, qid: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.execute("DELETE FROM play_queue WHERE id = ?", (qid,))
    return RedirectResponse(_back(request, "/admin"), status_code=303)


@router.post("/skip")
async def skip_current(request: Request, csrf: str = Form("")):
    """Stop feeding the current track and flush the buffered audio."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    if rt.feeder and rt.feeder.request_skip():
        log.info("skip requested from admin panel")
    return RedirectResponse(_back(request, "/admin"), status_code=303)


@router.post("/generation/pause")
async def generation_pause(request: Request, csrf: str = Form("")):
    """Stop automatic commissioning (queued/running jobs still finish)."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.set_setting("generation_paused", True)
    log.info("generation paused from admin panel")
    return RedirectResponse(_back(request, "/admin"), status_code=303)


@router.post("/generation/resume")
async def generation_resume(request: Request, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.set_setting("generation_paused", False)
    log.info("generation resumed from admin panel")
    return RedirectResponse(_back(request, "/admin"), status_code=303)


@router.post("/playback/pause")
async def playback_pause(request: Request, csrf: str = Form("")):
    """Cut to silence now and stay silent until resumed.

    Setting the flag alone would only take effect at the next track
    boundary (and up to MAX_LEAD_S of audio is already buffered), so we
    also skip the current track out of the buffer — the same flush a
    manual skip performs. The encoder keeps running; listeners hear
    silence instead of a stalled stream and stay connected.
    """
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.set_setting("playback_paused", True)
    if rt.feeder and rt.feeder.request_skip():
        log.info("playback paused from admin panel (cut current track)")
    else:
        log.info("playback paused from admin panel (nothing airing)")
    return RedirectResponse(_back(request, "/admin"), status_code=303)


@router.post("/playback/resume")
async def playback_resume(request: Request, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.set_setting("playback_paused", False)
    log.info("playback resumed from admin panel")
    return RedirectResponse(_back(request, "/admin"), status_code=303)


def _back(request: Request, default: str) -> str:
    ref = request.headers.get("referer") or ""
    return ref if ref.startswith("/") or ref.startswith(str(request.base_url)) else default


@router.post("/track/{track_id}/delete")
async def track_delete(request: Request, track_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    # Never queued again, regardless of what follows. Trashing is safe even
    # for the track currently on air: it finishes its airing, the janitor
    # only hard-deletes after the buffer expires and nothing is airing.
    rt.db.execute("DELETE FROM play_queue WHERE track_id = ?", (track_id,))
    library.schedule_delete(rt.db, rt.cfg, track_id)
    return RedirectResponse(_back(request, "/admin/library"), status_code=303)


@router.post("/track/{track_id}/undelete")
async def track_undelete(request: Request, track_id: int, csrf: str = Form("")):
    """Admin rescue while a condemned track is still in the trash."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    library.undelete_track(rt.db, track_id)
    return RedirectResponse(_back(request, "/admin/library?status=trashed"), status_code=303)


@router.post("/track/{track_id}/delete_now")
async def track_delete_now(request: Request, track_id: int, csrf: str = Form("")):
    """Skip the buffer: empty the trash for this track immediately — unless
    it is on air right now, in which case the deadline is expired instead
    and the janitor collects it the moment the airing ends."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    airing = rt.feeder is not None and track_id in rt.feeder.active_ids
    if airing:
        rt.db.execute("UPDATE tracks SET delete_after = ? WHERE id = ? AND status = 'trashed'",
                      (iso(utcnow()), track_id))
    else:
        library.delete_track(rt.db, rt.cfg, track_id)
    return RedirectResponse(_back(request, "/admin/library?status=trashed"), status_code=303)


@router.post("/track/{track_id}/hold")
async def track_hold(request: Request, track_id: int, csrf: str = Form("")):
    """Toggle 'never auto-air this track' — the DJ's variation and selection
    picks skip held tracks; an admin's explicit queueing (air next, restart
    resume of an in-flight airing) is deliberate and still works. Use the
    queue's own remove button to cancel a queued item."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT hold FROM tracks WHERE id = ?", (track_id,))
    if row:
        rt.db.execute("UPDATE tracks SET hold = ? WHERE id = ?",
                      (0 if row["hold"] else 1, track_id))
    return RedirectResponse(f"/admin/track/{track_id}", status_code=303)


@router.post("/track/{track_id}/rename")
async def track_rename(request: Request, track_id: int, title: str = Form(""), csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    rt.db.execute("UPDATE tracks SET title = ? WHERE id = ?",
                  (title.strip()[:200] or f"track #{track_id}", track_id))
    return RedirectResponse(f"/admin/track/{track_id}", status_code=303)


@router.post("/track/{track_id}/variation")
async def track_variation(request: Request, track_id: int,
                          style_prompt_id: int = Form(0), custom_style: str = Form(""),
                          seed: int = Form(0), csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT id, abc_text FROM tracks WHERE id = ? AND status = 'ready'",
                    (track_id,))
    if not row or not row["abc_text"]:
        return PlainTextResponse("track has no ABC score to vary", status_code=400)
    dj.enqueue_job(
        rt.db,
        prompt_id=style_prompt_id or None,
        parent_track_id=track_id,
        style_override=custom_style.strip() or None,
        seed=seed or None,
    )
    return RedirectResponse("/admin/jobs", status_code=303)


# ------------------------------------------------------------------ prompts

@router.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    prompts = []
    for p in rt.db.query("SELECT * FROM prompts ORDER BY name"):
        stats = dj.prompt_air_stats(rt.db, p["id"])
        p = dict(p)
        p["tracks"] = rt.db.one(
            "SELECT COUNT(*) AS n FROM tracks WHERE prompt_id = ? AND status != 'deleted'",
            (p["id"],))["n"]
        p["weight"] = round(dj.prompt_weight(rt.db, rt.cfg, p["id"]), 3)
        p["avg_score"] = round(stats["avg_score"], 3)
        p["aired"] = stats["aired"]
        p["tags"] = library.prompt_tags(rt.db, p["id"])
        prompts.append(p)
    tags = []
    for t in rt.db.query("SELECT id, name FROM tags ORDER BY name"):
        stats = dj.tag_air_stats(rt.db, t["id"])
        tracks = rt.db.one(
            "SELECT COUNT(*) AS n FROM track_tags WHERE tag_id = ?", (t["id"],))["n"]
        tags.append({"name": t["name"], "aired": stats["aired"],
                     "avg_score": round(stats["avg_score"], 3), "tracks": tracks})
    workflows = []
    admin_off = set(rt.db.get_setting("workflows_disabled") or [])
    for w in getattr(rt, "workflows", {}).values():
        stats = dj.workflow_air_stats(rt.db, w.name)
        tracks = rt.db.one(
            "SELECT COUNT(*) AS n FROM tracks WHERE workflow_id = ? AND status != 'deleted'",
            (w.name,))["n"]
        on = w.enabled and w.name not in admin_off
        weight = (dj.workflow_weight(rt.db, rt.cfg, w.name)
                  if rt.llm is not None and on else None)
        workflows.append({"name": w.name, "description": w.description,
                          "enabled": on, "file_enabled": w.enabled,
                          "stages": len(w.stages),
                          "tags": list(w.static_tags), "tracks": tracks,
                          "aired": stats["aired"],
                          "avg_score": round(stats["avg_score"], 3),
                          "weight": round(weight, 3) if weight is not None else None})
    return request.app.state.templates.TemplateResponse(
        request, "admin/prompts.html",
        _ctx(request, prompts=prompts, tags=tags, workflows=workflows,
             has_llm=rt.llm is not None))


@router.post("/workflows/{workflow_id}/toggle")
async def workflow_toggle(request: Request, workflow_id: str, csrf: str = Form("")):
    """Enable/disable a lyrics workflow at runtime (settings table — no
    file edits, no restart; the DJ stops commissioning it immediately,
    queued jobs still finish)."""
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    if workflow_id not in getattr(rt, "workflows", {}):
        return PlainTextResponse("no such workflow", status_code=404)
    disabled = list(rt.db.get_setting("workflows_disabled") or [])
    if workflow_id in disabled:
        disabled.remove(workflow_id)
    else:
        disabled.append(workflow_id)
    rt.db.set_setting("workflows_disabled", disabled)
    return RedirectResponse(_back(request, "/admin/prompts"), status_code=303)


def _prompt_form(request: Request, prompt_id: int | None):
    rt = request.app.state.rt
    row = rt.db.one("SELECT * FROM prompts WHERE id = ?", (prompt_id,)) if prompt_id else None
    return row


@router.get("/prompts/edit", response_class=HTMLResponse)
async def prompt_new(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    return request.app.state.templates.TemplateResponse(
        request, "admin/prompt_edit.html",
        _ctx(request, p=None, error=None,
             action="/admin/prompts/edit"))


@router.get("/prompts/edit/{prompt_id}", response_class=HTMLResponse)
async def prompt_edit(request: Request, prompt_id: int):
    if not _is_admin(request):
        return _login_redirect()
    row = _prompt_form(request, prompt_id)
    if not row:
        return RedirectResponse("/admin/prompts", status_code=303)
    return request.app.state.templates.TemplateResponse(
        request, "admin/prompt_edit.html",
        _ctx(request, p=row, error=None,
             action=f"/admin/prompts/edit/{prompt_id}"))


def _upsert_prompt(rt, prompt_id: int | None, name: str, style: str, song: str,
                   cot: str, enabled: bool) -> str | None:
    if not name.strip() or not style.strip():
        return "Name and style are required."
    try:
        if prompt_id:
            rt.db.execute(
                "UPDATE prompts SET name = ?, style_text = ?, song_text = ?, cot_mode = ?,"
                " enabled = ? WHERE id = ?",
                (name.strip(), style, song, cot, int(enabled), prompt_id))
        else:
            rt.db.execute(
                "INSERT INTO prompts (name, style_text, song_text, cot_mode, enabled, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (name.strip(), style, song, cot, int(enabled), iso(utcnow())))
    except Exception:
        return "Could not save (duplicate name?)."
    return None


@router.post("/prompts/edit")
async def prompt_create(request: Request, name: str = Form(""), style: str = Form(""),
                        song: str = Form(""), cot: str = Form("full"),
                        enabled: bool = Form(False)):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    err = _upsert_prompt(rt, None, name, style, song, cot, enabled)
    if err:
        return request.app.state.templates.TemplateResponse(
            request, "admin/prompt_edit.html",
            _ctx(request, p=None, error=err, action="/admin/prompts/edit"), status_code=400)
    return RedirectResponse("/admin/prompts", status_code=303)


@router.post("/prompts/edit/{prompt_id}")
async def prompt_update(request: Request, prompt_id: int, name: str = Form(""),
                        style: str = Form(""), song: str = Form(""),
                        cot: str = Form("full"), enabled: bool = Form(False),
                        csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    err = _upsert_prompt(rt, prompt_id, name, style, song, cot, enabled)
    if err:
        row = _prompt_form(request, prompt_id)
        return request.app.state.templates.TemplateResponse(
            request, "admin/prompt_edit.html",
            _ctx(request, p=row, error=err, action=f"/admin/prompts/edit/{prompt_id}"),
            status_code=400)
    return RedirectResponse("/admin/prompts", status_code=303)


@router.post("/prompts/{prompt_id}/toggle")
async def prompt_toggle(request: Request, prompt_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT enabled FROM prompts WHERE id = ?", (prompt_id,))
    if row:
        rt.db.execute("UPDATE prompts SET enabled = ? WHERE id = ?",
                      (0 if row["enabled"] else 1, prompt_id))
    return RedirectResponse("/admin/prompts", status_code=303)


@router.post("/prompts/{prompt_id}/delete")
async def prompt_delete(request: Request, prompt_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    used = rt.db.one("SELECT COUNT(*) AS n FROM tracks WHERE prompt_id = ?", (prompt_id,))["n"]
    if used:
        return PlainTextResponse(
            f"prompt is referenced by {used} track(s); disable it instead", status_code=409)
    rt.db.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    return RedirectResponse("/admin/prompts", status_code=303)


@router.post("/prompts/import")
async def prompts_import(request: Request, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    library.import_prompts(rt.db, rt.cfg)
    return RedirectResponse("/admin/prompts", status_code=303)


@router.get("/prompts/export")
async def prompts_export(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    lines = ["# infiniradio prompt library export\n"]
    for p in rt.db.query("SELECT * FROM prompts ORDER BY id"):
        import json as _json
        lines.append(f"# id {p['id']}\n[{p['name']}]\n"
                     f"style = {_json.dumps(p['style_text'])}\n"
                     f"song = {_json.dumps(p['song_text'])}\n"
                     f"cot = {_json.dumps(p['cot_mode'])}\n"
                     f"enabled = {bool(p['enabled'])}\n\n")
    return PlainTextResponse("".join(lines), media_type="text/plain")


# ------------------------------------------------------------------ styles

@router.get("/styles", response_class=HTMLResponse)
async def styles_page(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    styles = []
    for s in rt.db.query("SELECT * FROM styles ORDER BY name"):
        stats = dj.style_air_stats(rt.db, s["id"])
        s = dict(s)
        s["tracks"] = rt.db.one(
            "SELECT COUNT(*) AS n FROM tracks WHERE style_id = ? AND status != 'deleted'",
            (s["id"],))["n"]
        s["weight"] = round(dj.style_weight(rt.db, rt.cfg, s["id"]), 3)
        s["avg_score"] = round(stats["avg_score"], 3)
        s["aired"] = stats["aired"]
        s["tags"] = library.style_tags(rt.db, s["id"])
        styles.append(s)
    return request.app.state.templates.TemplateResponse(
        request, "admin/styles.html", _ctx(request, styles=styles))


def _style_form(request: Request, style_id: int | None):
    rt = request.app.state.rt
    return rt.db.one("SELECT * FROM styles WHERE id = ?", (style_id,)) if style_id else None


@router.get("/styles/edit", response_class=HTMLResponse)
async def style_new(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    return request.app.state.templates.TemplateResponse(
        request, "admin/style_edit.html",
        _ctx(request, s=None, definition=_render_style_definition({}, {}), error=None,
             action="/admin/styles/edit"))


@router.get("/styles/edit/{style_id}", response_class=HTMLResponse)
async def style_edit(request: Request, style_id: int):
    if not _is_admin(request):
        return _login_redirect()
    row = _style_form(request, style_id)
    if not row:
        return RedirectResponse("/admin/styles", status_code=303)
    return request.app.state.templates.TemplateResponse(
        request, "admin/style_edit.html",
        _ctx(request, s=row, definition=_render_style_definition_row(row), error=None,
             action=f"/admin/styles/edit/{style_id}"))


def _render_style_definition(vars_: dict, weighted: dict) -> str:
    from .. import styles as styles_mod

    return styles_mod.render_definition(vars_, weighted)


def _render_style_definition_row(row: dict) -> str:
    import json as _json

    return _render_style_definition(
        _json.loads(row["vars_json"] or "{}"),
        {k: [(w, v) for w, v in entries]
         for k, entries in _json.loads(row["weighted_json"] or "{}").items()})


def _upsert_style(rt, style_id: int | None, name: str, description: str,
                  tags: str, definition: str, enabled: bool) -> str | None:
    from .. import styles as styles_mod

    if not name.strip():
        return "Name is required."
    try:
        vars_, weighted = styles_mod.parse_definition(definition, "definition")
    except styles_mod.StyleError as exc:
        return str(exc)
    vars_json = json.dumps(vars_, ensure_ascii=False)
    weighted_json = json.dumps({k: [[w, v] for w, v in entries]
                                for k, entries in weighted.items()},
                               ensure_ascii=False)
    try:
        if style_id:
            rt.db.execute(
                "UPDATE styles SET name = ?, description = ?, vars_json = ?,"
                " weighted_json = ?, enabled = ? WHERE id = ?",
                (name.strip(), description, vars_json, weighted_json,
                 int(enabled), style_id))
        else:
            style_id = rt.db.execute(
                "INSERT INTO styles (name, description, vars_json, weighted_json,"
                " enabled, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (name.strip(), description, vars_json, weighted_json,
                 int(enabled), iso(utcnow())))
    except Exception:
        return "Could not save (duplicate name?)."
    tag_names = [t for t in re.split(r"[,\s]+", tags) if t]
    if tag_names:
        library.apply_style_tags(rt.db, rt.cfg, style_id, tag_names)
    return None


@router.post("/styles/edit")
async def style_create(request: Request, name: str = Form(""),
                       description: str = Form(""), tags: str = Form(""),
                       definition: str = Form(""), enabled: bool = Form(False)):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    err = _upsert_style(rt, None, name, description, tags, definition, enabled)
    if err:
        return request.app.state.templates.TemplateResponse(
            request, "admin/style_edit.html",
            _ctx(request, s=None, definition=definition, error=err,
                 action="/admin/styles/edit"), status_code=400)
    return RedirectResponse("/admin/styles", status_code=303)


@router.post("/styles/edit/{style_id}")
async def style_update(request: Request, style_id: int, name: str = Form(""),
                       description: str = Form(""), tags: str = Form(""),
                       definition: str = Form(""), enabled: bool = Form(False),
                       csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    err = _upsert_style(rt, style_id, name, description, tags, definition, enabled)
    if err:
        row = _style_form(request, style_id)
        return request.app.state.templates.TemplateResponse(
            request, "admin/style_edit.html",
            _ctx(request, s=row, definition=definition, error=err,
                 action=f"/admin/styles/edit/{style_id}"), status_code=400)
    return RedirectResponse("/admin/styles", status_code=303)


@router.post("/styles/{style_id}/toggle")
async def style_toggle(request: Request, style_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    row = rt.db.one("SELECT enabled FROM styles WHERE id = ?", (style_id,))
    if row:
        rt.db.execute("UPDATE styles SET enabled = ? WHERE id = ?",
                      (0 if row["enabled"] else 1, style_id))
    return RedirectResponse("/admin/styles", status_code=303)


@router.post("/styles/{style_id}/delete")
async def style_delete(request: Request, style_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    used = rt.db.one("SELECT COUNT(*) AS n FROM tracks WHERE style_id = ?",
                     (style_id,))["n"]
    if used:
        return PlainTextResponse(
            f"style is referenced by {used} track(s); disable it instead", status_code=409)
    rt.db.execute("DELETE FROM style_tags WHERE style_id = ?", (style_id,))
    rt.db.execute("UPDATE generation_jobs SET style_id = NULL"
                  " WHERE style_id = ? AND status IN ('queued', 'cancelled')", (style_id,))
    rt.db.execute("DELETE FROM styles WHERE id = ?", (style_id,))
    return RedirectResponse("/admin/styles", status_code=303)


@router.post("/styles/import")
async def styles_import(request: Request, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    library.import_styles(rt.db, rt.cfg, only_new=False)
    return RedirectResponse("/admin/styles", status_code=303)


@router.get("/styles/export")
async def styles_export(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    lines = ["# infiniradio style library export\n"]
    for s in rt.db.query("SELECT * FROM styles ORDER BY id"):
        lines.append(f"# id {s['id']}  used {s['times_used']}x\n[{s['name']}]\n")
        lines.append(f"description = {json.dumps(s['description'])}\n")
        lines.append(f"enabled = {bool(s['enabled'])}\n")
        lines.append(_render_style_definition_row(s))
        tag_names = library.style_tags(rt.db, s["id"])
        if tag_names:
            lines.append(f"tags = {json.dumps(tag_names)}\n")
        lines.append("\n")
    return PlainTextResponse("".join(lines), media_type="text/plain")


# ------------------------------------------------------------------ jobs

@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    jobs = rt.db.query(
        "SELECT j.*, p.name AS prompt_name, st.name AS style_name, pt.title AS parent_title"
        " FROM generation_jobs j"
        " LEFT JOIN prompts p ON p.id = j.prompt_id"
        " LEFT JOIN styles st ON st.id = j.style_id"
        " LEFT JOIN tracks pt ON pt.id = j.parent_track_id"
        " ORDER BY j.id DESC LIMIT 200")
    for j in jobs:
        j["stage_progress"] = _stage_progress(rt, j)
    return request.app.state.templates.TemplateResponse(
        request, "admin/jobs.html", _ctx(request, jobs=jobs))


def _stage_progress(rt, job: dict) -> str:
    """'4/7 · style' — completed/total stages of a job's lyrics phase, read
    from the live progress column (persisted as each stage completes)."""
    import json as _json

    if job.get("phase") != "lyrics" or not job.get("workflow_id"):
        return ""
    try:
        progress = _json.loads(job.get("progress") or "{}")
    except _json.JSONDecodeError:
        return ""
    completed = progress.get("completed") or []
    spec = getattr(rt, "workflows", {}).get(job["workflow_id"])
    total = len(spec.stages) if spec else None
    label = f"{len(completed)}/{total}" if total else f"{len(completed)} stages done"
    return label + (f" · {completed[-1]}" if completed else "")


def _between(later: str | None, earlier: str | None) -> str | None:
    """'M:SS' between two ISO stamps; None when either side is missing."""
    if not later or not earlier:
        return None
    try:
        from ..util import parse_iso
        secs = max(0.0, (parse_iso(later) - parse_iso(earlier)).total_seconds())
    except ValueError:
        return None
    return fmt_duration(secs)


def _job_timings(job: dict) -> list[tuple[str, str | None]]:
    """The job's lifetime broken into waiting / lyrics / waiting / rendering
    components from the phase timestamps (set at each claim, so they
    describe the most recent run of each phase). Running phases show the
    elapsed-so-far; jobs from before the timestamps existed show nothing."""
    rows: list[tuple[str, str | None]] = []
    now = iso(utcnow())
    running = job.get("status") == "running"
    is_workflow = bool(job.get("workflow_id"))

    if is_workflow and job.get("lyrics_started_at"):
        rows.append(("waiting for lyrics",
                     _between(job["lyrics_started_at"], job.get("created_at"))))
        if job.get("lyrics_finished_at"):
            rows.append(("lyrics", _between(job["lyrics_finished_at"],
                                            job["lyrics_started_at"])))
        elif running and job.get("phase") == "lyrics":
            rows.append(("lyrics (in progress)", _between(now, job["lyrics_started_at"])))
        if job.get("lyrics_finished_at") and job.get("render_started_at"):
            rows.append(("waiting for render",
                         _between(job["render_started_at"], job["lyrics_finished_at"])))
    if job.get("render_started_at"):
        end = job.get("finished_at") or (now if running else None)
        value = _between(end, job["render_started_at"])
        if value:
            if not is_workflow:
                rows.append(("waiting", _between(job["render_started_at"],
                                                 job.get("created_at"))))
            rows.append(("rendering" if job.get("finished_at")
                         else "rendering (in progress)", value))
    total_end = job.get("finished_at") or (now if running else None)
    if total_end and rows:
        rows.append(("total", _between(total_end, job.get("created_at"))))
    return rows


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: int):
    """Full job view: phase timings + the workflow's per-stage LLM trace."""
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    job = rt.db.one(
        "SELECT j.*, p.name AS prompt_name, st.name AS style_name, pt.title AS parent_title"
        " FROM generation_jobs j"
        " LEFT JOIN prompts p ON p.id = j.prompt_id"
        " LEFT JOIN styles st ON st.id = j.style_id"
        " LEFT JOIN tracks pt ON pt.id = j.parent_track_id"
        " WHERE j.id = ?", (job_id,))
    if not job:
        return _login_redirect()
    stages = rt.db.query(
        "SELECT * FROM job_stage_log WHERE job_id = ? ORDER BY id", (job_id,))
    import json as _json

    for s in stages:
        for dst, src in (("vars", "vars_json"), ("gate", "gate_json"), ("usage", "usage_json")):
            try:
                s[dst] = _json.loads(s[src] or ("{}" if dst != "gate" else "[]"))
            except _json.JSONDecodeError:
                s[dst] = {} if dst != "gate" else []
    outputs = {}
    if job.get("progress"):
        try:
            outputs = (_json.loads(job["progress"]) or {}).get("outputs") or {}
        except _json.JSONDecodeError:
            outputs = {}
    llm_s = sum(s["duration_s"] or 0.0 for s in stages)
    llm_conts = sum(int((s.get("usage") or {}).get("continuations") or 0) for s in stages)
    workflow = rt.workflows.get(job["workflow_id"]) if job.get("workflow_id") else None
    return request.app.state.templates.TemplateResponse(
        request, "admin/job.html",
        _ctx(request, j=job, stages=stages, outputs=outputs, workflow=workflow,
             stage_progress=_stage_progress(rt, job),
             timings=_job_timings(job),
             llm_summary={"calls": len(stages), "seconds": round(llm_s, 1),
                          "continuations": llm_conts}))


@router.post("/jobs/{job_id}/retry")
async def job_retry(request: Request, job_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    job = rt.db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    if job and job["status"] == "failed":
        # Drop the failed placeholder row (it never produced audio) so
        # retries don't accumulate dead rows; the runner creates a fresh
        # track when the job restarts. Clear the FK before deleting.
        if job["track_id"]:
            t = rt.db.one("SELECT status FROM tracks WHERE id = ?", (job["track_id"],))
            if t and t["status"] == "failed":
                rt.db.execute("UPDATE generation_jobs SET track_id = NULL"
                              " WHERE id = ?", (job_id,))
                rt.db.execute("DELETE FROM tracks WHERE id = ? AND status = 'failed'",
                              (job["track_id"],))
        rt.db.execute("UPDATE generation_jobs SET status = 'queued', error = NULL,"
                      " finished_at = NULL"
                      " WHERE id = ? AND status = 'failed'", (job_id,))
    return RedirectResponse("/admin/jobs", status_code=303)


@router.post("/jobs/{job_id}/cancel")
async def job_cancel(request: Request, job_id: int, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    job = rt.db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    if job and job["status"] == "queued":
        rt.db.execute("UPDATE generation_jobs SET status = 'cancelled',"
                      " finished_at = ? WHERE id = ?", (iso(utcnow()), job_id))
    elif job and job["status"] == "running":
        if rt.generator and job.get("phase") != "lyrics":
            await rt.generator.interrupt()
        # A lyrics-phase job needs no backend interrupt; its runner notices
        # the status change between stages/attempts and stops.
        rt.db.execute("UPDATE generation_jobs SET status = 'cancelled',"
                      " finished_at = ? WHERE id = ?", (iso(utcnow()), job_id))
        # The placeholder track would otherwise count as "generating"
        # forever; fail_track only touches it if it is still generating, so
        # a generation that completed concurrently stays ready.
        if job["track_id"]:
            library.fail_track(rt.db, job["track_id"])
        log.info("cancelled running job %s (track %s finalized)",
                 job_id, job["track_id"])
    return RedirectResponse("/admin/jobs", status_code=303)


@router.post("/cleanup")
async def cleanup_now(request: Request, csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    if rt.generator:
        await rt.generator.cleanup_now()
    return RedirectResponse("/admin", status_code=303)


# ------------------------------------------------------------------ generate

@router.get("/generate", response_class=HTMLResponse)
async def generate_form(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    prompts = rt.db.query("SELECT * FROM prompts WHERE enabled = 1 ORDER BY name")
    workflows = dj.enabled_workflows(rt)
    styles = rt.db.query("SELECT id, name, description FROM styles WHERE enabled = 1"
                         " ORDER BY name") if rt.llm is not None else []
    return request.app.state.templates.TemplateResponse(
        request, "admin/generate.html",
        _ctx(request, prompts=prompts, workflows=workflows, styles=styles,
             has_llm=rt.llm is not None))


@router.post("/generate")
async def generate_submit(request: Request, prompt_id: int = Form(0),
                          workflow_id: str = Form(""), style_id: int = Form(0),
                          custom_name: str = Form(""), custom_style: str = Form(""),
                          custom_song: str = Form("[Intro]\n\n[Instrumental]\n\n[Outro]"),
                          csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    workflow = None
    if workflow_id.strip():
        workflow = getattr(rt, "workflows", {}).get(workflow_id.strip())
        if workflow is None or rt.llm is None:
            return PlainTextResponse("no such lyrics workflow (or LLM disabled)",
                                     status_code=400)
    if workflow is not None:
        # No explicit style chosen -> let the DJ's weighted pick decide.
        style = None
        if style_id:
            style = rt.db.one("SELECT id FROM styles WHERE id = ? AND enabled = 1",
                              (style_id,))
        elif workflow.styled:
            style = dj.pick_style(rt)
        dj.enqueue_job(rt.db, workflow_id=workflow.name,
                       style_id=style["id"] if style else None)
    elif prompt_id:
        prompt = rt.db.one("SELECT id FROM prompts WHERE id = ?", (prompt_id,))
        if prompt:
            dj.enqueue_job(rt.db, prompt_id=prompt_id)
    elif custom_style.strip():
        # Ad-hoc style: store it as a disabled prompt so it stays editable
        # and referenced by its tracks, but out of the automatic bandit.
        name = custom_name.strip() or f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        rt.db.execute(
            "INSERT INTO prompts (name, style_text, song_text, cot_mode, enabled, created_at)"
            " VALUES (?, ?, ?, 'full', 0, ?)",
            (name, custom_style.strip(), custom_song, iso(utcnow())))
        pid = rt.db.one("SELECT id FROM prompts WHERE name = ?", (name,))["id"]
        dj.enqueue_job(rt.db, prompt_id=pid)
    return RedirectResponse("/admin/jobs", status_code=303)


# ------------------------------------------------------------------ settings

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    items = []
    for key, (default, cast, label, help_) in RUNTIME_SETTINGS.items():
        items.append({"key": key, "value": runtime_get(rt.cfg, rt.db, key),
                      "label": label, "help": help_})
    return request.app.state.templates.TemplateResponse(
        request, "admin/settings.html", _ctx(request, items=items, saved=False))


@router.post("/settings")
async def settings_save(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    form = await request.form()
    if not _csrf_ok(request, str(form.get("csrf", ""))):
        return PlainTextResponse("forbidden", status_code=403)
    for key, (default, cast, _l, _h) in RUNTIME_SETTINGS.items():
        raw = form.get(key)
        if raw is None:
            continue
        try:
            rt.db.set_setting(key, cast(str(raw)))
        except (TypeError, ValueError):
            pass
    return RedirectResponse("/admin/settings", status_code=303)
