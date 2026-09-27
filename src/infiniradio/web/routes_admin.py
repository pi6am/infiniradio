"""Admin panel: dashboard, library, prompts, jobs, settings. Server-rendered
Jinja2 + htmx-free plain forms (full page submissions) to keep it simple."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
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
    out = dict(row)
    out["prompt_name"] = prompt["name"] if prompt else ("custom" if row["style_text"] else "?")
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
                    httponly=True, samesite="lax")
    return resp


@router.post("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
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
        " COALESCE(SUM(protected = 1 AND status = 'ready'), 0) AS protected,"
        " COALESCE(SUM(CASE WHEN status = 'ready' THEN duration_s ELSE 0 END), 0.0)"
        " AS playtime_s FROM tracks"
    )
    gen_status = await rt.generator.status() if rt.generator else {"backend": "off", "ok": None}
    jobs = db.query("SELECT j.*, p.name AS prompt_name, pt.title AS parent_title"
                    " FROM generation_jobs j"
                    " LEFT JOIN prompts p ON p.id = j.prompt_id"
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
        " p.name AS prompt_name FROM play_queue q"
        " JOIN tracks t ON t.id = q.track_id"
        " LEFT JOIN prompts p ON p.id = t.prompt_id ORDER BY q.id"
    )
    return request.app.state.templates.TemplateResponse(
        request, "admin/dashboard.html",
        _ctx(request,
             np=np, quota=quota, stats=stats, gen=gen_status, jobs=jobs,
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
    return request.app.state.templates.TemplateResponse(
        request, "admin/track.html",
        _ctx(request, t=track, parent=parent,
             children=[_track_with_meta(db, c) for c in children],
             votes=votes, plays=plays, prompts=prompts))


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
    # Never queued again, regardless of what follows.
    rt.db.execute("DELETE FROM play_queue WHERE track_id = ?", (track_id,))
    feeder = rt.feeder
    if feeder and track_id in feeder.active_ids:
        # The track is being fed right now: skip it out of the buffer first
        # (the decoder may hold the playout file open, which also blocks
        # deletion on Windows), then remove the files.
        feeder.request_skip()
        for _ in range(150):  # up to ~15s
            if track_id not in feeder.active_ids:
                break
            await asyncio.sleep(0.1)
    library.delete_track(rt.db, rt.cfg, track_id)
    return RedirectResponse(_back(request, "/admin/library"), status_code=303)


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
        prompts.append(p)
    return request.app.state.templates.TemplateResponse(
        request, "admin/prompts.html", _ctx(request, prompts=prompts))


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


# ------------------------------------------------------------------ jobs

@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    if not _is_admin(request):
        return _login_redirect()
    rt = request.app.state.rt
    jobs = rt.db.query(
        "SELECT j.*, p.name AS prompt_name, pt.title AS parent_title"
        " FROM generation_jobs j"
        " LEFT JOIN prompts p ON p.id = j.prompt_id"
        " LEFT JOIN tracks pt ON pt.id = j.parent_track_id"
        " ORDER BY j.id DESC LIMIT 200")
    return request.app.state.templates.TemplateResponse(
        request, "admin/jobs.html", _ctx(request, jobs=jobs))


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
        if rt.generator:
            await rt.generator.interrupt()
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
    return request.app.state.templates.TemplateResponse(
        request, "admin/generate.html", _ctx(request, prompts=prompts))


@router.post("/generate")
async def generate_submit(request: Request, prompt_id: int = Form(0),
                          custom_name: str = Form(""), custom_style: str = Form(""),
                          custom_song: str = Form("[Intro]\n\n[Instrumental]\n\n[Outro]"),
                          csrf: str = Form("")):
    if not _is_admin(request) or not _csrf_ok(request, csrf):
        return PlainTextResponse("forbidden", status_code=403)
    rt = request.app.state.rt
    if prompt_id:
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
