"""The DJ: track selection, generation commissioning, job execution.

Selection: weighted sampling over eligible (ready, not recently aired) tracks
using Wilson scores, with a share of picks reserved for never-played tracks so
new music gets airtime and can earn votes. Falls back to least-recently-played
when nothing is eligible, so the stream never starves.

Commissioning: a background tick keeps the library above `dj.min_library` by
queueing generation jobs. Prompt choice is Thompson sampling over each
prompt's tracks' votes (self-optimizing prompt library). With probability
`dj.variation_rate` the job is a variation: a popular track's saved ABC
re-rendered — possibly under a *different style prompt* (new instrumentation)
and always with a fresh seed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from datetime import timedelta

from . import janitor, library
from .config import Config, runtime_get
from .db import DB
from .generator.base import GenRequest, Generator
from .util import iso, parse_iso, utcnow, wilson_score

log = logging.getLogger(__name__)

SOFTMAX_TEMPERATURE = 0.35
PLAY_COUNT_FATIGUE = 0.05
COMMISSION_TICK_S = 60.0
JANITOR_TICK_S = 120.0
JOB_POLL_S = 2.0


# ------------------------------------------------------------------- selection

def recent_track_ids(db: DB, n: int) -> set[int]:
    rows = db.query(
        "SELECT track_id FROM play_log ORDER BY id DESC LIMIT ?", (int(n),)
    )
    return {r["track_id"] for r in rows}


def pick_next(db: DB, cfg: Config, exclude_ids: set[int] = set()) -> dict | None:
    last_n = recent_track_ids(db, runtime_get(cfg, db, "dj.repeat_window_tracks"))
    cutoff = iso(utcnow() - timedelta(hours=runtime_get(cfg, db, "dj.repeat_window_hours")))

    rows = db.query("SELECT * FROM tracks WHERE status = 'ready'")
    eligible = [
        r for r in rows
        if r["id"] not in last_n
        and r["id"] not in exclude_ids
        and (r["last_played_at"] is None or r["last_played_at"] <= cutoff)
    ]

    if not eligible:
        # Nothing eligible (tiny library): fall back to least-recently-played.
        rest = [r for r in rows if r["id"] not in exclude_ids]
        if not rest:
            return None
        rest.sort(key=lambda r: (r["last_played_at"] is not None, r["last_played_at"] or ""))
        return rest[0]

    share = runtime_get(cfg, db, "dj.new_track_share")
    fresh = [r for r in eligible if r["play_count"] == 0]
    if fresh and (not eligible_without_fresh(eligible) or random.random() < share):
        return random.choice(fresh)

    weights = []
    for r in eligible:
        score = wilson_score(r["vote_up"], r["vote_down"])
        w = math.exp(score / SOFTMAX_TEMPERATURE) / (1.0 + PLAY_COUNT_FATIGUE * r["play_count"])
        weights.append(w)
    return random.choices(eligible, weights=weights, k=1)[0]


def eligible_without_fresh(eligible: list[dict]) -> list[dict]:
    return [r for r in eligible if r["play_count"] > 0]


# ------------------------------------------------------------- prompt bandit

def prompt_posterior(db: DB, prompt_id: int) -> tuple[float, float]:
    row = db.one(
        "SELECT COALESCE(SUM(vote_up), 0) AS up, COALESCE(SUM(vote_down), 0) AS down"
        " FROM tracks WHERE prompt_id = ? AND status IN ('ready', 'deleted')",
        (prompt_id,),
    ) or {"up": 0, "down": 0}
    return 1.0 + row["up"], 1.0 + row["down"]


def thompson_prompt(db: DB, exclude_prompt_id: int | None = None) -> dict | None:
    prompts = db.query("SELECT * FROM prompts WHERE enabled = 1")
    if exclude_prompt_id is not None:
        others = [p for p in prompts if p["id"] != exclude_prompt_id]
        if others:
            prompts = others
    if not prompts:
        return None
    scored = []
    for p in prompts:
        a, b = prompt_posterior(db, p["id"])
        scored.append((random.betavariate(a, b), p))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def variation_parent(db: DB, cfg: Config) -> dict | None:
    threshold = runtime_get(cfg, db, "dj.variation_score_threshold")
    rows = db.query(
        "SELECT * FROM tracks WHERE status = 'ready' AND abc_text != '' AND parent_track_id IS NULL"
    )
    scored = [
        (wilson_score(r["vote_up"], r["vote_down"]), r) for r in rows
        if r["play_count"] > 0 and wilson_score(r["vote_up"], r["vote_down"]) >= threshold
    ]
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return random.choices([r for _, r in scored],
                          weights=[s + 0.01 for s, _ in scored], k=1)[0]


# ------------------------------------------------------------------ jobs

def enqueue_job(db: DB, *, prompt_id: int | None = None, parent_track_id: int | None = None,
                style_override: str | None = None, seed: int | None = None) -> int:
    job_id = db.execute(
        "INSERT INTO generation_jobs (prompt_id, parent_track_id, style_override, seed,"
        " status, created_at) VALUES (?, ?, ?, ?, 'queued', ?)",
        (prompt_id, parent_track_id, style_override, seed, iso(utcnow())),
    )
    if prompt_id:
        db.execute(
            "UPDATE prompts SET times_used = times_used + 1, last_used_at = ? WHERE id = ?",
            (iso(utcnow()), prompt_id),
        )
    return job_id


def _random_seed() -> int:
    return random.getrandbits(31)


def build_request(db: DB, job: dict) -> GenRequest:
    parent = None
    if job.get("parent_track_id"):
        parent = db.one("SELECT * FROM tracks WHERE id = ?", (job["parent_track_id"],))
    prompt = None
    if job.get("prompt_id"):
        prompt = db.one("SELECT * FROM prompts WHERE id = ?", (job["prompt_id"],))

    if parent:
        # Variation: re-render the parent's saved ABC. Style may come from a
        # different prompt (new instrumentation); lyrics stay the parent's.
        style = job.get("style_override") or (prompt["style_text"] if prompt else parent["style_text"])
        return GenRequest(
            style=style,
            lyrics=parent["lyrics_snapshot"],
            seed=job.get("seed") or _random_seed(),
            cot_mode="full",
            abc=parent["abc_text"],
            parent_track_id=parent["id"],
        )

    style = job.get("style_override") or (prompt["style_text"] if prompt else "")
    lyrics = prompt["song_text"] if prompt else ""
    cot = prompt["cot_mode"] if prompt else "full"
    return GenRequest(style=style, lyrics=lyrics,
                      seed=job.get("seed") or _random_seed(), cot_mode=cot)


# ------------------------------------------------------------- background loops

async def job_runner(rt) -> None:
    """Serially process queued generation jobs."""
    log.info("job runner started (backend: %s)", rt.generator.name if rt.generator else "off")
    while True:
        try:
            job = rt.db.one("SELECT * FROM generation_jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
            if not job or not rt.generator:
                await asyncio.sleep(JOB_POLL_S)
                continue

            db, cfg = rt.db, rt.cfg
            db.execute("UPDATE generation_jobs SET status = 'running', submitted_at = ?"
                       " WHERE id = ?", (iso(utcnow()), job["id"]))
            track_id = None
            try:
                req = build_request(db, job)
                track_id = library.create_track(
                    db, cfg,
                    prompt_id=job["prompt_id"], parent_track_id=job["parent_track_id"],
                    style_text=req.style, lyrics_snapshot=req.lyrics,
                    seed=req.seed, cot_mode=req.cot_mode,
                )
                db.execute("UPDATE generation_jobs SET track_id = ? WHERE id = ?",
                           (track_id, job["id"]))
                log.info("job %s: generating %s", job["id"],
                         "variation of track %s" % job["parent_track_id"]
                         if job["parent_track_id"] else "new track (seed %s)" % req.seed)
                result = await rt.generator.generate(req, library.track_dir(cfg, track_id))
                await library.ingest(db, cfg, track_id, result)
                db.execute("UPDATE generation_jobs SET status = 'done', finished_at = ?,"
                           " comfy_prompt_id = ? WHERE id = ?",
                           (iso(utcnow()), str(result.info.get("comfy_prompt_id", "")), job["id"]))
            except Exception as exc:
                log.exception("job %s failed", job["id"])
                if track_id:
                    library.fail_track(db, track_id)
                db.execute("UPDATE generation_jobs SET status = 'failed', error = ?,"
                           " finished_at = ? WHERE id = ?",
                           (str(exc)[:1000], iso(utcnow()), job["id"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("job runner iteration failed")
            await asyncio.sleep(JOB_POLL_S)


async def _commission_tick(rt) -> None:
    """One commissioning decision (see commission_loop)."""
    if not rt.generator:
        return  # no generator configured (playout-only mode)
    paused, reason = janitor.generation_paused(rt.db, rt.cfg)
    if paused:
        log.debug("commissioning paused: %s", reason)
        return

    counts = rt.db.one(
        "SELECT COALESCE(SUM(status = 'ready'), 0) AS ready,"
        " COALESCE(SUM(status = 'generating'), 0) AS generating FROM tracks"
    ) or {"ready": 0, "generating": 0}
    queued = rt.db.one(
        "SELECT COUNT(*) AS n FROM generation_jobs WHERE status = 'queued'"
    )["n"]
    min_library = runtime_get(rt.cfg, rt.db, "dj.min_library")
    want = min_library - (counts["ready"] + counts["generating"] + queued)

    if want > 0:
        for _ in range(min(want, 4)):  # cap the burst; runner executes serially
            if not _commission_one(rt, counts["ready"], min_library):
                break
        return

    # Above the minimum: an infinite radio keeps producing fresh music at a
    # steady cadence anyway — the disk quota (with played-track eviction) is
    # the real throttle, not the library size.
    cadence = runtime_get(rt.cfg, rt.db, "dj.generate_every_minutes")
    if cadence <= 0:
        return
    if rt.db.one("SELECT id FROM generation_jobs WHERE status IN ('queued', 'running')"):
        return
    last = rt.db.one(
        "SELECT finished_at FROM generation_jobs WHERE status = 'done'"
        " ORDER BY id DESC LIMIT 1"
    )
    if last and parse_iso(last["finished_at"]) > utcnow() - timedelta(minutes=cadence):
        return
    _commission_one(rt, counts["ready"], min_library)


def _commission_one(rt, ready_count: int, min_library: int) -> bool:
    """Enqueue one job: a variation of a proven track, or a fresh prompt."""
    if random.random() < runtime_get(rt.cfg, rt.db, "dj.variation_rate"):
        parent = variation_parent(rt.db, rt.cfg)
        if parent:
            prompt = thompson_prompt(rt.db, exclude_prompt_id=parent["prompt_id"])
            prompt_id = prompt["id"] if prompt else parent["prompt_id"]
            dup = rt.db.one(
                "SELECT 1 FROM tracks WHERE parent_track_id = ? AND prompt_id = ?"
                " AND status != 'deleted'", (parent["id"], prompt_id))
            if not dup:
                enqueue_job(rt.db, prompt_id=prompt_id,
                            parent_track_id=parent["id"])
                log.info("commissioned variation of track %s (%s) in style %s",
                         parent["id"], parent["title"],
                         prompt["name"] if prompt else "(same style)")
                return True

    prompt = thompson_prompt(rt.db)
    if prompt:
        enqueue_job(rt.db, prompt_id=prompt["id"])
        log.info("commissioned new track with prompt %r (library %d/%d)",
                 prompt["name"], ready_count, min_library)
        return True
    if ready_count == 0:
        log.warning("library empty but no enabled prompts exist — "
                    "add prompts in the admin panel or set prompts.seed_dir")
    return False


async def commission_loop(rt) -> None:
    """Keep the library stocked; decide between fresh tracks and variations."""
    while True:
        try:
            await _commission_tick(rt)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("commission tick failed")
        await asyncio.sleep(COMMISSION_TICK_S)


async def janitor_loop(rt) -> None:
    while True:
        try:
            active = rt.feeder.active_ids if rt.feeder else set()
            janitor.run_janitor(rt.db, rt.cfg, active)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("janitor tick failed")
        await asyncio.sleep(JANITOR_TICK_S)
