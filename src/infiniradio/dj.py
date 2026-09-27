"""The DJ: track selection, generation commissioning, job execution.

Selection: weighted sampling over eligible (ready, not recently aired) tracks
using Wilson scores, with a share of picks reserved for never-played tracks so
new music gets airtime and can earn votes. Falls back to least-recently-played
when nothing is eligible, so the stream never starves.

Commissioning: just-in-time and demand-driven. A single job is commissioned
only when the "runway" — eligible material that can air without violating
the repeat windows — drops below dj.runway_minutes, and only when fewer
than dj.max_queued_jobs are waiting. This keeps every decision (prompt
choice, variation vs fresh) informed by the latest prompts and listener
votes, and grows the library until the repeat windows are comfortably
satisfied; the disk-quota pause is the hard stop. Prompt choice is
diversity-safe: each prompt's selection weight is dj.prompt_base_weight plus
the average Wilson score of its aired tracks — quality steers gently while
the per-track average (not vote totals) prevents a numerous prompt from
feeding back into more selections. With probability `dj.variation_rate` the
job is a variation: a popular track's saved ABC re-rendered — possibly under
a *different style prompt* (new instrumentation) and always with a fresh
seed.
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
from .util import iso, utcnow, wilson_score

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


def eligible_tracks(db: DB, cfg, exclude_ids: set[int] = set()) -> list[dict]:
    """Ready tracks the DJ may pick right now without violating the repeat
    windows: not among the most recent repeat_window_tracks airs, and not
    played within repeat_window_hours."""
    last_n = recent_track_ids(db, runtime_get(cfg, db, "dj.repeat_window_tracks"))
    cutoff = iso(utcnow() - timedelta(hours=runtime_get(cfg, db, "dj.repeat_window_hours")))
    rows = db.query("SELECT * FROM tracks WHERE status = 'ready'")
    return [r for r in rows
            if r["id"] not in last_n
            and r["id"] not in exclude_ids
            and (r["last_played_at"] is None or r["last_played_at"] <= cutoff)]


def pick_next(db: DB, cfg: Config, exclude_ids: set[int] = set()) -> dict | None:
    eligible = eligible_tracks(db, cfg, exclude_ids)

    if not eligible:
        # Nothing eligible (tiny library): fall back to least-recently-played.
        rest = [r for r in db.query("SELECT * FROM tracks WHERE status = 'ready'")
                if r["id"] not in exclude_ids]
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


# --------------------------------------------------------- prompt selection

def prompt_air_stats(db: DB, prompt_id: int) -> dict:
    """Per-track listener verdicts for a prompt's AIRED tracks."""
    rows = db.query(
        "SELECT vote_up, vote_down FROM tracks"
        " WHERE prompt_id = ? AND play_count > 0", (prompt_id,))
    scores = [wilson_score(r["vote_up"], r["vote_down"]) for r in rows]
    return {"aired": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0.0}


def prompt_weight(db: DB, cfg, prompt_id: int) -> float:
    """dj.prompt_base_weight + mean Wilson score of the prompt's aired tracks.

    Averaging *per track* (rather than summing votes) keeps the weight
    independent of how many tracks a prompt already has: a prompt cannot
    feed back into more selections just by being numerous — that aggregate
    feedback loop let one or two styles dominate generation. The base
    weight gives every prompt — including never-aired ones — a nonzero
    exploration floor, so diversity never collapses to a single style.
    """
    base = runtime_get(cfg, db, "dj.prompt_base_weight")
    return base + prompt_air_stats(db, prompt_id)["avg_score"]


def pick_prompt(db: DB, cfg, exclude_prompt_id: int | None = None) -> dict | None:
    """Soft diversity-weighted prompt choice (weight ~ base + avg score)."""
    prompts = db.query("SELECT * FROM prompts WHERE enabled = 1")
    if exclude_prompt_id is not None:
        others = [p for p in prompts if p["id"] != exclude_prompt_id]
        if others:
            prompts = others
    if not prompts:
        return None
    weights = [prompt_weight(db, cfg, p["id"]) for p in prompts]
    return random.choices(prompts, weights=weights, k=1)[0]


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
    """One commissioning decision (see commission_loop).

    Just-in-time and demand-driven: commission a single job only when the
    "runway" — eligible material that can be aired without violating the
    repeat windows — drops below dj.runway_minutes, and only when fewer
    than dj.max_queued_jobs are already waiting. Decisions stay informed by
    the latest prompts and votes, and the queue never grows faster than
    the serial GPU runner drains it. The library grows until the repeat
    windows are comfortably satisfied (the runway then stays above the
    threshold); the disk-quota pause remains the hard stop.
    """
    if not rt.generator:
        return  # no generator configured (playout-only mode)
    paused, reason = janitor.generation_paused(rt.db, rt.cfg)
    if paused:
        log.debug("commissioning paused: %s", reason)
        return

    queued = rt.db.one(
        "SELECT COUNT(*) AS n FROM generation_jobs WHERE status = 'queued'"
    )["n"]
    if queued >= runtime_get(rt.cfg, rt.db, "dj.max_queued_jobs"):
        return

    eligible = eligible_tracks(rt.db, rt.cfg)
    runway_s = sum(r["duration_s"] or 0.0 for r in eligible)
    if runway_s >= runtime_get(rt.cfg, rt.db, "dj.runway_minutes") * 60.0:
        return  # enough pickable material left; no urgency

    _commission_one(rt, len(eligible), runway_s)


def _commission_one(rt, eligible_n: int, runway_s: float) -> bool:
    """Enqueue one job: a variation of a proven track, or a fresh prompt."""
    if random.random() < runtime_get(rt.cfg, rt.db, "dj.variation_rate"):
        parent = variation_parent(rt.db, rt.cfg)
        if parent:
            prompt = pick_prompt(rt.db, rt.cfg, exclude_prompt_id=parent["prompt_id"])
            prompt_id = prompt["id"] if prompt else parent["prompt_id"]
            dup = rt.db.one(
                "SELECT 1 FROM tracks WHERE parent_track_id = ? AND prompt_id = ?"
                " AND status != 'deleted'", (parent["id"], prompt_id))
            if not dup:
                enqueue_job(rt.db, prompt_id=prompt_id,
                            parent_track_id=parent["id"])
                log.info("commissioned variation of track %s (%s) in style %s "
                         "(runway %.1f min)", parent["id"], parent["title"],
                         prompt["name"] if prompt else "(same style)",
                         runway_s / 60.0)
                return True

    prompt = pick_prompt(rt.db, rt.cfg)
    if prompt:
        enqueue_job(rt.db, prompt_id=prompt["id"])
        log.info("commissioned new track with prompt %r (runway %.1f min over "
                 "%d eligible tracks)", prompt["name"], runway_s / 60.0,
                 eligible_n)
        return True
    if eligible_n == 0:
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
