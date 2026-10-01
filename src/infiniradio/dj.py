"""The DJ: track selection, generation commissioning, job execution.

Selection: weighted sampling over eligible (ready, not recently aired) tracks
using Wilson scores, with a share of picks reserved for never-played tracks so
new music gets airtime and can earn votes. Falls back to least-recently-played
when nothing is eligible, so the stream never starves.

Commissioning: just-in-time and demand-driven. A single job is commissioned
only when the "runway" — eligible material that can air without violating
the repeat windows — drops below dj.runway_minutes, and only when fewer
than dj.max_queued_jobs are waiting. This keeps every decision (generator
choice, variation vs fresh) informed by the latest prompts and listener
votes, and grows the library until the repeat windows are comfortably
satisfied; the disk-quota pause is the hard stop. Fresh jobs pick a
*generator* — an enabled prompt, or an enabled lyrics workflow when an LLM
is configured — with the same weighted rule (weight = dj.prompt_base_weight
plus the average Wilson score of its aired tracks), so votes arbitrate
between prompt styles and workflow variants alike. With probability
`dj.variation_rate` the job is a variation: a popular track's saved ABC
re-rendered — possibly under a *different style prompt* (new
instrumentation) and always with a fresh seed.

Execution: two serial runners share the generation_jobs table. The render
runner claims phase='render' jobs (instrumental prompts, and lyrics jobs
after their pipeline finished); the lyrics runner claims phase='lyrics'
jobs and drives the LLM workflow engine (stage progress persisted to the
job row for resume; every LLM exchange logged to job_stage_log). With
separate text/music backends (the default) a job's lyrics generate while
an earlier job renders; lyrics.serialize_with_render = true gives both
runners one shared lock instead (single-GPU stations).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from datetime import timedelta
from typing import Any

from . import janitor, library, lyrics, naming
from .config import Config, runtime_get
from .db import DB
from .generator.base import GenRequest, Generator
from .lyrics import AttemptTrace, WorkflowCancelled, WorkflowSpec
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
    windows: not among the most recent repeat_window_tracks airs, not
    played within repeat_window_hours. Trashed and held tracks are
    never eligible (hold is the admin's 'never air this' switch)."""
    last_n = recent_track_ids(db, runtime_get(cfg, db, "dj.repeat_window_tracks"))
    cutoff = iso(utcnow() - timedelta(hours=runtime_get(cfg, db, "dj.repeat_window_hours")))
    rows = db.query("SELECT * FROM tracks WHERE status = 'ready' AND hold = 0")
    return [r for r in rows
            if r["id"] not in last_n
            and r["id"] not in exclude_ids
            and (r["last_played_at"] is None or r["last_played_at"] <= cutoff)]


def pick_next(db: DB, cfg: Config, exclude_ids: set[int] = set()) -> dict | None:
    eligible = eligible_tracks(db, cfg, exclude_ids)

    if not eligible:
        # Nothing eligible (tiny library): fall back to least-recently-played.
        rest = [r for r in db.query(
            "SELECT * FROM tracks WHERE status = 'ready' AND hold = 0")
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


# --------------------------------------------------------- prompt / generator selection

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


def workflow_air_stats(db: DB, workflow_id: str) -> dict:
    """Same verdicts for a lyrics workflow's aired tracks."""
    rows = db.query(
        "SELECT vote_up, vote_down FROM tracks"
        " WHERE workflow_id = ? AND play_count > 0", (workflow_id,))
    scores = [wilson_score(r["vote_up"], r["vote_down"]) for r in rows]
    return {"aired": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0.0}


def workflow_weight(db: DB, cfg, workflow_id: str) -> float:
    base = runtime_get(cfg, db, "dj.prompt_base_weight")
    return base + workflow_air_stats(db, workflow_id)["avg_score"]


def tag_air_stats(db: DB, tag_id: int) -> dict:
    """Per-track verdicts for every AIRED track carrying a tag — the
    cross-cutting view ("do japanese-drafted songs outperform?") that
    per-generator weights cannot show."""
    rows = db.query(
        "SELECT t.vote_up, t.vote_down FROM track_tags tt"
        " JOIN tracks t ON t.id = tt.track_id"
        " WHERE tt.tag_id = ? AND t.play_count > 0", (tag_id,))
    scores = [wilson_score(r["vote_up"], r["vote_down"]) for r in rows]
    return {"aired": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0.0}


def style_air_stats(db: DB, style_id: int) -> dict:
    """Per-track verdicts for a style's AIRED tracks — the steering signal
    that lets listener votes arbitrate between styles."""
    rows = db.query(
        "SELECT vote_up, vote_down FROM tracks"
        " WHERE style_id = ? AND play_count > 0", (style_id,))
    scores = [wilson_score(r["vote_up"], r["vote_down"]) for r in rows]
    return {"aired": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0.0}


def style_weight(db: DB, cfg, style_id: int) -> float:
    base = runtime_get(cfg, db, "dj.style_base_weight")
    return base + style_air_stats(db, style_id)["avg_score"]


def pick_style(rt) -> dict | None:
    """Soft diversity-weighted style choice, same math as prompts and
    workflows: weight = base + avg Wilson of aired tracks."""
    styles = rt.db.query("SELECT * FROM styles WHERE enabled = 1")
    if not styles:
        return None
    weights = [style_weight(rt.db, rt.cfg, s["id"]) for s in styles]
    return random.choices(styles, weights=weights, k=1)[0]


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


def pick_generator(rt) -> tuple[str, object] | None:
    """A fresh commission's generator: ('prompt', row) or ('workflow', spec).

    dj.lyrics_share of fresh jobs go to lyrics workflows when any are
    enabled (and an LLM is configured); the rest use instrumental prompts.
    Within the chosen pool the usual weighted-bandit rule applies (weight =
    base + avg Wilson of aired tracks), so votes still steer which
    workflow / which prompt wins.
    """
    prompts = [("prompt", p) for p in rt.db.query("SELECT * FROM prompts WHERE enabled = 1")]
    workflows: list[tuple[str, object]] = []
    if rt.llm is not None:
        workflows = [("workflow", w) for w in enabled_workflows(rt)]
    if prompts and workflows:
        share = runtime_get(rt.cfg, rt.db, "dj.lyrics_share")
        pool = workflows if random.random() < share else prompts
    else:
        pool = workflows or prompts
    if not pool:
        return None
    weights = [
        prompt_weight(rt.db, rt.cfg, c[1]["id"]) if c[0] == "prompt"
        else workflow_weight(rt.db, rt.cfg, c[1].name)
        for c in pool
    ]
    return random.choices(pool, weights=weights, k=1)[0]


def enabled_workflows(rt) -> list:
    """Workflows selectable right now: enabled in their workflow.toml AND
    not switched off from the admin panel. The admin override lives in the
    settings table, so disabling a workflow on a running server is one
    click — no file edits, no restart (queued jobs still finish)."""
    disabled = set(rt.db.get_setting("workflows_disabled") or [])
    return [w for w in getattr(rt, "workflows", {}).values()
            if w.enabled and w.name not in disabled]


def variation_parent(db: DB, cfg: Config) -> dict | None:
    threshold = runtime_get(cfg, db, "dj.variation_score_threshold")
    rows = db.query(
        "SELECT * FROM tracks WHERE status = 'ready' AND hold = 0"
        " AND abc_text != '' AND parent_track_id IS NULL"
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
                style_override: str | None = None, seed: int | None = None,
                workflow_id: str | None = None,
                style_id: int | None = None) -> int:
    """Queue a generation job. Workflow jobs enter the 'lyrics' phase first
    (theme title stashed in progress); everything else goes straight to
    'render'.

    With style_id, the style's fixed vars and weighted choices are resolved
    right here with the job's seed and stashed in progress — the resolution
    is then reproducible from the job row alone, and a later style edit
    cannot silently change what an already-commissioned job will render.
    """
    from . import styles as styles_mod

    phase = "lyrics" if workflow_id else "render"
    progress: dict[str, Any] = {}
    if workflow_id:
        progress["theme_title"] = naming.unique_title(db)
    if style_id is not None:
        row = db.one("SELECT * FROM styles WHERE id = ?", (style_id,))
        if row is None:
            raise ValueError(f"style {style_id!r} does not exist")
        if seed is None:
            seed = _random_seed()
        progress["overrides"] = styles_mod.resolve(
            json.loads(row["vars_json"] or "{}"),
            json.loads(row["weighted_json"] or "{}"), seed)
        db.execute(
            "UPDATE styles SET times_used = times_used + 1, last_used_at = ? WHERE id = ?",
            (iso(utcnow()), style_id))
    job_id = db.execute(
        "INSERT INTO generation_jobs (prompt_id, parent_track_id, style_override, seed,"
        " workflow_id, style_id, phase, progress, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
        (prompt_id, parent_track_id, style_override, seed, workflow_id, style_id, phase,
         json.dumps(progress, ensure_ascii=False), iso(utcnow())),
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

    if job.get("workflow_id"):
        # Lyrics workflow: style/lyrics are the workflow's outputs, stashed
        # in progress by the lyrics runner when the pipeline completed.
        progress = _job_progress(job)
        outputs = progress.get("outputs") or {}
        return GenRequest(
            style=outputs.get("style") or "",
            lyrics=outputs.get("lyrics") or "",
            seed=job.get("seed") or _random_seed(),
            cot_mode="full",
        )

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
    lyrics_text = prompt["song_text"] if prompt else ""
    cot = prompt["cot_mode"] if prompt else "full"
    return GenRequest(style=style, lyrics=lyrics_text,
                      seed=job.get("seed") or _random_seed(), cot_mode=cot)


def _job_progress(job: dict) -> dict:
    try:
        return json.loads(job.get("progress") or "{}")
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------- background loops

def reconcile_stale_jobs(db: DB) -> None:
    """Finalize state left behind by a crash or kill — call once at startup.

    Nothing can legitimately be running at boot: the runners only claim
    queued jobs (either phase), and a placeholder track exists only while
    its render runs. So any 'running' job or 'generating' track belongs to
    the previous process. Fail both; queued jobs survive and run normally
    (a queued lyrics-phase job resumes from its saved stage progress).
    """
    n_jobs = db.one("SELECT COUNT(*) AS n FROM generation_jobs WHERE status = 'running'")["n"]
    if n_jobs:
        db.execute(
            "UPDATE generation_jobs SET status = 'failed',"
            " error = 'interrupted by server restart', finished_at = ?"
            " WHERE status = 'running'", (iso(utcnow()),))
    n_tracks = db.one("SELECT COUNT(*) AS n FROM tracks WHERE status = 'generating'")["n"]
    if n_tracks:
        db.execute("UPDATE tracks SET status = 'failed' WHERE status = 'generating'")
    if n_jobs or n_tracks:
        log.warning("startup reconciliation: failed %d stale running job(s) and"
                    " %d orphaned generating track(s)", n_jobs, n_tracks)


def _fail_job(db: DB, job_id: int, error: str) -> None:
    db.execute("UPDATE generation_jobs SET status = 'failed', error = ?, finished_at = ?"
               " WHERE id = ? AND status = 'running'",
               (str(error)[:1000], iso(utcnow()), job_id))


def _log_stage_attempt(db: DB, job_id: int, trace: AttemptTrace) -> None:
    """Persist one LLM exchange in job_stage_log (append-only diagnostics).
    Continuation rounds (truncation recovery) and the final call's raw
    finish_reason are folded into usage_json."""
    usage = dict(trace.usage)
    if trace.continuations:
        usage["continuations"] = trace.continuations
    if trace.finish_reason:
        usage["finish_reason"] = trace.finish_reason
    db.execute(
        "INSERT INTO job_stage_log (job_id, stage_id, attempt, prompt_text, response_text,"
        " vars_json, gate_json, model, usage_json, duration_s, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, trace.stage_id, trace.attempt, trace.prompt_text, trace.response_text,
         json.dumps(trace.vars_snapshot, ensure_ascii=False),
         json.dumps([{"code": f.code, "check": f.check, "detail": f.detail}
                     for f in trace.failures]),
         trace.model, json.dumps(usage), round(trace.duration_s, 3), iso(utcnow())),
    )


async def _run_lyrics_phase(rt, job: dict) -> None:
    """Run one job's lyrics workflow, then requeue the job for rendering.

    The workflow engine emits every attempt via on_attempt (persisted to
    job_stage_log as it happens — a crash loses nothing already run) and
    takes a guard that notices admin cancellation between stages. Success
    stashes the outputs in progress and flips the phase to 'render'; the
    render runner picks the job up from there. In serialize mode the
    shared gen lock keeps the text GPU idle while a render is in flight.
    """
    db = rt.db
    spec: WorkflowSpec | None = rt.workflows.get(job["workflow_id"])
    db.execute("UPDATE generation_jobs SET status = 'running', submitted_at = ?,"
               " lyrics_started_at = ? WHERE id = ?",
               (iso(utcnow()), iso(utcnow()), job["id"]))
    if spec is None:
        _fail_job(db, job["id"],
                  f"workflow {job['workflow_id']!r} is not loaded (renamed or removed?)")
        return

    progress = _job_progress(job)
    theme_title = progress.get("theme_title") or naming.unique_title(db)
    overrides = {k: v for k, v in (progress.get("overrides") or {}).items()
                 if k not in lyrics.RESERVED_VARS}
    # The engine merges workflow [vars] defaults under these; style
    # overrides win, and the seed title is reserved.
    initial: dict[str, Any] = dict(overrides)
    if spec.seed_title:
        initial["title"] = theme_title

    def guard() -> bool:
        row = db.one("SELECT status FROM generation_jobs WHERE id = ?", (job["id"],))
        return row is not None and row["status"] == "running"

    def with_job_state(snapshot: dict) -> dict:
        """Engine snapshots carry only workflow state; re-attach the
        commission-time facts (theme title, resolved style overrides) so
        they survive every progress write."""
        snap = dict(snapshot)
        snap["theme_title"] = theme_title
        if overrides:
            snap["overrides"] = overrides
        return snap

    def on_stage(_stage_id: str, snapshot: dict) -> None:
        # Persist stage completions as they happen: the jobs page shows live
        # progress, and a failed run's retry resumes from the last completed
        # stage instead of re-paying for the whole pipeline.
        db.execute("UPDATE generation_jobs SET progress = ? WHERE id = ?",
                   (json.dumps(with_job_state(snapshot), ensure_ascii=False), job["id"]))

    if progress.get("completed"):
        log.info("job %s: lyrics via workflow %s (theme %r, resuming at stage %s)",
                 job["id"], spec.name, theme_title, progress["completed"][-1])
    else:
        log.info("job %s: lyrics via workflow %s (theme %r)",
                 job["id"], spec.name, theme_title)

    async def _work() -> None:
        result = await lyrics.run_workflow(
            rt.llm, spec, initial_vars=initial,
            tag_vocabulary=library.tag_vocabulary(db),
            progress=progress, guard=guard,
            on_attempt=lambda tr: _log_stage_attempt(db, job["id"], tr),
            on_stage=on_stage,
            max_continuations=int(rt.cfg.get("llm.max_continuations", 2)))
        if not result.ok:
            _fail_job(db, job["id"], result.error)
            return
        final_progress = with_job_state(result.progress)
        final_progress["outputs"] = {
            "title": result.title or "",
            "theme_title": theme_title,
            "style": result.style,
            "lyrics": result.lyrics,
            "tags": result.tags,
        }
        # Re-check status: an admin cancellation made while the last stage
        # was in flight wins over completion.
        row = db.one("SELECT status FROM generation_jobs WHERE id = ?", (job["id"],))
        if row is not None and row["status"] == "running":
            db.execute(
                "UPDATE generation_jobs SET progress = ?, phase = 'render', status = 'queued',"
                " lyrics_finished_at = ? WHERE id = ?",
                (json.dumps(final_progress, ensure_ascii=False), iso(utcnow()), job["id"]))
            log.info("job %s: lyrics done — %s (queued for render)", job["id"],
                     result.title or theme_title)

    lock = getattr(rt, "gen_lock", None)
    try:
        if lock is not None:
            async with lock:
                await _work()
        else:
            await _work()
    except WorkflowCancelled:
        log.info("job %s: lyrics cancelled mid-workflow", job["id"])
    except Exception as exc:
        log.exception("job %s lyrics phase failed", job["id"])
        _fail_job(db, job["id"], str(exc))


async def lyrics_runner(rt) -> None:
    """Serially process queued lyrics jobs (LLM pipeline, one at a time).

    Runs alongside the render runner; in the default (two-machine) mode a
    job's lyrics generate while an earlier job renders. rt.gen_lock (set
    when lyrics.serialize_with_render = true) serializes the two.
    """
    log.info("lyrics runner started (llm: %s)", rt.llm.name if rt.llm else "off")
    while True:
        try:
            job = rt.db.one(
                "SELECT * FROM generation_jobs WHERE status = 'queued' AND phase = 'lyrics'"
                " ORDER BY id LIMIT 1")
            if not job or not rt.llm:
                await asyncio.sleep(JOB_POLL_S)
                continue
            await _run_lyrics_phase(rt, job)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("lyrics runner iteration failed")
            await asyncio.sleep(JOB_POLL_S)


async def job_runner(rt) -> None:
    """Serially process queued render jobs (phase = 'render').

    Claims only render-phase jobs: instrumental prompt jobs enqueue
    directly into this phase; lyrics jobs arrive here after the lyrics
    runner flips their phase. ComfyUI renders one at a time, exactly as
    before the lyrics feature.
    """
    log.info("job runner started (backend: %s)", rt.generator.name if rt.generator else "off")
    while True:
        try:
            job = rt.db.one(
                "SELECT * FROM generation_jobs WHERE status = 'queued' AND phase = 'render'"
                " ORDER BY id LIMIT 1")
            if not job or not rt.generator:
                await asyncio.sleep(JOB_POLL_S)
                continue

            db, cfg = rt.db, rt.cfg
            db.execute("UPDATE generation_jobs SET status = 'running', submitted_at = ?,"
                       " render_started_at = ? WHERE id = ?",
                       (iso(utcnow()), iso(utcnow()), job["id"]))
            track_id = None
            try:
                req = build_request(db, job)
                progress = _job_progress(job)
                outputs = progress.get("outputs") or {}
                track_id = library.create_track(
                    db, cfg,
                    prompt_id=job["prompt_id"], parent_track_id=job["parent_track_id"],
                    workflow_id=job["workflow_id"], style_id=job["style_id"],
                    title=outputs.get("title") or "",
                    theme_title=outputs.get("theme_title") or "",
                    style_text=req.style, lyrics_snapshot=req.lyrics,
                    seed=req.seed, cot_mode=req.cot_mode,
                )
                db.execute("UPDATE generation_jobs SET track_id = ? WHERE id = ?",
                           (track_id, job["id"]))
                if job["workflow_id"]:
                    log.info("job %s: rendering %r (workflow %s, seed %s)", job["id"],
                             outputs.get("title") or outputs.get("theme_title"),
                             job["workflow_id"], req.seed)
                else:
                    log.info("job %s: generating %s", job["id"],
                             "variation of track %s" % job["parent_track_id"]
                             if job["parent_track_id"] else "new track (seed %s)" % req.seed)
                lock = getattr(rt, "gen_lock", None)
                if lock is not None:
                    async with lock:
                        result = await rt.generator.generate(
                            req, library.track_dir(cfg, track_id))
                else:
                    result = await rt.generator.generate(req, library.track_dir(cfg, track_id))
                await library.ingest(db, cfg, track_id, result,
                                     tags=outputs.get("tags") or None)
                # Conditional on 'running': if the job was cancelled from the
                # admin panel while we were generating, the cancellation wins.
                db.execute("UPDATE generation_jobs SET status = 'done', finished_at = ?,"
                           " comfy_prompt_id = ? WHERE id = ? AND status = 'running'",
                           (iso(utcnow()), str(result.info.get("comfy_prompt_id", "")), job["id"]))
            except Exception as exc:
                log.exception("job %s failed", job["id"])
                if track_id:
                    library.fail_track(db, track_id)
                _fail_job(db, job["id"], str(exc))
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
    """Enqueue one job: a variation of a proven track, or a fresh generator.

    A fresh commission picks between enabled prompts and enabled lyrics
    workflows with the same weighted-bandit rule (weight = base + avg
    Wilson of aired tracks), so listener votes arbitrate between prompt
    styles and workflow variants alike. With two runners, one commission
    per free queue slot still yields lyrics/render overlap: a job whose
    lyrics finished renders while the next job's lyrics generate.
    """
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

    picked = pick_generator(rt)
    if picked is not None:
        kind, obj = picked
        if kind == "workflow":
            style = pick_style(rt) if obj.styled else None
            enqueue_job(rt.db, workflow_id=obj.name,
                        style_id=style["id"] if style else None)
            log.info("commissioned new track via lyrics workflow %r%s (runway "
                     "%.1f min over %d eligible tracks)", obj.name,
                     f" in style {style['name']!r}" if style else "",
                     runway_s / 60.0, eligible_n)
        else:
            enqueue_job(rt.db, prompt_id=obj["id"])
            log.info("commissioned new track with prompt %r (runway %.1f min over "
                     "%d eligible tracks)", obj["name"], runway_s / 60.0, eligible_n)
        return True
    if eligible_n == 0:
        log.warning("library empty but no enabled generators exist — "
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
