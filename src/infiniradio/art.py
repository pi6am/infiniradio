"""Track art: LLM-written image prompts through data-driven workflows,
rendered by an OpenAI-compatible images endpoint.

A workflow is a directory under art.workflows_dir (default:
art_workflows/) holding a workflow.toml plus Jinja templates — the same
shape as lyrics workflows (stages, gates, extracts, feedback retries,
inheritance), executed by the SAME engine (lyrics.run_workflow). Only
the output contract differs: ``[output] prompt`` names the variable that
holds the final image prompt. Art specs duck-type the engine's output
contract (the prompt rides the required `style` slot); no engine fork.

Jobs are kind='art' rows in generation_jobs:
  phase 'art'        the LLM writes the image prompt from the track's
                     metadata (claimed by the lyrics runner — cheap text
                     work, FIFO by id so queued music lyrics go first)
  phase 'art_render' the images endpoint renders it (claimed by the
                     render runner, ALTERNATING with music renders: the
                     typical endpoint proxies the same GPU that makes
                     music, so neither queue may starve the other)

Everything tunable at runtime lives in the settings table (art.enabled /
art.new_tracks / art.backfill / art.backfill_max_queued, plus the
art_workflow_weights override map and the art_workflows_disabled list —
see the admin prompts page).

Audition (repo root, no server):

    python -m infiniradio.art --list
    python -m infiniradio.art --validate
    python -m infiniradio.art --dry-run cover-illustrated
    python -m infiniradio.art --run cover-illustrated --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import janitor, library, lyrics
from .config import Config, runtime_get
from .db import DB
from .lyrics import (OutputSpec, RESERVED_VARS, StageSpec, WorkflowCancelled,
                     WorkflowError, _jinja_env, _merge_raw, _only_keys,
                     _parse_stage, _resolve_raw)
from .util import iso, utcnow

log = logging.getLogger(__name__)

_ART_WORKFLOW_KEYS = {"name", "description", "enabled", "weight", "params",
                      "vars", "output", "requires", "stage", "abstract",
                      "base", "remove_stages"}
_ART_OUTPUT_KEYS = {"prompt"}
# Known [requires] predicates. A workflow whose requires do not match the
# track is filtered out BEFORE the weighted pick (see workflow_applies);
# unknown keys are startup errors, per the names/ rule.
_ART_REQUIRES_KEYS = {"lyrics"}


# ------------------------------------------------------------------ spec

@dataclass(frozen=True)
class ArtSpec:
    name: str                      # == directory name (the slug used in the DB)
    directory: Path
    description: str
    enabled: bool
    weight: float                  # relative probability of being picked
    params: dict[str, Any]         # available to every template as params.*
    stages: tuple[StageSpec, ...]
    prompt_var: str                # the variable holding the final image prompt
    env: Any                       # jinja2.Environment
    vars: dict[str, str] = field(default_factory=dict)
    requires: dict[str, Any] = field(default_factory=dict)  # track predicates
    abstract: bool = False         # a base: loaded, validated, never picked
    static_tags: tuple[str, ...] = ()   # engine duck-typing: art has none

    @property
    def output(self) -> OutputSpec:
        """Duck-type the lyrics engine's output contract: the image prompt
        rides `style` (the engine's one required slot); the caller reads
        result.vars[prompt_var] itself, so no other field is mapped."""
        return OutputSpec(style=self.prompt_var, lyrics=None)


def _read_art_dir(directory: Path) -> dict:
    """Parse one art workflow's TOML, check keys and name (the raw material
    for inheritance merging — validation of the RESOLVED spec happens in
    _build_art_spec)."""
    wf_path = directory / "workflow.toml"
    where = str(wf_path)
    import tomllib
    try:
        data = tomllib.loads(wf_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise WorkflowError(f"{where}: cannot read workflow: {exc}") from exc
    _only_keys(data, _ART_WORKFLOW_KEYS, where)
    name = str(data.get("name") or "").strip()
    if name != directory.name:
        raise WorkflowError(
            f"{where}: name {name!r} must match the directory name "
            f"{directory.name!r} (copying a workflow? rename both)")
    return data


def _build_art_spec(directory: Path, data: dict, chain: list[Path]) -> ArtSpec:
    """Validate and build a RESOLVED art workflow spec."""
    where = str(directory / "workflow.toml")
    name = str(data.get("name") or "").strip()

    stages_data = data.get("stage") or []
    if not stages_data:
        raise WorkflowError(f"{where}: no [[stage]] entries")
    stages = tuple(_parse_stage(s, f"{where} stage #{i + 1}")
                   for i, s in enumerate(stages_data))
    seen: set[str] = set()
    for s in stages:
        if s.id in seen:
            raise WorkflowError(f"{where}: duplicate stage id {s.id!r}")
        seen.add(s.id)
        if s.block:
            raise WorkflowError(
                f"{where}: stage {s.id!r}: block = is a lyrics concept "
                "(song-section extraction); art stages use saves/extract")
        if s.fresh_retries:
            if s.thread != "new":
                raise WorkflowError(
                    f"{where}: stage {s.id!r}: fresh_retries needs thread = 'new'")
            if s.feedback_template:
                raise WorkflowError(
                    f"{where}: stage {s.id!r}: fresh_retries re-rolls the stage "
                    "prompt in a new conversation — unset feedback_template")
    if stages[0].thread == "continue":
        raise WorkflowError(f"{where}: the first stage cannot thread = 'continue'")

    output_data = data.get("output") or {}
    _only_keys(output_data, _ART_OUTPUT_KEYS, f"{where}: output")
    prompt_var = output_data.get("prompt")
    if not prompt_var:
        raise WorkflowError(f"{where}: output.prompt is required"
                            " (the variable holding the image prompt)")
    produced = ({s.saves for s in stages if s.saves}
                | {k for s in stages for k, _pats in s.extract})
    if str(prompt_var) not in produced:
        raise WorkflowError(
            f"{where}: output.prompt = {prompt_var!r} is not saved or "
            "extracted by any stage")

    env = _jinja_env(chain)
    vars_defaults: dict[str, str] = {}
    for key, value in (data.get("vars") or {}).items():
        if not str(key).isidentifier() or not str(key)[0].islower():
            raise WorkflowError(f"{where}: vars: {key!r} must be a lowercase identifier")
        if str(key) in RESERVED_VARS:
            raise WorkflowError(f"{where}: vars: {key!r} is reserved by the engine")
        if not isinstance(value, str):
            raise WorkflowError(f"{where}: vars: {key!r} must be a string")
        vars_defaults[str(key)] = value
    for s in stages:
        try:
            env.get_template(s.template)
        except Exception as exc:  # jinja2.TemplateError + missing file
            raise WorkflowError(f"{where}: stage {s.id!r}: template {s.template!r}: {exc}") from exc
        if s.feedback_template:
            try:
                env.get_template(s.feedback_template)
            except Exception as exc:
                raise WorkflowError(
                    f"{where}: stage {s.id!r}: feedback_template "
                    f"{s.feedback_template!r}: {exc}") from exc

    try:
        weight = float(data.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise WorkflowError(f"{where}: weight must be a number") from exc
    if weight < 0:
        raise WorkflowError(f"{where}: weight must be >= 0")

    requires_data = data.get("requires") or {}
    _only_keys(requires_data, _ART_REQUIRES_KEYS, f"{where}: requires")
    for key, value in requires_data.items():
        if not isinstance(value, bool):
            raise WorkflowError(f"{where}: requires.{key} must be true or false")

    return ArtSpec(
        name=name,
        directory=directory,
        description=str(data.get("description") or ""),
        enabled=bool(data.get("enabled", True)),
        weight=weight,
        params=dict(data.get("params") or {}),
        stages=stages,
        prompt_var=str(prompt_var),
        env=env,
        vars=vars_defaults,
        requires=dict(requires_data),
        abstract=bool(data.get("abstract", False)),
    )


def load_art_workflows(cfg: Config) -> dict[str, ArtSpec]:
    """Load every art workflow under art.workflows_dir (validated,
    inheritance resolved at load time — a resolved child is
    indistinguishable from a hand-written one)."""
    directory = cfg.path("art.workflows_dir")
    if not directory.is_dir():
        log.info("no art workflows directory at %s — art generation off", directory)
        return {}
    raw: dict[str, tuple[Path, dict]] = {}
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir() or not (sub / "workflow.toml").is_file():
            continue
        data = _read_art_dir(sub)
        raw[str(data["name"])] = (sub, data)
    specs: dict[str, ArtSpec] = {}
    for name, (sub, _data) in raw.items():
        merged, chain = _resolve_raw(name, raw, frozenset())
        merged.pop("_where", None)
        specs[name] = _build_art_spec(sub, merged, chain)
    concrete = sum(not s.abstract for s in specs.values())
    enabled = sum(s.enabled and not s.abstract for s in specs.values())
    log.info("loaded %d art workflow(s) from %s (%d enabled, %d abstract base(s))",
             len(specs), directory, enabled, len(specs) - concrete)
    return specs


# ------------------------------------------------------------------ selection

def effective_weight(db: DB, spec: ArtSpec) -> float:
    """File weight, unless the admin override map (settings table, live
    without restarts) retunes it. Zero is a disabled-but-kept weight."""
    overrides = db.get_setting("art_workflow_weights") or {}
    try:
        w = float(overrides.get(spec.name, spec.weight))
    except (TypeError, ValueError):
        w = spec.weight
    return max(0.0, w)


def enabled_art_workflows(rt) -> list[ArtSpec]:
    """Selectable right now: enabled in workflow.toml, not abstract bases,
    not switched off from the admin panel (the settings-table override —
    one click, no restart; queued jobs still finish)."""
    disabled = set(rt.db.get_setting("art_workflows_disabled") or [])
    return [w for w in getattr(rt, "artworkflows", {}).values()
            if w.enabled and not w.abstract and w.name not in disabled]


def workflow_applies(spec: ArtSpec, track: dict) -> bool:
    """The FILTER half of selection: does this workflow's [requires] match
    the track? Evaluated before any weighting — a filtered-out workflow
    has zero chance regardless of its weight (weights arbitrate between
    the APPLICABLE workflows, they cannot buy entry)."""
    if "lyrics" in spec.requires:
        has = library.authored_lyrics(track.get("lyrics_snapshot") or "") is not None
        if has is not spec.requires["lyrics"]:
            return False
    return True


def pick_art_workflow(rt, track: dict) -> ArtSpec | None:
    """Filter [requires] first, then weighted-pick among the remainder."""
    pool = [w for w in enabled_art_workflows(rt) if workflow_applies(w, track)]
    if not pool:
        return None
    weights = [effective_weight(rt.db, w) for w in pool]
    if not any(w > 0 for w in weights):
        # All-zero is a degenerate config, not "never pick": uniform.
        return random.choice(pool)
    return random.choices(pool, weights=weights, k=1)[0]


def pipeline_ready(rt) -> bool:
    """The art pipeline is CONFIGURED: images backend, LLM (image-prompt
    writing) and workflows all present. This is the runners' claim gate —
    and it is deliberately independent of the runtime switches: a queued
    art job runs whenever this holds, mirroring music, where pausing
    generation never stops a manually queued song."""
    return (getattr(rt, "images", None) is not None
            and rt.llm is not None
            and bool(getattr(rt, "artworkflows", {})))


def auto_generation_on(rt) -> bool:
    """AUTOMATIC art (new tracks + backfill) may enqueue: the pipeline is
    configured and the runtime switch is on. art.enabled governs only
    these automatic paths — manual regeneration from a track page runs
    regardless of it."""
    return pipeline_ready(rt) and bool(runtime_get(rt.cfg, rt.db, "art.enabled"))


# ------------------------------------------------------------------ jobs

def enqueue_art_job(rt, track_id: int, *, workflow_id: str | None = None,
                    seed: int | None = None) -> int | None:
    """Queue an art job for an existing READY track (art targets real
    tracks — retry/cancel must never touch the row).

    workflow_id None = the filter-then-weight pick for THIS track (stored
    on the job so the choice is reproducible). An explicit id (admin
    regen) bypasses [requires] — the dropdown exists to experiment — and
    skips the weighting. Idempotent while one is already queued/running
    for the track.
    """
    db = rt.db
    track = db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    if track is None or track["status"] != "ready":
        return None
    pending = db.one(
        "SELECT id FROM generation_jobs WHERE kind = 'art' AND track_id = ?"
        " AND status IN ('queued', 'running')", (track_id,))
    if pending:
        return pending["id"]
    spec = None
    if workflow_id:
        spec = getattr(rt, "artworkflows", {}).get(workflow_id)
        if spec is None or spec.abstract:
            return None
    else:
        spec = pick_art_workflow(rt, track)
        if spec is None:
            return None
    job_id = db.execute(
        "INSERT INTO generation_jobs (kind, phase, track_id, workflow_id, seed,"
        " progress, status, created_at)"
        " VALUES ('art', 'art', ?, ?, ?, '{}', 'queued', ?)",
        (track_id, spec.name,
         seed if seed is not None else random.getrandbits(31), iso(utcnow())))
    log.info("queued art job %s for track %s via workflow %r",
             job_id, track_id, spec.name)
    return job_id


def maybe_enqueue_for_track(rt, track_id: int) -> None:
    """The art.new_tracks hook: art for a freshly ingested track."""
    if not auto_generation_on(rt):
        return
    if not runtime_get(rt.cfg, rt.db, "art.new_tracks"):
        return
    enqueue_art_job(rt, track_id)


def backfill_tick(rt) -> None:
    """The art.backfill trickle (janitor clock): enqueue art for the oldest
    ready track that never had an art job, within the queue budget.

    "Never had one" is deliberate: a track whose art job FAILED is not
    retried automatically (that would loop a broken workflow against the
    API forever) — its track page has the regenerate button.
    """
    if not auto_generation_on(rt):
        return
    if not runtime_get(rt.cfg, rt.db, "art.backfill"):
        return
    paused, _reason = janitor.generation_paused(rt.db, rt.cfg)
    if paused:
        return
    budget = max(0, int(runtime_get(rt.cfg, rt.db, "art.backfill_max_queued")))
    active = rt.db.one(
        "SELECT COUNT(*) AS n FROM generation_jobs WHERE kind = 'art'"
        " AND status IN ('queued', 'running')")["n"]
    if active >= budget:
        return
    row = rt.db.one(
        "SELECT t.id FROM tracks t"
        " WHERE t.status = 'ready' AND t.art_id IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM generation_jobs j"
        "                 WHERE j.kind = 'art' AND j.track_id = t.id)"
        " ORDER BY t.id LIMIT 1")
    if row:
        enqueue_art_job(rt, row["id"])


def _job_progress(job: dict) -> dict:
    try:
        return json.loads(job.get("progress") or "{}")
    except json.JSONDecodeError:
        return {}


def _fail_job(db: DB, job_id: int, error: str) -> None:
    db.execute("UPDATE generation_jobs SET status = 'failed', error = ?,"
               " finished_at = ? WHERE id = ? AND status = 'running'",
               (str(error)[:1000], iso(utcnow()), job_id))


def track_context(db: DB, track: dict) -> dict[str, Any]:
    """Template context from a track row: everything an image-prompt writer
    could want. `track_tags` (not `tags` — the engine reserves that name
    for the lyrics vocabulary, which art workflows don't use)."""
    genre = ""
    if track["prompt_id"]:
        row = db.one("SELECT name FROM prompts WHERE id = ?", (track["prompt_id"],))
        genre = row["name"] if row else ""
    return {
        "title": track["title"] or "",
        "theme_title": track["theme_title"] or "",
        "style": track["style_text"] or "",
        "lyrics": track["lyrics_snapshot"] or "",
        "track_tags": library.track_tags(db, track["id"]),
        "genre": genre,
        "tempo": track["tempo"] or 0,
        "music_key": track["music_key"] or "",
        "duration_s": track["duration_s"] or 0.0,
        "is_variation": track["parent_track_id"] is not None,
    }


# ------------------------------------------------------------------ phases

async def run_art_phase(rt, job: dict) -> None:
    """Phase 'art': run the LLM workflow that writes the image prompt,
    then requeue the job as 'art_render'. Mirrors dj._run_lyrics_phase
    (progress persisted per stage; retries resume; guard notices admin
    cancellation; serialize mode holds the shared gen lock)."""
    db = rt.db
    spec: ArtSpec | None = getattr(rt, "artworkflows", {}).get(job["workflow_id"])
    db.execute("UPDATE generation_jobs SET status = 'running', submitted_at = ?,"
               " lyrics_started_at = ? WHERE id = ?",
               (iso(utcnow()), iso(utcnow()), job["id"]))
    if spec is None:
        _fail_job(db, job["id"],
                  f"art workflow {job['workflow_id']!r} is not loaded (renamed or removed?)")
        return
    track = db.one("SELECT * FROM tracks WHERE id = ?", (job["track_id"],))
    if track is None or track["status"] != "ready":
        _fail_job(db, job["id"], f"track {job['track_id']} is missing or not ready")
        return

    progress = _job_progress(job)
    initial = track_context(db, track)

    def guard() -> bool:
        row = db.one("SELECT status FROM generation_jobs WHERE id = ?", (job["id"],))
        return row is not None and row["status"] == "running"

    def on_stage(_stage_id: str, snapshot: dict) -> None:
        db.execute("UPDATE generation_jobs SET progress = ? WHERE id = ?",
                   (json.dumps(snapshot, ensure_ascii=False), job["id"]))

    log.info("job %s: art prompt via workflow %s for track %s (%r)",
             job["id"], spec.name, job["track_id"], track["title"])

    async def _work() -> None:
        result = await lyrics.run_workflow(
            rt.llm, spec, initial_vars=initial, progress=progress, guard=guard,
            on_attempt=lambda tr: lyrics.log_attempt(db, job["id"], tr),
            on_stage=on_stage,
            max_continuations=int(rt.cfg.get("llm.max_continuations", 2)))
        if not result.ok:
            _fail_job(db, job["id"], result.error)
            return
        image_prompt = str(result.vars.get(spec.prompt_var, "") or "").strip()
        final = dict(result.progress)
        final["outputs"] = {"image_prompt": image_prompt}
        # Cancellation made while the last stage was in flight wins.
        row = db.one("SELECT status FROM generation_jobs WHERE id = ?", (job["id"],))
        if row is not None and row["status"] == "running":
            db.execute(
                "UPDATE generation_jobs SET progress = ?, phase = 'art_render',"
                " status = 'queued', lyrics_finished_at = ? WHERE id = ?",
                (json.dumps(final, ensure_ascii=False), iso(utcnow()), job["id"]))
            log.info("job %s: image prompt done (%d chars) — queued for render",
                     job["id"], len(image_prompt))

    lock = getattr(rt, "gen_lock", None)
    try:
        if lock is not None:
            async with lock:
                await _work()
        else:
            await _work()
    except WorkflowCancelled:
        log.info("job %s: art prompt cancelled mid-workflow", job["id"])
    except Exception as exc:
        log.exception("job %s art prompt phase failed", job["id"])
        _fail_job(db, job["id"], str(exc))


async def run_art_render(rt, job: dict) -> None:
    """Phase 'art_render': send the image prompt to the images backend and
    store the artwork on the track. Called by dj.job_runner (which marked
    the job running and holds the GPU alternation)."""
    db = rt.db
    outputs = _job_progress(job).get("outputs") or {}
    prompt = str(outputs.get("image_prompt") or "").strip()
    if not prompt:
        _fail_job(db, job["id"], "no image prompt in job outputs (lyrics phase failed?)")
        return
    track = db.one("SELECT id, status, title FROM tracks WHERE id = ?", (job["track_id"],))
    if track is None or track["status"] != "ready":
        _fail_job(db, job["id"], f"track {job['track_id']} is missing or not ready")
        return
    log.info("job %s: rendering art for track %s (%r), %d-char prompt",
             job["id"], job["track_id"], track["title"], len(prompt))
    image_bytes = await rt.images.generate(
        prompt, size=str(rt.cfg.get("art.size", "1024x1024")), seed=job.get("seed"))
    # The images call cannot be interrupted mid-flight; an admin
    # cancellation that landed while it ran discards the bytes instead of
    # overwriting art the admin asked to stop.
    row = db.one("SELECT status FROM generation_jobs WHERE id = ?", (job["id"],))
    if row is None or row["status"] != "running":
        log.info("job %s: cancelled during art render — discarding image", job["id"])
        return
    await library.store_art(db, rt.cfg, job["track_id"], image_bytes,
                            job_id=job["id"], seed=job.get("seed"),
                            workflow_id=job["workflow_id"], prompt=prompt)
    db.execute("UPDATE generation_jobs SET status = 'done', finished_at = ?,"
               " used_seed = ? WHERE id = ? AND status = 'running'",
               (iso(utcnow()), job.get("seed"), job["id"]))
    log.info("job %s: art stored for track %s", job["id"], job["track_id"])


# ------------------------------------------------------------------ audition CLI

def _sample_context() -> dict[str, Any]:
    """A plausible track for --dry-run/--run without a server."""
    return {
        "title": "Glass Cathedral",
        "theme_title": "Neon Rain",
        "style": "Dreamy synthwave with warm analog pads, a slow beat and "
                 "airy female vocals.",
        "lyrics": "[Verse]\nNeon rain on empty streets\n\n[Chorus]\nWe build "
                  "a glass cathedral in the dark",
        "track_tags": ["synthwave", "vocal", "dreamy"],
        "genre": "synthwave nightdrive",
        "tempo": 92,
        "music_key": "Am",
        "duration_s": 187.0,
        "is_variation": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="infiniradio.art",
                                 description="Art workflow audition CLI (no server).")
    ap.add_argument("--list", action="store_true", help="list workflows")
    ap.add_argument("--validate", action="store_true", help="validate and exit")
    ap.add_argument("--dry-run", metavar="NAME", help="render stage prompts, no LLM")
    ap.add_argument("--run", metavar="NAME", help="run a workflow end to end")
    ap.add_argument("--stage", metavar="ID", help="stop after this stage")
    ap.add_argument("--mock", action="store_true", help="use the mock LLM client")
    args = ap.parse_args()

    from .config import load_config
    cfg = load_config()
    try:
        specs = load_art_workflows(cfg)
    except WorkflowError as exc:
        raise SystemExit(f"art workflow error: {exc}")
    if not specs:
        raise SystemExit("no art workflows found "
                         f"under {cfg.path('art.workflows_dir')}")

    if args.list or args.validate:
        for s in specs.values():
            print(f"{s.name:24s} {'abstract' if s.abstract else ('on' if s.enabled else 'off')}"
                  f"  w={s.weight:g}  {len(s.stages)} stage(s)  {s.description}")
        print(f"{len(specs)} workflow(s) OK")
        return

    name = args.dry_run or args.run
    spec = specs.get(name or "")
    if spec is None:
        raise SystemExit(f"no art workflow named {name!r} "
                         f"(have: {', '.join(sorted(specs))})")
    ctx = dict(spec.vars)          # engine seeds [vars] defaults first,
    ctx.update(_sample_context())  # then the track context overrides
    ctx["params"] = spec.params
    if args.dry_run:
        for s in spec.stages:
            print(f"===== stage {s.id} ({s.template}) =====")
            print(spec.env.get_template(s.template).render(**ctx))
        return

    import asyncio

    async def _run() -> None:
        if args.mock:
            from .llm import MockLLMClient
            client = MockLLMClient()
        else:
            from .llm import make_llm_client
            client = make_llm_client(cfg)
            if client is None:
                raise SystemExit("[llm] not enabled — use --mock for pipeline testing")
        try:
            result = await lyrics.run_workflow(
                client, spec, initial_vars=ctx, stop_after=args.stage)
        finally:
            await client.close()
        for tr in result.attempts:
            print(f"--- {tr.stage_id} attempt {tr.attempt} "
                  f"({'ok' if tr.ok else tr.failure_text}) ---")
            print(tr.response_text[:2000])
        if result.ok:
            print("\nIMAGE PROMPT:")
            print(result.vars.get(spec.prompt_var, ""))
        else:
            raise SystemExit(f"workflow failed: {result.error}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
