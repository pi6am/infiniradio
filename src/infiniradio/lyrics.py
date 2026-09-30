"""Lyrics workflows: multi-stage LLM pipelines that write songs.

A workflow is a directory under lyrics.workflows_dir (default:
lyrics_workflows/) holding a workflow.toml control-flow file plus one Jinja
template per LLM prompt. Everything content-related lives in those files;
this module is the engine. Workflows are loaded and validated once at
startup with fail-fast errors (the names/ word-list rule: a typo is a
startup error, never a silent skip), and there is deliberately no hot
reload.

Each stage runs: render prompt → call the LLM → gate → extract → save
variables. Failed checks produce structured failure codes; a feedback
template composes the retry hint from them (the same template serves every
failure kind — it branches, e.g. {% if extract_failed("song_name") %}), and
the exchange continues in the same conversation thread. Extraction misses
are failures like any other, so they retry generically. Transport-level
trouble is retried inside the LLM client; an LLMError that survives that
counts as one stage attempt with code "llm_error".

Tags describe the OUTPUT track (e.g. "vocal", "english" — the final lyrics
language); the intermediate language a workflow drafts in is part of its
identity and travels as tracks.workflow_id.

Every attempt emits an AttemptTrace (prompt sent, reply, failures, token
usage) — the server persists those in job_stage_log, and the audition CLI
prints them. run_workflow() itself never touches the DB: it is an async
function over (client, spec, initial vars, prior progress), so the whole
pipeline is testable in-process.

Audition (repo root, no server involved):

    python -m infiniradio.lyrics --list
    python -m infiniradio.lyrics --validate
    python -m infiniradio.lyrics --dry-run jp-title-first
    python -m infiniradio.lyrics --run jp-title-first --mock
    python -m infiniradio.lyrics --run jp-title-first            # real [llm] endpoint
"""

from __future__ import annotations

import logging
import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jinja2

from .config import Config
from .llm.base import LLMClient, LLMError, Message

log = logging.getLogger(__name__)

# Every failure code the engine can report. Feedback templates branch on
# these; the set is closed so startup validation can catch template typos
# is unnecessary — but codes appearing in traces stay stable for queries.
FAILURE_CODES = ("too_long", "too_short", "missing_pattern", "missing_any",
                 "forbidden_present", "extract_failed", "llm_error")

# Names the engine itself puts into the template context; workflows cannot
# save or extract variables under these (their value would be clobbered).
RESERVED_VARS = frozenset({"title", "params", "tags", "failures", "failed",
                           "extract_failed", "attempt", "previous"})

_WORKFLOW_KEYS = {"name", "description", "enabled", "tags", "seed_title",
                  "params", "vars", "styled", "output", "stage"}
_STAGE_KEYS = {"id", "template", "saves", "block", "extract", "thread",
               "max_attempts", "feedback_template", "model", "temperature", "gate"}
_GATE_KEYS = {"max_chars", "min_chars", "all_of", "any_of", "forbidden"}
_OUTPUT_KEYS = {"title", "style", "lyrics", "tags", "max_tags"}


class WorkflowError(RuntimeError):
    """Bad workflow definition; the server must not start."""


class WorkflowCancelled(Exception):
    """The caller's guard said stop (e.g. the job was cancelled mid-run)."""

    def __init__(self, workflow: str, stage_id: str):
        super().__init__(f"workflow {workflow!r} cancelled before stage {stage_id!r}")
        self.workflow = workflow
        self.stage_id = stage_id


# ------------------------------------------------------------------ specs

@dataclass(frozen=True)
class Failure:
    code: str      # one of FAILURE_CODES
    check: str     # gate check name, extract variable, or "llm"
    detail: str    # human-readable, for logs/admin


@dataclass(frozen=True)
class GateSpec:
    max_chars: int | None = None
    min_chars: int | None = None
    all_of: tuple[str, ...] = ()     # every pattern must appear (AND)
    any_of: tuple[str, ...] = ()     # at least one must appear (OR)
    forbidden: tuple[str, ...] = ()  # none may appear


@dataclass(frozen=True)
class StageSpec:
    id: str
    template: str                    # file name inside the workflow directory
    saves: str | None = None         # variable name for the full raw reply
    block: str | None = None         # song-block variable: the reply's fenced song
                                     # (or bare text) with commentary and a leading
                                     # [Title] section removed — see extract_song_block
    extract: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (variable, ordered
                                     # alternative regexes, each with 1 group; the
                                     # first match wins) — run against the RAW reply
    thread: str = "new"              # "new" | "continue" (previous stage's thread)
    max_attempts: int = 1
    feedback_template: str | None = None       # rendered with the failures
    model: str | None = None         # per-stage [llm] overrides (A/B testing)
    temperature: float | None = None
    gate: GateSpec | None = None     # applies to the song block when block is set


@dataclass(frozen=True)
class OutputSpec:
    style: str                       # required variable names
    lyrics: str
    title: str | None = None         # optional: falls back to the seed title
    tags: str | None = None          # optional: comma-separated tag list
    max_tags: int | None = None      # truncate the dynamic (LLM-picked) tags
                                     # to this many, most relevant first;
                                     # static tags don't count against it


@dataclass(frozen=True)
class WorkflowSpec:
    name: str                        # == directory name (the slug used in the DB)
    directory: Path
    description: str
    enabled: bool
    static_tags: tuple[str, ...]     # tags every output track carries
    seed_title: bool                 # generate a names/-list title as "title"
    params: dict[str, Any]           # available to every template as params.*
    stages: tuple[StageSpec, ...]
    output: OutputSpec
    env: jinja2.Environment
    vars: dict[str, str] = field(default_factory=dict)  # default values for
                                     # style-override variables; templates read
                                     # them at top level ({{ style_guide }})
    styled: bool = False             # the DJ picks a style for this workflow


# ------------------------------------------------------------------ traces

@dataclass
class AttemptTrace:
    """One LLM call: everything needed to debug or replay it."""

    stage_id: str
    attempt: int
    prompt_text: str
    response_text: str
    failures: list[Failure] = field(default_factory=list)
    ok: bool = False
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    vars_snapshot: dict[str, str] = field(default_factory=dict)

    @property
    def failure_text(self) -> str:
        return "; ".join(f"{f.code}[{f.check}] {f.detail}" for f in self.failures)


@dataclass
class StageResult:
    stage_id: str
    ok: bool
    attempts: int = 0
    skipped: bool = False            # resumed over (output already in progress)
    failures: list[Failure] = field(default_factory=list)


@dataclass
class WorkflowResult:
    ok: bool
    error: str = ""
    vars: dict[str, Any] = field(default_factory=dict)
    title: str | None = None         # extracted song name, if the workflow has one
    style: str = ""
    lyrics: str = ""
    tags: list[str] = field(default_factory=list)  # static + dynamic, normalized
    stages: list[StageResult] = field(default_factory=list)
    attempts: list[AttemptTrace] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)  # for resume/persist


# ------------------------------------------------------------------ loading

def _jinja_env(directory: Path) -> jinja2.Environment:
    # ChoiceLoader so a future `base = ...` workflow-inheritance scheme is a
    # loader search path, not a redesign. Text output: no autoescaping;
    # strict_undefined so a typo'd variable fails the render, not silently "".
    return jinja2.Environment(
        loader=jinja2.ChoiceLoader([jinja2.FileSystemLoader(str(directory))]),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _only_keys(data: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise WorkflowError(
            f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}")


def _compile(pattern: str, where: str, *, groups: int) -> re.Pattern:
    try:
        rx = re.compile(pattern, re.DOTALL)
    except re.error as exc:
        raise WorkflowError(f"{where}: bad regex {pattern!r}: {exc}") from exc
    if rx.groups != groups:
        raise WorkflowError(
            f"{where}: regex {pattern!r} must contain exactly {groups} "
            f"capturing group(s), has {rx.groups}")
    return rx


def _var_name(name: str, where: str) -> str:
    if not name.isidentifier() or not name[0].islower():
        raise WorkflowError(f"{where}: variable name {name!r} must be a lowercase identifier")
    if name in RESERVED_VARS:
        raise WorkflowError(
            f"{where}: variable name {name!r} is reserved by the engine "
            f"({', '.join(sorted(RESERVED_VARS))})")
    return name


def _parse_stage(data: dict, where: str) -> StageSpec:
    _only_keys(data, _STAGE_KEYS, where)
    stage_id = str(data.get("id") or "").strip()
    if not stage_id.isidentifier():
        raise WorkflowError(f"{where}: stage id {stage_id!r} must be an identifier")
    template = str(data.get("template") or "").strip()
    if not template:
        raise WorkflowError(f"{where}: 'template' is required")
    saves = data.get("saves")
    if saves is not None:
        saves = _var_name(str(saves), f"{where}: saves")
    block = data.get("block")
    if block is not None:
        block = _var_name(str(block), f"{where}: block")
    extract_list: list[tuple[str, tuple[str, ...]]] = []
    for key, val in (data.get("extract") or {}).items():
        name = _var_name(str(key), f"{where}: extract")
        raw_patterns = val if isinstance(val, list) else [val]
        patterns = tuple(str(p) for p in raw_patterns)
        if not patterns or any(not p.strip() for p in patterns):
            raise WorkflowError(f"{where}: extract {name!r} needs at least one pattern")
        for p in patterns:
            _compile(p, f"{where}: extract {name!r}", groups=1)
        extract_list.append((name, patterns))
    extract = tuple(extract_list)
    thread = str(data.get("thread") or "new")
    if thread not in ("new", "continue"):
        raise WorkflowError(f"{where}: thread must be 'new' or 'continue', got {thread!r}")
    gate_data = data.get("gate") or {}
    _only_keys(gate_data, _GATE_KEYS, f"{where}: gate")
    gate = GateSpec(
        max_chars=gate_data.get("max_chars"),
        min_chars=gate_data.get("min_chars"),
        all_of=tuple(str(p) for p in gate_data.get("all_of") or ()),
        any_of=tuple(str(p) for p in gate_data.get("any_of") or ()),
        forbidden=tuple(str(p) for p in gate_data.get("forbidden") or ()),
    ) if gate_data else None
    for attr in ("all_of", "any_of", "forbidden"):
        for pattern in getattr(gate, attr) if gate else ():
            _compile(pattern, f"{where}: gate.{attr}", groups=0)
    return StageSpec(
        id=stage_id, template=template, saves=saves, block=block, extract=extract,
        thread=thread,
        max_attempts=max(1, int(data.get("max_attempts") or 1)),
        feedback_template=(str(data["feedback_template"])
                           if data.get("feedback_template") else None),
        model=(str(data["model"]) if data.get("model") else None),
        temperature=(float(data["temperature"]) if data.get("temperature") is not None else None),
        gate=gate,
    )


def load_workflow(directory: Path) -> WorkflowSpec:
    wf_path = directory / "workflow.toml"
    where = str(wf_path)
    try:
        data = tomllib.loads(wf_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise WorkflowError(f"{where}: cannot read workflow: {exc}") from exc
    _only_keys(data, _WORKFLOW_KEYS, where)

    name = str(data.get("name") or "").strip()
    if name != directory.name:
        raise WorkflowError(
            f"{where}: name {name!r} must match the directory name "
            f"{directory.name!r} (copying a workflow? rename both)")

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
    if stages[0].thread == "continue":
        raise WorkflowError(f"{where}: the first stage cannot thread = 'continue'")

    output_data = data.get("output") or {}
    _only_keys(output_data, _OUTPUT_KEYS, f"{where}: output")
    produced = ({s.saves for s in stages if s.saves}
                | {s.block for s in stages if s.block}
                | {k for s in stages for k, _ in s.extract})
    for key in ("style", "lyrics"):
        var = output_data.get(key)
        if not var:
            raise WorkflowError(f"{where}: output.{key} is required")
        if str(var) not in produced:
            raise WorkflowError(
                f"{where}: output.{key} = {var!r} is not saved or extracted by any stage")
    for key in ("title", "tags"):
        var = output_data.get(key)
        if var and str(var) not in produced:
            raise WorkflowError(
                f"{where}: output.{key} = {var!r} is not saved or extracted by any stage")
    output = OutputSpec(
        style=str(output_data["style"]), lyrics=str(output_data["lyrics"]),
        title=(str(output_data["title"]) if output_data.get("title") else None),
        tags=(str(output_data["tags"]) if output_data.get("tags") else None),
        max_tags=(int(output_data["max_tags"]) if output_data.get("max_tags") is not None
                  else None))
    if output.max_tags is not None and output.max_tags < 1:
        raise WorkflowError(f"{where}: output.max_tags must be >= 1")

    env = _jinja_env(directory)
    vars_defaults: dict[str, str] = {}
    for key, value in (data.get("vars") or {}).items():
        if not str(key).isidentifier() or not str(key)[0].islower():
            raise WorkflowError(f"{where}: vars: {key!r} must be a lowercase identifier")
        if str(key) in RESERVED_VARS:
            raise WorkflowError(f"{where}: vars: {key!r} is reserved by the engine")
        if not isinstance(value, str):
            raise WorkflowError(f"{where}: vars: {key!r} must be a string "
                                "(defaults for style overrides)")
        vars_defaults[str(key)] = value
    for s in stages:
        try:
            env.get_template(s.template)
        except jinja2.TemplateError as exc:
            raise WorkflowError(f"{where}: stage {s.id!r}: template {s.template!r}: {exc}") from exc
        if s.feedback_template:
            try:
                env.get_template(s.feedback_template)
            except jinja2.TemplateError as exc:
                raise WorkflowError(
                    f"{where}: stage {s.id!r}: feedback_template "
                    f"{s.feedback_template!r}: {exc}") from exc

    return WorkflowSpec(
        name=name,
        directory=directory,
        description=str(data.get("description") or ""),
        enabled=bool(data.get("enabled", True)),
        static_tags=tuple(str(t).strip().lower() for t in data.get("tags") or () if str(t).strip()),
        seed_title=bool(data.get("seed_title", False)),
        params=dict(data.get("params") or {}),
        stages=stages,
        output=output,
        env=env,
        # [vars]: defaults for style-override variables. Styles supply the
        # values; the workflow supplies the fallbacks (e.g. an unstyled run
        # keeps its natural rhyme discipline). Same naming rules as styles.
        vars=vars_defaults,
        styled=bool(data.get("styled", False)),
    )


def load_workflows(cfg: Config) -> dict[str, WorkflowSpec]:
    """Load every workflow under lyrics.workflows_dir (validated, ordered)."""
    directory = cfg.path("lyrics.workflows_dir")
    if not directory.is_dir():
        log.info("no lyrics workflows directory at %s — lyrics generation off", directory)
        return {}
    specs: dict[str, WorkflowSpec] = {}
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir() or not (sub / "workflow.toml").is_file():
            continue  # plain files and template-share dirs are not workflows
        spec = load_workflow(sub)
        specs[spec.name] = spec
    enabled = sum(s.enabled for s in specs.values())
    log.info("loaded %d lyrics workflow(s) from %s (%d enabled)",
             len(specs), directory, enabled)
    return specs


# ------------------------------------------------------------------ execution

def parse_tag_list(text: str) -> list[str]:
    """Tags from a free-form reply, best effort: splits on commas and
    newlines, tolerates list markers (1. / - / *) and quoting, normalizes
    case, dedupes in order. `'1. Rock\\n- "dream pop"' -> ['rock', 'dream
    pop']`. Garbage prose survives this only to be filtered against the
    vocabulary later."""
    out: list[str] = []
    for part in re.split(r"[,\n]", text.lower()):
        tag = re.sub(r"^(?:\d+[.)]|[-*•·>]+)\s*", "", part.strip())  # markers
        tag = tag.strip().strip("`\"'“”‘’").strip()
        if tag and tag not in out:
            out.append(tag)
    return out


# ------------------------------------------------------------------ song blocks

# A fenced block: ``` opener (optional language tag), content, ``` closer.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# A section header on its own line: [Intro], [Verse 1], [Chorus] ...
_GENERIC_SECTION_RE = r"(?m)^\s*\[[^\]\n]{1,40}\]\s*$"
# A [Title] section: the tag line plus everything up to the next section
# header (the title text itself is dropped — the song's name lives in
# [output] title / tracks.title, not in the lyrics).
_TITLE_SECTION_RE = re.compile(
    r"(?ms)^[ \t]*\[Title\][^\n]*\n.*?(?=^[ \t]*\[[^\]\n]+\][ \t]*$|\Z)")


def _strip_title_section(text: str) -> str:
    m = _TITLE_SECTION_RE.search(text)
    if not m:
        return text
    return (text[:m.start()] + text[m.end():]).strip("\n")


def extract_song_block(text: str, section_patterns: tuple[str, ...] = ()) -> str:
    """The song inside an LLM reply, robust to output-style variation.

    Quantized models post-"thinking" wrap the song in commentary, code
    fences, both, or neither. Rules, in order:

    1. the first fenced block whose content contains a section header
       (patterns default to the generic [Header]-on-its-own-line shape)
       is the song;
    2. otherwise the reply is treated as a bare song: a [Title] section is
       stripped and, when section headers are present, everything before
       the first one (leading commentary prose) is dropped;
    3. fenced content also gets a leading [Title] section stripped.

    Parsing beats retrying here: one local pass, no extra LLM round-trip,
    and the raw reply stays in the stage trace for inspection.
    """
    patterns = section_patterns or (_GENERIC_SECTION_RE,)
    for m in _FENCE_RE.finditer(text):
        content = m.group(1)
        if any(re.search(p, content, re.DOTALL) for p in patterns):
            return _strip_title_section(content).strip()
    bare = _strip_title_section(text)
    header = re.search(_GENERIC_SECTION_RE, bare)
    if header:
        bare = bare[header.start():]
    return bare.strip()


def _section_patterns(stage: StageSpec) -> tuple[str, ...]:
    """What "looks like a song" for block extraction: the stage's own gate
    declaration (any_of, else all_of), else the generic header shape."""
    if stage.gate:
        if stage.gate.any_of:
            return stage.gate.any_of
        if stage.gate.all_of:
            return stage.gate.all_of
    return (_GENERIC_SECTION_RE,)


def _evaluate(stage: StageSpec, reply_text: str,
              gate_text: str) -> tuple[list[Failure], dict[str, str]]:
    """All gate checks + extrations, evaluated together; never saves.

    Gates check gate_text (the song block when the stage declares one);
    extract regexes run against the raw reply (e.g. a Name: line outside
    the fence).
    """
    failures: list[Failure] = []
    extracted: dict[str, str] = {}
    g = stage.gate
    if g is not None:
        if g.max_chars is not None and len(gate_text) > g.max_chars:
            failures.append(Failure("too_long", "max_chars",
                                    f"{len(gate_text)} chars > {g.max_chars}"))
        if g.min_chars is not None and len(gate_text) < g.min_chars:
            failures.append(Failure("too_short", "min_chars",
                                    f"{len(gate_text)} chars < {g.min_chars}"))
        for pattern in g.all_of:
            if not re.search(pattern, gate_text, re.DOTALL):
                failures.append(Failure("missing_pattern", "all_of",
                                        f"required pattern {pattern!r} not found"))
        if g.any_of and not any(re.search(p, gate_text, re.DOTALL) for p in g.any_of):
            failures.append(Failure("missing_any", "any_of",
                                    "none of the accepted patterns found "
                                    "(e.g. section headers)"))
        for pattern in g.forbidden:
            if re.search(pattern, gate_text, re.DOTALL):
                failures.append(Failure("forbidden_present", "forbidden",
                                        f"forbidden pattern {pattern!r} found"))
    for key, patterns in stage.extract:
        value: str | None = None
        for pattern in patterns:      # ordered alternatives, first match wins
            m = re.search(pattern, reply_text, re.DOTALL)
            if m:
                value = m.group(1)
                break
        if value is None or not value.strip():
            failures.append(Failure(
                "extract_failed", key,
                f"could not extract {key!r} (tried {len(patterns)} pattern(s): "
                + "; ".join(patterns) + ")"))
        else:
            extracted[key] = value.strip()
    return failures, extracted


def _render(env: jinja2.Environment, spec: WorkflowSpec, template: str,
            vars_: dict[str, Any], tag_vocabulary: list[str] | None,
            **extra: Any) -> str:
    ctx: dict[str, Any] = dict(vars_)
    ctx["params"] = spec.params
    ctx["tags"] = list(tag_vocabulary or [])
    ctx.update(extra)
    try:
        return env.get_template(template).render(**ctx)
    except jinja2.TemplateError as exc:
        raise WorkflowError(f"workflow {spec.name!r}: template {template!r}: {exc}") from exc


def _progress_snapshot(completed: list[str], vars_: dict[str, Any],
                       threads: dict[str, list[Message]],
                       current_thread: str | None) -> dict[str, Any]:
    return {
        "completed": completed,
        "vars": vars_,
        "threads": {k: [[m.role, m.content] for m in v] for k, v in threads.items()},
        "current_thread": current_thread,
    }


def _usage_add(total: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        try:
            total[k] = total.get(k, 0) + int(v)
        except (TypeError, ValueError):
            pass


async def run_workflow(client: LLMClient, spec: WorkflowSpec, *,
                       initial_vars: dict[str, Any] | None = None,
                       tag_vocabulary: list[str] | None = None,
                       progress: dict[str, Any] | None = None,
                       guard: Callable[[], bool] | None = None,
                       on_attempt: Callable[[AttemptTrace], None] | None = None,
                       on_stage: Callable[[str, dict[str, Any]], None] | None = None,
                       stop_after: str | None = None,
                       ) -> WorkflowResult:
    """Execute a workflow spec against an LLM client.

    ``progress`` resumes at the first stage not in progress["completed"] —
    saved variables and conversation threads are restored, so a job retry
    never re-pays for completed stages. ``guard`` is checked between stages
    and attempts (the job runner uses it to notice cancellation);
    WorkflowCancelled is raised when it turns false. ``on_attempt`` fires
    after every LLM call so traces can be persisted as they happen, and
    ``on_stage`` fires with (stage_id, progress snapshot) after every stage
    COMPLETES — persisting that snapshot is what makes live progress
    visible and failure-resume cheap. ``stop_after`` ends the run
    successfully once that stage completes (audition/debugging aid).
    """
    progress = progress or {}
    completed: list[str] = list(progress.get("completed") or [])
    # Workflow [vars] defaults first, then caller-supplied variables (seed
    # title, style overrides), then any resumable state. The caller never
    # needs to know which defaults a workflow declares.
    vars_: dict[str, Any] = dict(getattr(spec, "vars", {}) or {})
    vars_.update(initial_vars or {})
    vars_.update(progress.get("vars") or {})
    threads: dict[str, list[Message]] = {
        key: [Message(role=r, content=c) for r, c in msgs]
        for key, msgs in (progress.get("threads") or {}).items()
    }
    current_thread: str | None = progress.get("current_thread")

    result = WorkflowResult(ok=False)
    usage_total: dict[str, int] = {}

    def finish(ok: bool, error: str = "") -> WorkflowResult:
        result.ok = ok
        result.error = error
        result.vars = vars_
        result.usage = usage_total
        result.progress = _progress_snapshot(completed, vars_, threads, current_thread)
        if ok:
            result.title = str(vars_.get(spec.output.title)) if spec.output.title else None
            result.style = str(vars_.get(spec.output.style, "") or "")
            result.lyrics = str(vars_.get(spec.output.lyrics, "") or "")
            dynamic = parse_tag_list(str(vars_.get(spec.output.tags, "") or "")) \
                if spec.output.tags else []
            if tag_vocabulary:
                # Sanitize: keep only known tags. A thinking-leak reply
                # mentions plenty of words — only vocabulary matches count,
                # so 30k chars of deliberation still yields a clean (or
                # empty) pick, with no LLM round-trip spent on it.
                allowed = {t.lower() for t in tag_vocabulary}
                dynamic = [t for t in dynamic if t in allowed]
            if spec.output.max_tags is not None:
                dynamic = dynamic[:spec.output.max_tags]
            tags = list(spec.static_tags)
            tags += [t for t in dynamic if t not in tags]
            result.tags = tags
        return result

    for stage in spec.stages:
        if stage.id in completed:
            if stage.thread == "new":
                current_thread = stage.id
            result.stages.append(StageResult(stage.id, True, 0, skipped=True))
            continue
        if guard is not None and not guard():
            raise WorkflowCancelled(spec.name, stage.id)

        if stage.thread == "new":
            thread: list[Message] = []
            threads[stage.id] = thread
            current_thread = stage.id
        else:
            thread = threads.get(current_thread or "")
            if thread is None:  # validation makes this unreachable; defensive
                return finish(False, f"stage {stage.id!r}: no previous thread to continue")

        sr = StageResult(stage.id, False)
        result.stages.append(sr)
        first_prompt: str | None = None
        prev_failures: list[Failure] = []
        prev_reply = ""

        for attempt in range(1, stage.max_attempts + 1):
            if guard is not None and not guard():
                raise WorkflowCancelled(spec.name, stage.id)
            if attempt == 1 or stage.feedback_template is None:
                prompt = _render(spec.env, spec, stage.template, vars_, tag_vocabulary)
                if attempt == 1:
                    first_prompt = prompt
            else:
                prompt = _render(
                    spec.env, spec, stage.feedback_template, vars_, tag_vocabulary,
                    failures=prev_failures,
                    failed=lambda code: any(f.code == code for f in prev_failures),
                    extract_failed=lambda key: any(
                        f.code == "extract_failed" and f.check == key for f in prev_failures),
                    attempt=attempt,
                    previous=prev_reply,
                )

            msg = Message("user", prompt)
            t0 = time.monotonic()
            try:
                reply = await client.chat(
                    thread + [msg], model=stage.model, temperature=stage.temperature)
            except LLMError as exc:
                # The client already exhausted transport retries; re-sending
                # later (endpoint may recover) costs one stage attempt. The
                # conversation is untouched — the model never saw the request.
                trace = AttemptTrace(stage.id, attempt, prompt, "",
                                     [Failure("llm_error", "llm", str(exc))])
                result.attempts.append(trace)
                if on_attempt:
                    on_attempt(trace)
                sr.attempts = attempt
                sr.failures = trace.failures
                prev_failures = trace.failures
                log.warning("workflow %s stage %s attempt %d: llm error: %s",
                            spec.name, stage.id, attempt, exc)
                continue

            block_text = extract_song_block(reply.content, _section_patterns(stage)) \
                if stage.block else ""
            failures, extracted = _evaluate(
                stage, reply.content, block_text if stage.block else reply.content)
            ok = not failures
            vars_snapshot = dict(extracted)
            if ok and stage.block:
                vars_snapshot[stage.block] = block_text
            if ok and stage.saves:
                vars_snapshot[stage.saves] = reply.content.strip()
            trace = AttemptTrace(stage.id, attempt, prompt, reply.content,
                                 failures, ok, reply.model, reply.usage,
                                 time.monotonic() - t0, vars_snapshot)
            result.attempts.append(trace)
            if on_attempt:
                on_attempt(trace)
            _usage_add(usage_total, reply.usage)

            # The exchange happened; failed attempts stay in the history the
            # feedback turn continues from.
            thread.append(msg)
            thread.append(Message("assistant", reply.content))

            if ok:
                if stage.block:
                    vars_[stage.block] = block_text
                if stage.saves:
                    vars_[stage.saves] = reply.content.strip()
                vars_.update(extracted)
                sr.ok = True
                sr.attempts = attempt
                completed.append(stage.id)
                if on_stage is not None:
                    # Snapshot as soon as a stage completes: persisting this
                    # makes progress visible mid-run and lets a failed run's
                    # retry resume here instead of starting over.
                    on_stage(stage.id, _progress_snapshot(
                        completed, vars_, threads, current_thread))
                log.info("workflow %s: stage %s ok (attempt %d, %.1fs, %d tokens)",
                         spec.name, stage.id, attempt, trace.duration_s,
                         reply.usage.get("completion_tokens", 0))
                break
            sr.attempts = attempt
            sr.failures = failures
            prev_failures = failures
            prev_reply = reply.content
            log.info("workflow %s: stage %s attempt %d rejected: %s",
                     spec.name, stage.id, attempt, trace.failure_text)
        else:
            return finish(False, (
                f"stage {stage.id!r} failed after {stage.max_attempts} attempt(s): "
                + "; ".join(f"{f.code}[{f.check}]" for f in sr.failures)))

        if stop_after == stage.id:
            return finish(True)

    return finish(True)


# ------------------------------------------------------------------ audition CLI

def _cli_client(cfg: Config, mock: bool) -> LLMClient:
    from .llm import make_llm_client
    from .llm.mock import MockLLMClient
    if mock:
        return MockLLMClient()
    client = make_llm_client(cfg)
    if client is None:
        raise SystemExit("no LLM configured — enable [llm] in config.toml or pass --mock")
    return client


def _print_trunc(label: str, text: str, limit: int) -> None:
    """Print text verbatim (copy-paste friendly: no added indentation, no
    added trailing whitespace) under a header, truncating unless limit is
    huge. Whatever whitespace the LLM produced is preserved as-is."""
    text = text.rstrip("\n")
    shown = text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit} more chars]"
    print(f"--- {label} " + "-" * max(0, 60 - len(label)), flush=True)
    print(shown, flush=True)


def main() -> None:
    import argparse
    import asyncio
    import random as _random

    from .config import load_config
    from . import naming

    ap = argparse.ArgumentParser(
        prog="python -m infiniradio.lyrics",
        description="audition lyrics workflows: validate, dry-run prompts, "
                    "or run the full pipeline (mock or real LLM, no server)")
    ap.add_argument("-c", "--config", default=None, help="path to config.toml")
    ap.add_argument("--list", action="store_true", help="list workflows")
    ap.add_argument("--validate", action="store_true",
                    help="load + validate every workflow (also done at startup)")
    ap.add_argument("--dry-run", metavar="WORKFLOW",
                    help="render every template with placeholder variables; no LLM calls")
    ap.add_argument("--run", metavar="WORKFLOW", help="execute the workflow end to end")
    ap.add_argument("--mock", action="store_true",
                    help="use the scripted mock LLM (no endpoint needed)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the title generator")
    ap.add_argument("--style", metavar="NAME", default=None,
                    help="run with a style (from styles/*.toml seeds): fixed + weighted "
                         "variables resolved from --seed, passed to the workflow as "
                         "template variables")
    ap.add_argument("--stage", metavar="STAGE_ID",
                    help="with --run: stop after this stage (debugging aid)")
    ap.add_argument("--full", action="store_true", help="print full texts, not truncated")
    ap.add_argument("--title", default=None, help="override the seed title")
    args = ap.parse_args()

    cfg = load_config(args.config)
    workflows = load_workflows(cfg)
    if args.list or (not args.validate and not args.dry_run and not args.run):
        for spec in workflows.values():
            print(f"{'on ' if spec.enabled else 'off'}  {spec.name}: "
                  f"{spec.description}  [{len(spec.stages)} stages, tags: "
                  f"{', '.join(spec.static_tags) or '—'}]")
        return

    if args.validate:
        print(f"{len(workflows)} workflow(s) validated OK")
        return

    spec = workflows.get(args.dry_run or args.run)
    if spec is None:
        raise SystemExit(f"no such workflow (see --list); have: "
                         f"{', '.join(workflows) or 'none'}")

    rng = _random.Random(args.seed)
    title = args.title or naming.title(rng)

    # A style (seed-file view — the station reads styles from the database):
    # fixed vars + one seeded weighted choice per key, merged over the
    # workflow's [vars] defaults.
    style_overrides: dict[str, str] = {}
    if args.style:
        from . import styles as styles_mod
        try:
            style = styles_mod.load_seed_style(cfg, args.style)
        except styles_mod.StyleError as exc:
            raise SystemExit(f"error: {exc}")
        style_overrides = styles_mod.resolve(style.vars, style.weighted, rng.getrandbits(31))
        print(f"style {style.name!r} resolved with seed {args.seed}:")
        for key, value in style_overrides.items():
            print(f"  {key} = {value[:80]}{'…' if len(value) > 80 else ''}")

    if args.dry_run:
        from . import library as library_mod
        llm_tags, _static = library_mod.load_tag_seeds(cfg)
        vocab = llm_tags or ["pop", "rock", "male vocal", "female vocal"]
        produced = {s.saves for s in spec.stages if s.saves} | {
            k for s in spec.stages for k, _ in s.extract}
        placeholders = {name: f"⟨{name}: output of an earlier stage⟩" for name in produced}
        vars_: dict[str, Any] = {"title": title, **spec.vars, **style_overrides,
                                 **placeholders}
        # (the engine layers spec.vars the same way at run time)
        fake_failures = [Failure(code, "check", f"fake {code} for template checking")
                         for code in FAILURE_CODES]
        for stage in spec.stages:
            _print_trunc(f"{stage.id} → {stage.template}", _render(
                spec.env, spec, stage.template, vars_, vocab),
                10**9 if args.full else 1500)
            if stage.feedback_template:
                _print_trunc(f"{stage.id} (feedback) → {stage.feedback_template}", _render(
                    spec.env, spec, stage.feedback_template, vars_, vocab,
                    failures=fake_failures,
                    failed=lambda code: True,
                    extract_failed=lambda key: True,
                    attempt=2, previous="⟨the rejected reply⟩"),
                    10**9 if args.full else 1000)
        return

    async def _run() -> None:
        client = _cli_client(cfg, args.mock)
        from . import library as library_mod
        llm_tags, _static = library_mod.load_tag_seeds(cfg)
        limit = 10**9 if args.full else 1200
        t0 = time.monotonic()

        def show(trace) -> None:
            """Stream each LLM exchange as it completes (long pipelines
            shouldn't be silent until the end)."""
            status = "ok" if trace.ok else "REJECTED: " + trace.failure_text
            print(f"\n=== stage {trace.stage_id} attempt {trace.attempt} "
                  f"({trace.duration_s:.1f}s, {trace.usage.get('completion_tokens', '?')} tok) "
                  f"— {status}", flush=True)
            if trace.response_text:
                _print_trunc("prompt", trace.prompt_text, limit)
                _print_trunc("reply", trace.response_text, limit)

        try:
            # The engine layers workflow [vars] defaults under these.
            initial: dict[str, Any] = dict(style_overrides)
            if spec.seed_title:
                initial["title"] = title
            result = await run_workflow(
                client, spec, initial_vars=initial,
                tag_vocabulary=llm_tags or ["pop", "rock", "male vocal", "female vocal"],
                stop_after=args.stage, on_attempt=show)
        finally:
            await client.close()
        wall = time.monotonic() - t0
        print(f"\n{'=' * 72}", flush=True)
        mins = int(wall // 60)
        print(f"{'OK' if result.ok else 'FAILED'} in "
              f"{f'{mins}m ' if mins else ''}{wall % 60:.0f}s — "
              f"{len(result.stages)} stage(s), {len(result.attempts)} LLM call(s), "
              f"{result.usage.get('completion_tokens', 0)} completion token(s)",
              flush=True)
        if not result.ok:
            print(f"FAILED: {result.error}")
            raise SystemExit(1)
        print(f"title: {result.title!r}")
        print(f"tags: {', '.join(result.tags)}")
        _print_trunc("style", result.style, 10**9 if args.full else 600)
        _print_trunc("lyrics", result.lyrics, 10**9 if args.full else 2500)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
