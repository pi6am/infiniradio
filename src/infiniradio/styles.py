"""Song styles: named bundles of workflow variables, picked by the DJ.

A style is what listener votes steer (the bandit unit between raw
randomness and whole workflows): a fixed set of variable overrides
(``[vars]``, e.g. ``style_guide``) plus weighted choices resolved by
seeded RNG at commission time (``[weighted]``, e.g. ``structure_guide``).
Styles follow the prompts lifecycle: ``styles/*.toml`` seeds are imported
on first boot, afterwards the database is the source of truth (admin CRUD,
re-import, export).

Content conventions (enforced by taste, not by code — the engine knows
none of these keys):

- ``style_guide``: general direction — genre, mood, energy. Deliberately
  NOT in YuE2 style-prompt format: several stages consume it and it should
  steer, not dictate, the final render prompt.
- ``structure_guide``: the STRUCTURE of the lyrics — lines per verse,
  chorus repetition, line lengths, how much repetition to aim for. This is
  the artificial-variation lever; unguided models always write the same
  four-line verses.
- ``rhyme_guide``: rhyme discipline. Styles that favor repetitive songs
  weight stricter rhyme patterns higher; the correlation between guides
  is expressed by declaring the weights in the same style (or by one
  weighted key that templates expand).

The engine stays generic: resolved overrides reach a workflow as plain
template variables, exactly like the seed title does.
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lyrics import RESERVED_VARS

_STYLE_KEYS = {"name", "description", "enabled", "tags", "vars", "weighted"}


class StyleError(RuntimeError):
    """Bad style definition; rejected at import/form-save time."""


@dataclass(frozen=True)
class StyleDef:
    name: str
    description: str
    enabled: bool
    tags: tuple[str, ...]
    vars: dict[str, str]                          # fixed overrides
    weighted: dict[str, tuple[tuple[float, str], ...]]  # key -> ((weight, value), ...)


def _var_key(name: str, where: str) -> str:
    if not name.isidentifier() or not name[0].islower():
        raise StyleError(f"{where}: variable name {name!r} must be a lowercase identifier")
    if name in RESERVED_VARS:
        raise StyleError(f"{where}: variable name {name!r} is reserved by the engine")
    return name


def _parse_vars_weighted(data: dict, where: str) -> tuple[dict[str, str],
                                                          dict[str, tuple[tuple[float, str], ...]]]:
    unknown = sorted(set(data) - {"vars", "weighted"})
    if unknown:
        raise StyleError(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; "
                         "expected [vars] and/or [weighted]")
    vars_ = {k: v for k, v in (data.get("vars") or {}).items()}
    for key, value in vars_.items():
        _var_key(str(key), f"{where}: vars")
        if not isinstance(value, str) or not value.strip():
            raise StyleError(f"{where}: vars.{key} must be a non-empty string")
    weighted: dict[str, tuple[tuple[float, str], ...]] = {}
    for key, entries in (data.get("weighted") or {}).items():
        _var_key(str(key), f"{where}: weighted")
        if str(key) in vars_:
            raise StyleError(f"{where}: {key!r} appears in both [vars] and [weighted]")
        if not isinstance(entries, list) or not entries:
            raise StyleError(f"{where}: weighted.{key} needs at least one "
                             "{ weight = N, value = \"...\" } entry")
        parsed: list[tuple[float, str]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"weight", "value"}:
                raise StyleError(f"{where}: weighted.{key} entries must be "
                                 '{{ weight = N, value = "..." }} inline tables')
            weight, value = entry["weight"], entry["value"]
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise StyleError(f"{where}: weighted.{key}: weight must be > 0, got {weight!r}")
            if not isinstance(value, str) or not value.strip():
                raise StyleError(f"{where}: weighted.{key}: value must be a non-empty string")
            parsed.append((float(weight), value))
        weighted[str(key)] = tuple(parsed)
    return vars_, weighted


def parse_style(data: dict, where: str) -> StyleDef:
    """Validate a parsed style seed file (or admin form payload)."""
    unknown = sorted(set(data) - _STYLE_KEYS)
    if unknown:
        raise StyleError(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; "
                         f"allowed: {', '.join(sorted(_STYLE_KEYS))}")
    name = str(data.get("name") or "").strip()
    if not name:
        raise StyleError(f"{where}: 'name' is required")
    vars_, weighted = _parse_vars_weighted(
        {"vars": data.get("vars"), "weighted": data.get("weighted")}, where)
    return StyleDef(
        name=name,
        description=str(data.get("description") or ""),
        enabled=bool(data.get("enabled", True)),
        tags=tuple(str(t).strip().lower() for t in data.get("tags") or () if str(t).strip()),
        vars=vars_,
        weighted=weighted,
    )


def load_style_file(path: Path) -> StyleDef:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise StyleError(f"{path}: cannot read style: {exc}") from exc
    return parse_style(data, str(path))


def parse_definition(text: str, where: str) -> tuple[dict[str, str],
                                                     dict[str, tuple[tuple[float, str], ...]]]:
    """The admin form's definition body: just the [vars]/[weighted] tables."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise StyleError(f"{where}: not valid TOML: {exc}") from exc
    return _parse_vars_weighted(data, where)


def render_definition(vars_: dict[str, str],
                      weighted: dict[str, tuple[tuple[float, str], ...]]) -> str:
    """DB -> definition body for the admin form (round-trips parse_definition)."""
    def q(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)  # JSON string escaping is valid TOML

    lines: list[str] = []
    if vars_ or not weighted:
        lines.append("[vars]")
        for key, value in vars_.items():
            lines.append(f"{key} = {q(value)}")
    if weighted:
        if lines:
            lines.append("")
        lines.append("[weighted]")
        for key, entries in weighted.items():
            lines.append(f"{key} = [")
            for weight, value in entries:
                w = int(weight) if weight == int(weight) else weight
                lines.append(f"  {{ weight = {w}, value = {q(value)} }},")
            lines.append("]")
    return "\n".join(lines) + "\n"


def resolve(vars_: dict[str, Any], weighted: dict[str, Any], seed: int) -> dict[str, str]:
    """Fixed vars + one weighted choice per key, drawn from a seeded RNG.

    Deterministic for a given (definition, seed) — the job's seed makes
    every commission reproducible from the job row alone.
    """
    rng = random.Random(seed)
    out = {str(k): str(v) for k, v in vars_.items()}
    for key, entries in weighted.items():
        weights = [float(w) for w, _v in entries]
        values = [str(v) for _w, v in entries]
        out[str(key)] = rng.choices(values, weights=weights, k=1)[0]
    return out


def load_seed_style(cfg, name: str) -> StyleDef:
    """A style definition from the seed files, by name — the audition CLI's
    view (the station itself reads styles from the database)."""
    seed_dir = cfg.path("styles.seed_dir")
    for path in sorted(seed_dir.glob("*.toml")) if seed_dir.is_dir() else []:
        try:
            style = load_style_file(path)
        except StyleError as exc:
            raise StyleError(str(exc)) from exc
        if style.name == name:
            return style
    raise StyleError(f"no style seed named {name!r} under {seed_dir} "
                     "(styles created in the admin panel live in the database)")
