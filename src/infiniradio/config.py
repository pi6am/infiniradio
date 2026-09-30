"""Configuration: built-in defaults <- TOML file <- environment variables.

Runtime-tunable settings (the DJ knobs, quota, ...) live in the database
`settings` table once changed from the admin panel; this module provides the
bootstrap values. `runtime_get()` resolves the effective value.
"""

from __future__ import annotations

import copy
import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Callable

ENV_PREFIX = "INFINIRADIO__"

DEFAULTS: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "secret": "",
    },
    "site": {
        # Canonical absolute URL (e.g. "https://infiniradio.ai") used for
        # Open Graph og:url/og:image on the player page so link previews
        # (Discord, Telegram, ...) resolve absolute URLs regardless of how
        # the crawler arrived. Empty = derive from each request (dev/LAN).
        "base_url": "",
    },
    "admin": {
        "username": "admin",
        "password_hash": "",
    },
    "generator": {
        # comfyui | off  ("" = unspecified; the server refuses to start with
        # no generator and an empty library). The mock generator is
        # testing-only and cannot be selected here — use --mockaudio.
        "backend": "",
    },
    "comfyui": {
        "url": "http://127.0.0.1:8188",
        "workflow": "workflow/infiniradio.json",
        "audio_format": "flac",
        "audio_quality": "V0",
        "username": "",
        "password": "",
        "poll_interval": 2.0,
        "request_timeout": 1800.0,
        "cleanup_url": "",
        "cleanup_token": "",
        "cleanup_every_n": 1,
        "workflow_vars": {
            "style": "33:36",
            "lyrics": "33:37",
            "seed": "33:34",
            "plan": "33:31",
            "abc": "33:32",
            "audio_save": "10",
            # SaveText node — the planning pass keeps it and its upstream nodes.
            "abc_save": "43",
        },
    },
    # Two-pass fresh-song generation (see abc_audit.py): plan the ABC,
    # screen it, retry with a new seed on rejection, then render.
    "abc_audit": {
        "enabled": True,
        "max_attempts": 3,
        "max_chars": 5000,
        "min_chars": 300,
    },
    "library": {
        "data_dir": "data",
        "playout_bitrate": "192k",
        "loudness_target": -16.0,
        # Deletion buffer: condemned tracks keep their audio on disk (still
        # counting toward quota, never aired) for this long before the
        # janitor really deletes them — review time against coordinated
        # downvote attacks. 0 = delete immediately (ephemeral mode).
        "deletion_buffer_hours": 24.0,
    },
    "stream": {
        "segment_seconds": 6,
        "playlist_segments": 5,
        "bitrate": "128k",
        "sample_rate": 48000,
        "latency_hint": 20.0,
        "ffmpeg_path": "ffmpeg",
        "ffprobe_path": "ffprobe",
    },
    "dj": {
        "max_queued_jobs": 1,
        "runway_minutes": 5.0,
        "prompt_base_weight": 0.5,
        "repeat_window_tracks": 25,
        "repeat_window_hours": 6.0,
        "new_track_share": 0.35,
        "variation_rate": 0.15,
        "variation_score_threshold": 0.6,
        "protect_threshold": 3,
        "downvote_evict_net": -2,
        "downvote_evict_plays": 2,
        "quota_gb": 50.0,
    },
    # LLM lyrics generation (optional; see src/infiniradio/llm/). With
    # llm.enabled = false no client is constructed and lyrics workflows
    # never run — the station behaves exactly as before.
    "llm": {
        "enabled": False,
        # openai = any OpenAI-compatible /v1/chat/completions endpoint
        # (OpenRouter, DeepSeek, GLM, Kimi, koboldcpp, vLLM, ...). The mock
        # client is testing-only: --mockllm, never config.
        "backend": "openai",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "model": "",
        "temperature": 1.0,
        "max_output_tokens": 8192,
        "timeout_s": 900.0,
        # Transport-level retries (429/5xx/timeouts) with backoff; content
        # quality retries belong to the workflow stages, not the client.
        "transport_retries": 3,
        # When a reply is cut off by the output budget (finish_reason =
        # length — e.g. the model thinking past its entire token budget),
        # resubmit the partial as an assistant turn and ask it to continue,
        # at most this many times. Continuations are transport, not content
        # retries: they never consume a stage's attempts. 0 disables.
        "max_continuations": 2,
    },
    "lyrics": {
        # One directory per workflow: workflow.toml + Jinja templates.
        # Loaded and validated once at startup (no hot reload), like names/.
        "workflows_dir": "lyrics_workflows",
        # true = the text and music backends share one GPU: never run a
        # lyrics job while a render is in progress (serializes the phases).
        # false (default) = separate machines/GPUs: lyrics for the next job
        # generate while the current job renders.
        "serialize_with_render": False,
        # Tags picked by a workflow's tagging stage that are not already in
        # the vocabulary: true = add them, false = drop them. (Picks are
        # sanitized against the vocabulary before this ever applies.)
        "allow_new_tags": True,
    },
    "tags": {
        # Tag vocabulary seed (see tags.toml): llm_tags are offered to the
        # workflows' tagging stage, static_tags are vocabulary-only. Seeded
        # at every startup — edit the file to curate the list.
        "seed_file": "tags.toml",
    },
    "styles": {
        # Style seeds (see styles/): named variable-override bundles the DJ
        # picks with listener votes (style_guide direction + weighted
        # structure/rhyme choices). Imported on first boot; afterwards the
        # database is the source of truth (admin panel CRUD) — the prompts
        # lifecycle, not the tags one.
        "seed_dir": "styles",
    },
    "prompts": {
        "seed_dir": "prompts",
    },
    "naming": {
        # Word-list directory for generated track titles (one file per
        # slot: nouns.txt, adjectives.txt, verbs.txt, verbs_third.txt,
        # adverbs.txt). Loaded once at startup.
        "words_dir": "names",
    },
}

# key -> (default, cast, human label, help) — editable from the admin panel.
RUNTIME_SETTINGS: dict[str, tuple[Any, Callable[[str], Any], str, str]] = {
    "dj.prompt_base_weight": (0.5, float, "Prompt selection base weight",
                              "Every prompt's selection weight is this plus the average Wilson score of "
                              "its aired tracks — gentle quality steering with a diversity floor (higher "
                              "= more uniform selection)"),
    "dj.style_base_weight": (0.5, float, "Style selection base weight",
                             "Same rule for song styles (lyrics workflows): weight = this plus the "
                             "average Wilson score of the style's aired tracks"),
    "dj.lyrics_share": (0.5, float, "Lyrics workflow share",
                        "Fraction of fresh commissions that use a lyrics workflow when one "
                        "is enabled; the rest use instrumental prompts. Votes still steer "
                        "the pick within each pool (0 = instrumental only, 1 = lyrics only)"),
    "dj.max_queued_jobs": (1, int, "Max queued generation jobs",
                           "Commission at most this many jobs ahead of the one running; keeps decisions "
                           "just-in-time (latest prompts and votes)"),
    "dj.runway_minutes": (5.0, float, "Generation runway (minutes)",
                          "Only generate when less than this much eligible material (playable without "
                          "violating the repeat windows) remains; the library grows until the windows "
                          "are comfortably satisfied"),
    "dj.repeat_window_tracks": (25, int, "Repeat window (tracks)", "Do not play a track again within the last N aired tracks"),
    "dj.repeat_window_hours": (6.0, float, "Repeat window (hours)", "Do not play a track again within N hours"),
    "dj.new_track_share": (0.35, float, "New track share", "Fraction of picks reserved for never-played tracks (exploration)"),
    "dj.variation_rate": (0.15, float, "Variation rate", "Chance that a commissioned job is a variation of a popular track"),
    "dj.variation_score_threshold": (0.6, float, "Variation score threshold", "Wilson score a track needs before variations are commissioned"),
    "dj.protect_threshold": (3, int, "Auto-protect threshold", "Net upvotes at which a track becomes protected from eviction"),
    "dj.downvote_evict_net": (-2, int, "Downvote eviction votes", "Net votes at or below this makes a track eligible for early eviction"),
    "dj.downvote_evict_plays": (2, int, "Downvote eviction plays", "... after this many plays (give it a chance first)"),
    "library.quota_gb": (50.0, float, "Disk quota (GiB)", "Generation pauses when music files exceed this; nothing unplayed is ever deleted"),
    "library.deletion_buffer_hours": (24.0, float, "Deletion buffer (hours)",
                                      "Condemned tracks keep their audio (counting toward quota, never aired) "
                                      "for this long before real deletion — review time against downvote waves. "
                                      "Admin can restore while trashed. 0 = delete immediately"),
}


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


class Config:
    """Read-only view over the merged configuration tree."""

    def __init__(self, data: dict[str, Any], base_dir: Path):
        self._d = data
        self.base_dir = base_dir

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path against the process working directory."""
        p = Path(str(self.get(dotted, ""))).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def load_config(explicit: str | os.PathLike | None = None) -> Config:
    data = copy.deepcopy(DEFAULTS)
    env_path = os.environ.get("INFINIRADIO_CONFIG")
    candidates = [explicit, env_path, "config.toml"]
    loaded_from = None
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_file():
            with open(p, "rb") as fh:
                _deep_merge(data, tomllib.load(fh))
            loaded_from = p
            break
    if loaded_from is None and Path("config.toml").is_file():
        pass  # already covered by candidates

    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        if len(parts) < 2:
            continue
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)

    cfg = Config(data, Path.cwd().resolve())
    if loaded_from:
        logging.getLogger(__name__).debug("config loaded from %s", loaded_from)
    return cfg


def runtime_get(cfg: Config, db, key: str) -> Any:
    """Effective runtime value: DB settings > config file > built-in default."""
    default, cast, _, _ = RUNTIME_SETTINGS[key]
    section, _, option = key.partition(".")
    db_val = db.one("SELECT value FROM settings WHERE key = ?", (key,))
    if db_val is not None:
        try:
            return cast(db_val["value"])
        except (TypeError, ValueError):
            pass
    val = cfg.get(key, None)
    if val is not None and key in RUNTIME_SETTINGS:
        try:
            return cast(val)
        except (TypeError, ValueError):
            pass
    return default
