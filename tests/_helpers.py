"""Shared helpers for the in-process test suites.

House style: plain scripts, no test framework. Each test module exposes
``run()`` (assertions raise); ``tests/run.py`` executes them all, or run a
single module directly: ``python tests/test_styles.py``.

Every suite works against its own temp data dir — the repo's real data/
and config endpoints are never touched (mock generator + scripted mock LLM;
no server processes, no ports).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo() -> None:
    """Make the package importable and relative config paths resolve, no
    matter where the tests are launched from: src/ on sys.path and the
    repo root as cwd."""
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(REPO_ROOT)


def isolated_config():
    """load_config() with library.data_dir pointed at a fresh temp dir.

    Returns (cfg, tmp_path); the caller removes tmp_path when done. The
    repo's config.toml supplies everything else (workflow dir, tags.toml,
    styles/, names/) — read-only inputs the tests rely on.
    """
    from infiniradio.config import load_config

    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="infiniradio-test-"))
    object.__setattr__(cfg, "_d", {**cfg._d,
                                   "library": {**cfg._d["library"],
                                               "data_dir": str(tmp)}})
    return cfg, tmp


class RT:
    """Minimal Runtime stand-in — the DJ runners only use these attrs."""

    def __init__(self, cfg, db, generator, llm, workflows, gen_lock=None):
        self.cfg, self.db, self.generator, self.llm = cfg, db, generator, llm
        self.workflows, self.gen_lock = workflows, gen_lock
