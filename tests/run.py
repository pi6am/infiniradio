"""Run the in-process regression suites.

    python tests/run.py              # everything
    python tests/run.py test_styles  # one suite (or several)

Each suite is a plain script (see tests/_helpers.py); failures print a
traceback and the exit code is nonzero.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _helpers

_helpers.ensure_repo()

MODULES = [
    "test_lyrics_engine",   # workflow engine: retries, resume, parsing, truncation
    "test_songwriter",      # single-stage fine-tuned model: fresh retries, derived tags
    "test_jobs",            # two-phase job flow: lyrics -> render -> ingest
    "test_koboldcpp",       # KoboldCpp native music API: prepare, render, audit
    "test_styles",          # styles: import, resolution, commissioning, share
    "test_workflow_inheritance",  # base chains, stage merge, abstract bases
    "test_deletion",        # deletion buffer + hold
    "test_llm_status",      # bounded + cached LLM health ping
    "test_app",             # app factory: startup validation, routes
    "test_lyrics_public",   # player lyrics: header stripping, payload placement
    "test_lyrics_banned",   # banned phrases: gate, analysis, search, bulk ops
    "test_encoder",         # HLS generations: epoch naming, flush playlist bridge
    "test_feeder",          # in-place skip: same-timeline cut, played_s clamp
    "test_library_page",    # library tab: column sorting, paging, URL state
    "test_pick_next",       # DJ selection: novelty/uncertainty weights, relaxation
]


def main(argv: list[str]) -> int:
    names = argv or MODULES
    unknown = [n for n in names if n not in MODULES and (n + ".py") not in
               {m + ".py" for m in MODULES}]
    failures: list[str] = []
    for name in names:
        try:
            mod = __import__(name)
        except ImportError as exc:
            print(f"FAIL {name}: cannot import ({exc})")
            failures.append(name)
            continue
        t0 = time.monotonic()
        try:
            mod.run()
            print(f"PASS {name} ({time.monotonic() - t0:.1f}s)")
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failures.append(name)
    total = len(names)
    print(f"\n{total - len(failures)}/{total} suite(s) passed"
          + ("" if not failures else f"; failed: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
