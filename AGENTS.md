# AGENTS.md — working rules for AI agents in this repository

## Commits

- **Never commit or push until the user has tested the changes on their
  local server and explicitly approved them.** Leave work uncommitted in
  the working tree. The only exception is an explicit user request to
  commit.

## Testing

- The user runs the station locally and tests changes there. Do **not**
  spawn infiniradio server instances to verify changes yourself.
- In-process regression suites live in `tests/` — run them all with
  `python tests/run.py` (or one: `python tests/test_styles.py`). They use
  temp data dirs and the mock generator/LLM; never the real `data/`.
- Prefer, in order: `python tests/run.py`, compile/import checks, and —
  when diagnosing a live issue — read-only HTTP probes against the user's
  running instance.
- Never leave background processes behind.

## Live data

- `data/` belongs to the user's running station. Read-only diagnostics
  (sqlite `mode=ro`) are fine; never write to it.
