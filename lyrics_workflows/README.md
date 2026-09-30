# Lyrics workflows

One directory per workflow: a `workflow.toml` control-flow file plus one
Jinja template per LLM prompt. Everything content-related lives here; the
engine (`src/infiniradio/lyrics.py`) is generic. Workflows are loaded and
validated once at server startup — a broken workflow is a startup error,
never a silent skip — and there is no hot reload (edit + restart, like the
`names/` word lists).

```
lyrics_workflows/
  jp-title-first/
    workflow.toml        control flow: stages, gates, retries, outputs
    brainstorm.j2.md     one template per stage prompt
    foreign_lyrics.j2.md
    translate.j2.md
    style.j2.md
    refine.j2.md
    tags.j2.md
    feedback.j2.md       retry-feedback template (branches on failures)
```

## workflow.toml

```toml
name = "jp-title-first"          # MUST equal the directory name
description = "…"                # shown in admin
enabled = true                   # disabled workflows are loaded but not commissioned
seed_title = true                # generate a names/-list title as {{ title }}
tags = ["vocal", "english"]      # static tags on every output track

[params]                         # available to every template as {{ params.* }}
language = "Japanese"
script_note = "using kanji, hiragana and katakana as appropriate"

[output]                         # which variables become the GenRequest
title = "song_name"              # optional; falls back to the seed title
style = "style_prompt"           # required
lyrics = "final_lyrics"          # required
tags = "track_tags"              # optional: comma-separated tag list

[[stage]]
id = "refine"                    # identifier (used in logs and the admin UI)
template = "refine.j2.md"        # file name in this directory
block = "final_lyrics"           # SONG-BLOCK extraction (see below): the reply's
                                 # fenced song, or the bare reply, with commentary
                                 # and a leading [Title] section removed
saves = "refined_reply"          # variable name for the full raw reply (kept
                                 # verbatim, independent of block)
thread = "new"                   # "new" conversation | "continue" the previous
                                 # stage's conversation
max_attempts = 3                 # 1 attempt + N-1 retries with feedback
feedback_template = "feedback.j2.md"  # optional; without it the same prompt
                                 # is re-sent (nondeterminism retry)
model = "other-model"            # optional per-stage [llm] overrides
temperature = 0.8                #   (A/B test models within one workflow)

[stage.gate]
max_chars = 3000                 # too_long
min_chars = 20                   # too_short
all_of = ['X:']                  # every pattern must match (AND)  -> missing_pattern
any_of = ['\[Verse\]', '\[Chorus\]']   # at least one (OR) -> missing_any
forbidden = ['TODO']             # none may match -> forbidden_present

[stage.extract]                  # named regexes; a string or an ordered list
song_name = [                    # of alternatives — first match wins
  '(?m)^Name[ \t]*:[ \t]*`([^\n]*)`',   # (lists absorb output-style
  '(?m)^Name[ \t]*:[ \t]*"([^\n]*)"',   # variation without retries)
  '(?m)^Name[ \t]*:[ \t]*([^\n]+)',
]
style_prompt = '(?s)Style prompt:\s*```\s*\n(.*?)\n```'   # single pattern
```

All regexes compile with `re.DOTALL` (`.` matches newlines — fenced blocks
span lines). TOML single-quoted strings keep backslashes literal.

## Song-block extraction (`block`)

Quantized models — especially after a long "thinking" phase — vary their
output style: bare song, fenced song, song wrapped in commentary, or any
combination. Stages that produce *the song* declare `block = "var"` instead
of (or alongside) `saves`, and the engine extracts locally instead of
burning LLM round-trips on retries:

1. the first fenced block whose content contains a section header is the
   song (header patterns default to the stage's own gate `any_of`/`all_of`,
   else the generic `[Header]`-on-its-own-line shape);
2. with no matching fence, the reply is a bare song: a `[Title]` section is
   stripped and anything before the first section header (leading
   commentary prose) is dropped;
3. a leading `[Title]` section is always removed — the song's name travels
   via `[output] title`, never inside the lyrics.

When `block` is set, the gate measures the extracted block (not the raw
reply), so `max_chars` bounds the actual lyrics. The raw reply is preserved
verbatim in the job's stage log.

## Stage execution

Each stage runs: render prompt → LLM call → **all gate checks and
extractions evaluated together** → pass means variables are saved, fail
means every problem is reported at once and the retry composes one fix.
Extraction keys accept an ordered list of alternative patterns (first
match wins) — e.g. a `Name:` line in backticks, quotes, or bare — so
output-style variation parses locally instead of costing LLM round-trips.
Retries continue the *same conversation*: the failed reply stays in the
history and the rendered `feedback_template` follows as the next user turn.
A stage that exhausts `max_attempts` fails the job; the DJ retries it
later, and a retried job **resumes from its last completed stage** (saved
variables and conversation threads persist in the job row).

## Template context

Prompt templates see:

- `{{ title }}` — the seed title (when `seed_title = true`)
- every variable saved/extracted by earlier stages (`{{ brainstorm }}`, …)
- `{{ params.language }}` — the workflow's `[params]` table
- `{{ tags }}` — the station's tag vocabulary (for tagging stages)

Feedback templates additionally see `failures` (list of
`{code, check, detail}`), the predicates `failed("too_long")` and
`extract_failed("song_name")`, `attempt`, and `previous` (the rejected
reply). Failure codes: `too_long too_short missing_pattern missing_any
forbidden_present extract_failed llm_error`.

Reserved names (unavailable as save/extract variables): `title`, `params`,
`tags`, `failures`, `failed`, `extract_failed`, `attempt`, `previous`.

## Tags

Tags describe the **output track** — the final lyrics' language, the
vocals, the genre — not the process. A workflow drafting in Japanese but
producing English lyrics tags `english`; the drafting language travels as
`tracks.workflow_id`.

The vocabulary lives in **`tags.toml`** (seeded into the DB at every
startup; the file curates the list). Two groups:

- `llm_tags` — genres and vocal types, offered to a workflow's tagging
  stage. Keep entries mutually exclusive: overlaps (`vocal` vs
  `male vocal`) are what make the LLM dither.
- `static_tags` — vocabulary-only, never offered to the LLM. Workflow- and
  prompt-forced tags (`vocal`, `english`, `instrumental`) belong here.

The tagging stage asks for the most applicable tags *without a budget*
(budgets cause overthinking) and orders them most-to-least relevant. The
engine then sanitizes locally, spending no LLM round-trips: picks are
matched against the vocabulary — a thinking-leak reply full of prose
yields a clean (or empty) pick — and truncated to `[output] max_tags`,
most relevant first. Static workflow tags don't count against the limit
and can't be picked away.

## Auditioning (no server)

```
python -m infiniradio.lyrics --list
python -m infiniradio.lyrics --validate
python -m infiniradio.lyrics --dry-run jp-title-first     # rendered prompts
python -m infiniradio.lyrics --run jp-title-first --mock  # full pipeline, mock LLM
python -m infiniradio.lyrics --run jp-title-first --stage translate --mock
python -m infiniradio.lyrics --run jp-title-first         # real [llm] endpoint
```

Full-pipeline local testing (sine-tone audio + scripted LLM):

```
python -m infiniradio --mockaudio --mockllm
```

## Styles (enforced variation)

Unguided pipelines collapse to one genre and one song structure — the
model's prior wins. Styles push back: a style is a **named bundle of
workflow variables** the DJ picks per job (weighted by listener votes, the
same bandit math as prompts), defined in `styles/*.toml` and managed in
the admin panel after first boot (the prompts lifecycle):

```toml
[vars]                          # fixed overrides
style_guide = "A dark industrial techno EDM track. Heavy beat and layered harmonics."

[weighted]                      # one seeded-RNG choice per key, per job
structure_guide = [
  { weight = 3, value = "Repetitive catchy hook. Each verse has four lines. ..." },
  { weight = 1, value = "Sparse and mechanical. Verses of two to four short lines, ..." },
]
rhyme_guide = [
  { weight = 2, value = "Rewrite stanzas to follow an ABABCD rhyming pattern ..." },
]
```

Conventions (enforced by taste, not code — any variable name works):

- `style_guide`: general direction (genre/mood/energy). Several stages
  consume it; it steers rather than dictates the final YuE2 style prompt.
- `structure_guide`: the STRUCTURE of the lyrics — lines per section,
  chorus repetition, line lengths. The main artificial-variation lever.
- `rhyme_guide`: rhyme discipline; styles that favor repetition weight
  stricter patterns higher (the semi-coupling).

A workflow opts in with `styled = true` and declares fallbacks under
`[vars]`; templates read the variables at top level
(`{{ style_guide }}`, guarded by `{% if %}`). The engine layers them:
workflow `[vars]` defaults < style overrides (resolved at enqueue, stored
on the job, reproducible from its seed) < the seed title. Unstyled runs
use the defaults — the old unguided behavior stays reachable.

Audition a style without a server:

```
python -m infiniradio.lyrics --run jp-title-first --style jpop-anime --seed 7
```

(the CLI reads seed files; styles created in the admin panel live in the
database and are not visible to it.)

## Future: sharing between variants

`zh-title-first` is currently a full copy of `jp-title-first` with
different `[params]`. When a third variant appears, the plan is
`base = "title-first"` inheritance: stages/params deep-merge (child wins)
and template lookup falls through to the base directory (the engine
already uses a Jinja `ChoiceLoader`, so this is a merge step over specs,
not a redesign).
