# Art workflows

One directory per workflow: a `workflow.toml` control-flow file plus Jinja
templates for the LLM prompts — the same shape (and the same execution
engine) as `lyrics_workflows/`. The difference is the contract: an art
workflow reads a **track's** metadata and writes an **image prompt**,
which the images backend (`[art.api]`) renders. Loaded and validated once
at server startup — a broken workflow is a startup error, never a silent
skip. Enable/disable and weight overrides are live from the admin panel
(prompts page, art workflows table); everything else needs a restart.

```
art_workflows/
  cover/                 # abstract base: the shared two-stage pipeline
    workflow.toml        # abstract = true; stages, gates, extraction
    concept.j2.md        # one template per stage prompt
    prompt.j2.md
    feedback.j2.md       # retry feedback (branches on failures)
  cover-illustrated/     # children: medium + weight via [params]/weight
  cover-photographic/
  lyric-scene/           # standalone single stage straight from the lyrics
  analysis-abstract/     # single stage: song analysis -> textless abstract art
  analysis-typographic/  # single stage: song analysis -> art with the title
                         # set in expressive typography
```

## workflow.toml

```toml
name = "cover-illustrated"       # MUST equal the directory name
description = "…"                # shown in admin
enabled = true                   # disabled workflows load but are never picked
weight = 2.0                     # relative pick probability (default 1.0;
                                 #   0 = kept but effectively unpickable)
base = "cover"                   # optional inheritance, lyrics semantics
abstract = true                  # bases load and validate but are never picked

[params]                         # available to every template as params.*
medium = "flat-color illustration"

[vars]                           # default template variables (lowercase
art_direction = "…"              # identifiers; style-override semantics)

[requires]                       # track predicates — the FILTER before the
lyrics = true                    # weighted pick: workflows whose requires
                                 # don't match the track cannot be picked
                                 # at any weight. lyrics = true needs
                                 # authored lyrics; false = instrumentals
                                 # only; omit = either. Explicit admin
                                 # regeneration bypasses the filter.

[[stage]]                        # same stage spec as lyrics workflows:
id = "prompt"                    # id, template, saves, thread, max_attempts,
template = "prompt.j2.md"        # feedback_template, model/temperature/
max_attempts = 3                 # max_tokens/reasoning_effort overrides,
[stage.gate]                     # extra, gate (max_chars/min_chars/any_of/…)
max_chars = 900
min_chars = 60
[stage.extract]                  # ordered alternative regexes; first match
image_prompt = [                 # wins (each with exactly 1 group)
  '(?s)```text\n(.*?)\n```',
  '(?s)```\n(.*?)\n```',
  '(?s)image prompt:\s*(.+)',
]

[output]
prompt = "image_prompt"          # REQUIRED: the variable that must hold the
                                 # final image prompt (saved or extracted by
                                 # a stage). The only output key art has.
```

`block =` (song-section extraction) is a lyrics concept and rejected
here; everything else about stages/gates/extraction/inheritance works
exactly as documented in `lyrics_workflows/README.md`.

## Template context

Art templates see the track being decorated:

- `{{ title }}`, `{{ theme_title }}` — track title; the lyrics seed theme
- `{{ style }}` — the music style prompt the track was generated from
- `{{ lyrics }}` — the full lyrics snapshot (empty for instrumentals)
- `{{ track_tags }}` — the track's tags (list; `tags` is reserved)
- `{{ genre }}` — the generating prompt's name when there was one
- `{{ tempo }}` (BPM), `{{ music_key }}`, `{{ duration_s }}`,
  `{{ is_variation }}`
- every variable saved/extracted by earlier stages, `{{ params.* }}`,
  the workflow's `[vars]` defaults

Feedback templates additionally see `failures`, the `failed("code")` /
`extract_failed("name")` predicates, `attempt` and `previous` — the
lyrics conventions.

## Selection

Each job picks one enabled workflow in two steps: **filter** (`[requires]`
— e.g. `lyrics = true` keeps a workflow out of the running for
instrumental tracks no matter its weight), then **weighted random** over
the remainder (file `weight`, retunable live via the admin panel's weight
column; an explicit admin regeneration names its workflow and skips both
steps). `art.new_tracks` auto-queues art for freshly ingested tracks;
`art.backfill` trickles art for old tracks without any (a track whose art
job failed once is left alone — use the track page's regenerate button).
The dashboard's generation pause holds the backfill like all automatic
commissioning — manual regeneration from a track page runs regardless,
like the Generate page. Art renders interleave with music renders on the
GPU-side runner so neither starves.

## Auditioning (no server)

```
python -m infiniradio.art --list
python -m infiniradio.art --validate
python -m infiniradio.art --dry-run cover-illustrated   # rendered prompts
python -m infiniradio.art --run cover-illustrated --mock
python -m infiniradio.art --run cover-illustrated       # real [llm] endpoint
```
