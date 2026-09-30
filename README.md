# infiniradio

An infinite radio station: Dynamically generated music from
[ComfyUI](https://github.com/comfyanonymous/ComfyUI), scheduled by a DJ that
learns from listener votes, and broadcast as a live HLS stream everyone hears together.

Recommended music generation model: [YuE2](https://huggingface.co/m-a-p/YuE2-3B)

```
listeners ──HLS──> web app <──> DJ ──picks──> library ──PCM──> ffmpeg ──> HLS segments
                     │            │              ▲                (one shared live stream)
                     │            └─ commissions ▼
                     │            generation jobs ──> ComfyUI (remote GPU box) ──> new tracks
                     └── votes steer the DJ: popular songs are replayed (and varied),
                         disliked songs eventually make room for new ones
```

## Features

- **True shared broadcast** — one DJ, one live stream (`ffmpeg` → fMP4 HLS); HLS is plain
  HTTP, so scaling later is just putting nginx/CDN in front of `/live`.
- **Voting** — anonymous listeners up/downvote what they hear; scores use the Wilson
  lower bound so one upvote doesn't make a track a hit.
- **Self-optimizing prompts** — each prompt's selection weight is a configurable
  base plus the average Wilson score of its aired tracks: quality steers generation
  gently, and the per-track average prevents a prolific style from feeding back
  into dominance.
- **LLM-written songs (optional)** — with any OpenAI-compatible endpoint
  configured, multi-stage lyrics workflows (data-driven: a `workflow.toml`
  plus Jinja prompt templates per workflow directory, see
  `lyrics_workflows/README.md`) write real lyrics — random title →
  brainstorm → foreign-language draft → English translation → style prompt →
  rhythm refinement. Stages have gates and feedback retries (too long,
  missing format markers…), the lyrics pipeline overlaps audio rendering
  (two GPUs) or serializes onto one, every LLM exchange is logged for
  debugging, and listener votes arbitrate between workflow variants.
- **Styles (enforced variation)** — unguided pipelines collapse to one
  genre and structure; styles are vote-steered bundles of workflow
  variables (`styles/*.toml`, admin-managed): a general `style_guide` plus
  weighted `structure_guide`/`rhyme_guide` choices resolved per job by
  seeded RNG. The DJ picks styles with the same bandit math as prompts.
- **Tags** — tracks carry tags (static per prompt/workflow plus LLM-picked),
  with per-tag Wilson aggregation in the admin panel for cross-cutting
  "what actually works" analysis.
- **Variations** — every track's ABC score is kept; popular songs get re-rendered
  (optionally under a *different style prompt* = new instrumentation) with a fresh seed.
- **Disk quota without waste** — when over quota, generation *pauses*. Unplayed audio is
  never auto-deleted; eviction only touches played tracks (most-downvoted first).
  ABC text always survives.
- **Admin panel** — dashboard, library browser (listen/view ABC/lineage), prompt CRUD,
  job control, runtime settings, plus pause switches for generation and playback
  (paused playback airs silence so the stream stays live). Single admin account,
  scrypt password.
- **Generated track titles** — curated word lists + template patterns
  (`Cathedral of Glass`, `Drift Endlessly`); seedable, fully offline, and
  auditionable with `python -m infiniradio.naming`.
- **Listener telemetry** — unique listeners per minute and per aired track
  (web-player cookie, or an IP+UA hash for cookie-less clients like VLC),
  collected from HLS poll traffic; only counts are stored, never ids.
- **Cross-platform** — Linux/macOS/Windows, no named pipes or OS-specific tricks;
  the only external dependency is **ffmpeg**.

## Quickstart

```bash
# Python 3.11+, ffmpeg on PATH
python -m venv .venv
source .venv/bin/activate             # Or .venv\Scripts\activate.bat on Windows
python -m pip install -e .
cp config.toml.example config.toml    # set [generator] backend + [comfyui] url
python -m infiniradio passwd          # set the admin password
python -m infiniradio                 # serve on http://127.0.0.1:8000
```

## Testing without a GPU

```bash
python -m infiniradio --mockaudio          # sine-tone audio
python -m infiniradio --mockaudio --mockllm   # + scripted LLM: full lyrics pipeline
```

The mock generator is synthesizes placeholder tone "songs" so you can
exercise the whole pipeline (DJ, votes, eviction, admin) before pointing it
at a real GPU. The mock LLM scripts the reference workflows' stages —
including their gates — so commissioning, the two-phase job flow, stage
logging and tagging all run end to end with no endpoint at all.

## LLM lyrics workflows (optional)

Point `[llm]` at any OpenAI-compatible endpoint (OpenRouter, DeepSeek,
GLM, Kimi, koboldcpp, vLLM, llama.cpp, Ollama — they all speak
`/v1/chat/completions`) and the DJ starts commissioning lyrics jobs
alongside instrumental prompts, weighted by the same vote bandit:

```toml
[llm]
enabled = true
base_url = "http://gpu-box:5000/v1"
model = "glm-4.5-air"        # must match the endpoint's model name
```

Workflows live in `lyrics_workflows/<name>/` — a `workflow.toml` plus one
Jinja template per stage prompt; two ship as examples (`jp-title-first`,
its Chinese-drafting twin `zh-title-first`). **Enable/disable is live from
the admin panel** (prompts page) — no file edits or restarts on the
server; `enabled = false` in the TOML is the boot-time default and can
only be changed in git. The `dj.lyrics_share` runtime setting controls
what fraction of fresh commissions use a lyrics workflow; within that
pool, listener votes arbitrate via the weighted bandit. Audition prompts and runs
without a server:

```bash
python -m infiniradio.lyrics --list
python -m infiniradio.lyrics --dry-run jp-title-first
python -m infiniradio.lyrics --run jp-title-first --mock
python -m infiniradio.llm --status      # probe the configured endpoint
```

Details (stage gates, retry feedback, resume, tags, the job phase
pipeline) are in **[lyrics_workflows/README.md](lyrics_workflows/README.md)**.
If text and music share one GPU, set `lyrics.serialize_with_render = true`
to keep the phases from overlapping.

## Real generation (ComfyUI + YuE2)

1. Set up ComfyUI with a working YuE2 workflow (an example graph is in
   `workflow/infiniradio.json`; node IDs it patches are configured under
   `[comfyui.workflow_vars]`).
2. In `config.toml`:

   ```toml
   [generator]
   backend = "comfyui"

   [comfyui]
   url = "http://gpu-box:8188"
   audio_format = "flac"        # or "mp3" — patched into the workflow at submit
   ```

3. (Optional) On the **ComfyUI machine**, run the cleanup sidecar so finished
   artifacts don't overflow the drive:

   ```bash
   python sidecar/comfyui_cleanup.py --dir /path/to/ComfyUI/output/radio --token <secret>
   ```

   ```toml
   [comfyui]
   cleanup_url = "http://gpu-box:8765"
   cleanup_token = "<secret>"
   cleanup_every_n = 1
   ```

   The server triggers cleanup before every Nth submission. To run it as a
   systemd service on the ComfyUI machine, see DEPLOY.md §5.

The server downloads each job's audio + ABC via the ComfyUI API (`/prompt`, `/history`,
`/view`) into its own `data/music/<track_id>/` — no shared filesystem required.

### How variations work

The workflow contains an "ABC planning" boolean and an "empty ABC" text node wired
through a switch. For a variation we flip the switch and inject the parent's saved ABC:
the ABC-generation pass is skipped and the score is re-rendered with the chosen style
prompt and a new seed.

### The ABC sanity gate

Fresh songs are generated in two passes. First a text-only planning
pass produces an ABC score. While its length is outside configurable
bounds we retry with a different seed. Otherwise we render audio
using this ABC.

Use `python -m infiniradio.abc_audit --library --vs-votes` to check
the configuration against generated songs. Thresholds live under
`[abc_audit]` in the config.

## Layout

```
config.toml          settings (copy from config.toml.example)
workflow/            ComfyUI workflow (API format) used as submit template
prompts/             seed prompt library (TOML), imported on first boot
styles/              seed song styles (TOML), imported on first boot
lyrics_workflows/    LLM lyrics pipelines (workflow.toml + Jinja templates)
sidecar/             cleanup service for the ComfyUI machine
data/                runtime: SQLite DB, music/<id>/{original, playout.m4a, score.abc}, hls/
```

## Deployment

See **[DEPLOY.md](DEPLOY.md)** for the full guide: fresh install on a Linux
box, running as a systemd service, migrating an existing station to new
hardware, LAN access, and exposing the station to the internet through a
Cloudflare Tunnel.

Env overrides: `INFINIRADIO__SECTION__KEY=value` (e.g. `INFINIRADIO__COMFYUI__URL`).
Config path: `--config X` or `INFINIRADIO_CONFIG`.

## Notes

- The YuE2 weights are **CC BY-NC 4.0**. If you use YuE2 to generate music, the
  generated music is for non-commercial use only. infiniradio itself is MIT-licensed.
- Latency: standard HLS ≈ 15–30 s. A track's airtime is scheduled as
  write-time + `stream.latency_hint` (players trail the live edge by ~3
  segments), which is what keeps "on air now" aligned with what listeners
  actually hear.

