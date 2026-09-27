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
python -m infiniradio --mockaudio
```

The mock generator is synthesizes placeholder tone "songs" so you can
exercise the whole pipeline (DJ, votes, eviction, admin) before pointing it
at a real GPU.

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

   The server triggers cleanup before every Nth submission.

The server downloads each job's audio + ABC via the ComfyUI API (`/prompt`, `/history`,
`/view`) into its own `data/music/<track_id>/` — no shared filesystem required.

### How variations work

The workflow contains an "ABC planning" boolean and an "empty ABC" text node wired
through a switch. For a variation we flip the switch and inject the parent's saved ABC:
the ABC-generation pass is skipped and the score is re-rendered with the chosen style
prompt and a new seed.

## Layout

```
config.toml          settings (copy from config.toml.example)
workflow/            ComfyUI workflow (API format) used as submit template
prompts/             seed prompt library (TOML), imported on first boot
sidecar/             cleanup service for the ComfyUI machine
data/                runtime: SQLite DB, music/<id>/{original, playout.m4a, score.abc}, hls/
```

## Deployment

- **Linux**: systemd unit running `/path/to/venv/bin/python -m infiniradio`
- **Windows**: NSSM service or Task Scheduler entry
- **macOS**: launchd agent
- Put a reverse proxy (nginx/caddy) in front for TLS. Serve `/live` statically with
  caching to scale to many listeners; everything else stays on the app.

Env overrides: `INFINIRADIO__SECTION__KEY=value` (e.g. `INFINIRADIO__COMFYUI__URL`).
Config path: `--config X` or `INFINIRADIO_CONFIG`.

## Notes

- The YuE2 weights are **CC BY-NC 4.0**. If you use YuE2 to generate music, the
  generated music is for non-commercial use only. infiniradio itself is MIT-licensed.
- Latency: standard HLS ≈ 15–30 s. A track's airtime is scheduled as
  write-time + `stream.latency_hint` (players trail the live edge by ~3
  segments), which is what keeps "on air now" aligned with what listeners
  actually hear.

