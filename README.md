# infiniradio

An infinite radio station: [YuE2](https://huggingface.co/m-a-p/YuE2-3B) music generated
through [ComfyUI](https://github.com/comfyanonymous/ComfyUI), scheduled by a DJ that
learns from listener votes, and broadcast as a live HLS stream everyone hears together.

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
- **Self-optimizing prompts** — the DJ picks style prompts by Thompson sampling over
  each prompt's track votes.
- **Variations** — every track's ABC score is kept; popular songs get re-rendered
  (optionally under a *different style prompt* = new instrumentation) with a fresh seed.
- **Disk quota without waste** — when over quota, generation *pauses*. Unplayed audio is
  never auto-deleted; eviction only touches played tracks (most-downvoted first).
  ABC text always survives.
- **Admin panel** — dashboard, library browser (listen/view ABC/lineage), prompt CRUD,
  job control, runtime settings. Single admin account, scrypt password.
- **Cross-platform** — Linux/macOS/Windows, no named pipes or OS-specific tricks;
  the only external dependency is **ffmpeg**.

## Quickstart

```bash
# Python 3.11+, ffmpeg on PATH
uv venv && uv pip install -e .        # or: python -m pip install -e .
cp config.toml.example config.toml    # set [generator] backend + [comfyui] url
python -m infiniradio passwd          # set the admin password
python -m infiniradio                 # serve on http://127.0.0.1:8000
```

The server refuses to start if no generator backend is configured and the
library is empty (it would only ever broadcast silence). With a backend
configured but ComfyUI unreachable, it starts and surfaces the failures in
the admin dashboard.

## Testing without a GPU

```bash
python -m infiniradio --mockaudio
```

The mock generator is **only** reachable through this flag — never through
config — so a misconfigured deployment can never serve sine tones to
listeners. It synthesizes tone "songs" with fake ABC scores, letting you
exercise the whole pipeline (DJ, votes, eviction, admin) before pointing it
at a real GPU.

## Real generation (ComfyUI + YuE2)

1. Set up ComfyUI with your working YuE2 workflow (the tested graph is in
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

3. On the **ComfyUI machine**, run the cleanup sidecar so finished artifacts don't
   overflow the drive (stdlib only, works on Windows too):

   ```bash
   python sidecar/comfyui_cleanup.py --dir /path/to/ComfyUI/output/radio --token <secret>
   ```

   ```toml
   [comfyui]
   cleanup_url = "http://gpu-box:8765"
   cleanup_token = "<secret>"
   cleanup_every_n = 1
   ```

   The server triggers cleanup before every Nth submission — never on a timer,
   because ABC files appear while their audio is still generating.

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

- The YuE2 weights are **CC BY-NC 4.0**: this project is for non-commercial use only.
  infiniradio itself is MIT-licensed.
- Latency: standard HLS ≈ 15–30 s. A track's airtime is scheduled as
  write-time + `stream.latency_hint` (players trail the live edge by ~3
  segments), which is what keeps "on air now" aligned with what listeners
  actually hear.
- Generation is serial by design (one job at a time); YuE2 uses the whole GPU anyway.
