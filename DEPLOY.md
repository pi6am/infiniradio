# Deploying infiniradio

Setting the station up on a dedicated machine: a fresh install, running it as
a systemd service, migrating an existing station from another machine, and
getting listeners to it (LAN first, then internet via a Cloudflare Tunnel).
Commands are for Debian/Ubuntu; Fedora and Arch equivalents are noted where
they differ.

## The moving parts

```
listeners ──HTTP/HLS──> infiniradio app (Python + ffmpeg, this machine)
                          │
                          └── commissions jobs ──> ComfyUI (GPU machine, stays
                              and downloads audio          where it is)
```

A deployment needs:

| Piece | Requirement |
|---|---|
| OS | Any Linux; Python 3.11+ and ffmpeg are in every current distro |
| CPU/RAM | Modest — the app schedules and transcodes audio, the GPU work is remote |
| Disk | `data/` grows to the music quota (default 50 GiB) |
| Network | One TCP port for HTTP (default 8000); reachability to your ComfyUI |
| State | One directory tree (code + venv + `data/`); nothing lives outside it |

Everything the station knows — library, votes, prompts, telemetry, the admin
password hash and the session secret — lives in `data/infiniradio.db` plus
`data/music/`. Back up those and `config.toml` and you can rebuild anywhere.

---

## 1. Fresh install

### 1.1 System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git rsync
# Fedora: sudo dnf install -y python3 python3-pip ffmpeg git rsync
# Arch:   sudo pacman -Syu --needed python ffmpeg git rsync
#         (pip and venv are part of Arch's python package)
python3 --version    # needs 3.11+
ffmpeg -version
```

### 1.2 Lay down the tree

The convention in this guide is `/opt/infiniradio`, owned by a dedicated
service user. Set the directory up with your normal admin account, then hand
it over at the end.

```bash
sudo mkdir -p /opt/infiniradio
sudo chown "$USER" /opt/infiniradio

# Clone (or scp/rsync) the repository — the whole tree, it is self-contained:
git clone <YOUR_REPO_URL> /opt/infiniradio
```

### 1.3 Python environment

```bash
cd /opt/infiniradio
python3 -m venv .venv
.venv/bin/pip install -e .
```

(`-e` installs the tree in place, so a later `git pull` takes effect on the
next service restart; re-run `pip install -e .` only when dependencies change.)

### 1.4 Configuration

```bash
cp config.toml.example config.toml
${EDITOR:-nano} config.toml
```

Minimum for a real deployment:

```toml
[server]
host = "0.0.0.0"        # reachable on the LAN; see §6 before going public
port = 8000

[generator]
backend = "comfyui"

[comfyui]
url = "http://192.168.x.x:8188"   # your ComfyUI machine
```

Check that ComfyUI answers from this machine:

```bash
curl -sS http://192.168.x.x:8188/system_stats | head -c 200; echo
```

### 1.5 Admin password

```bash
cd /opt/infiniradio && .venv/bin/python -m infiniradio passwd
```

### 1.6 Smoke test in the foreground

```bash
cd /opt/infiniradio && .venv/bin/python -m infiniradio
```

You should see the feeder start, the live playlist appear under
`data/hls/`, and the player at `http://<machine>:8000/`. A fresh library
airs silence until the first ComfyUI jobs finish and get ingested. `Ctrl-C`
when convinced.

> The server refuses to start (exit code 2) if no generator backend is
> configured and the library is empty — that guard keeps you from silently
> airing silence forever. The mock generator is testing-only:
> `python -m infiniradio --mockaudio`.

---

## 2. Hand over to the service account

```bash
sudo useradd --system --home-dir /opt/infiniradio --shell /usr/sbin/nologin infiniradio
sudo chown -R infiniradio:infiniradio /opt/infiniradio
```

## 3. Run as a systemd service

The repo ships a unit file:

```bash
sudo cp deploy/infiniradio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now infiniradio
systemctl status infiniradio
```

Notes:

- `WorkingDirectory` matters: `config.toml`, `names/`, `prompts/`,
  `workflow/` and `data/` are all resolved relative to the process working
  directory, so it must point at the repo root. If you installed elsewhere,
  fix the paths in the unit first.
- Crashes restart automatically after 5 s; a restart resumes the interrupted
  song where listeners left it, so restarts are near-transparent.
- Logs go to the journal:

```bash
journalctl -u infiniradio -f          # follow
journalctl -u infiniradio -n 200      # recent
```

---

## 4. Migrating an existing station from another machine

The transfer is one rsync of the tree. On the new machine, first run
**§1.1 and §1.2** (system packages, and `/opt/infiniradio` owned by your
user) but skip the clone — the rsync brings the tree. Do the transfer with
the old instance stopped (so the database and files are consistent and
nothing writes mid-copy):

```bash
# ON THE OLD MACHINE (stop the server there first):
rsync -a --info=stats2 \
    --exclude .venv/ --exclude data/hls/ --exclude data/infiniradio.lock \
    /path/to/infiniradio/ newhost:/opt/infiniradio/
```

What the excludes mean:

- `.venv/` — virtualenvs embed absolute paths; rebuild on the target
  (§1.3).
- `data/hls/` — live stream segments, regenerated from scratch each time the
  encoder starts; copying them is pointless.
- `data/infiniradio.lock` — the running-instance guard is a kernel lock on
  the file, not its existence; the new machine takes its own.

Everything else travels: the code and git history (`.git/` included), the
database with all votes/telemetry/settings, the admin password hash and
session secret (both live in the DB — no `passwd` step needed), and
`data/music/`.

Then finish the new machine:

```bash
cd /opt/infiniradio
python3 -m venv .venv && .venv/bin/pip install -e .   # as in §1.3
# config.toml came along in the rsync; adjust [server] host if needed
# then §2 (service user + chown) and §3 (unit)
```

On first start the station resumes the airing the shutdown interrupted (a
few seconds ahead — the downtime is skipped), and listeners who keep the
page open re-sync automatically. Don't run both instances afterwards: the
library, votes and schedule diverge quickly.

---

## 5. The ComfyUI cleanup sidecar (optional)

ComfyUI never deletes the files it generates. infiniradio downloads each
finished job's audio + ABC into its own library, then asks this sidecar —
running **on the ComfyUI machine** — to delete the ingested originals.
Without it, the GPU box's drive fills up over time. Nothing runs on a
schedule: cleanup happens only when the radio server asks (files of
in-flight jobs are never touched), and every request must carry the
shared token.

The script is dependency-free (Python 3 standard library only) — the
system Python is enough, no venv needed.

### 5.1 Install on the ComfyUI machine

From a checkout of this repo (the radio machine has one):

```bash
scp sidecar/comfyui_cleanup.py you@gpu-box:/tmp/
```

On the ComfyUI machine:

```bash
sudo mkdir -p /opt/infiniradio-sidecar
sudo mv /tmp/comfyui_cleanup.py /opt/infiniradio-sidecar/
```

### 5.2 Configure

The watched folder and the shared token live in a root-only file — unit
files are world-readable, so secrets don't go in them:

```bash
sudo tee /etc/infiniradio-sidecar.env >/dev/null <<EOF
DIR=/home/comfy/ComfyUI/output/radio
TOKEN=$(openssl rand -hex 32)
PORT=8765
EOF
sudo chmod 600 /etc/infiniradio-sidecar.env
sudo cat /etc/infiniradio-sidecar.env        # you'll need TOKEN in §5.4
```

`DIR` is the ComfyUI output subfolder this station generates into.

### 5.3 Run as a systemd service

```bash
sudo cp deploy/infiniradio-sidecar.service /etc/systemd/system/
```

Edit the installed unit first: `User=`/`Group=` must be the account that
owns the output directory (deletions happen as that user) — usually the
user ComfyUI runs as. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now infiniradio-sidecar
journalctl -u infiniradio-sidecar -n 5      # "watching <dir> … auth: on"
```

Notes:

- The unit binds `0.0.0.0` so the radio machine can reach it (requests
  are token-gated); if ComfyUI and the radio server share a machine,
  switch to `--host 127.0.0.1`.
- The ComfyUI machine's firewall, if any, must allow the port (8765) from
  the radio machine.

### 5.4 Point the radio server at it

In the radio machine's `config.toml`:

```toml
[comfyui]
cleanup_url = "http://gpu-box:8765"
cleanup_token = "<the TOKEN from /etc/infiniradio-sidecar.env>"
cleanup_every_n = 1
```

Restart the radio service, then verify end-to-end from the radio machine:

```bash
curl -s -H "X-Token: <token>" http://gpu-box:8765/status; echo
```

That returns current file counts and disk usage. From the next generation
job on, ingested artifacts on the GPU box are cleaned up behind it.

---

## 6. Listener access

### 6.1 LAN

With `host = "0.0.0.0"` the player is at `http://<machine-LAN-IP>:8000/`
and the stream itself at `/live/live.m3u8` (VLC & friends). If the machine
runs a firewall:

```bash
sudo ufw allow 8000/tcp        # Ubuntu
# Fedora (firewalld): sudo firewall-cmd --add-port=8000/tcp --permanent && sudo firewall-cmd --reload
# Arch ships no firewall by default — only needed if you enabled ufw/firewalld yourself
```

### 6.2 Internet via a Cloudflare Tunnel

A tunnel needs no port forwarding and terminates TLS for you — ideal for a
small invited audience. The app can stay bound to localhost or the LAN;
cloudflared connects to it there.

Install cloudflared (from Cloudflare's package repo):

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

Elsewhere:

```bash
# Fedora: sudo dnf install -y \
#   https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm
# Arch:   sudo -Syu cloudflared, or the static binary:
#   https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
```

Quick throwaway test (ephemeral URL, no account needed):

```bash
cloudflared tunnel --url http://localhost:8000
# prints a https://<random>.trycloudflare.com URL — share it; Ctrl-C drops it
```

A permanent hostname (requires a domain on Cloudflare):

```bash
cloudflared tunnel login                       # authorize your zone
cloudflared tunnel create infiniradio          # writes <tunnel-id>.json
cloudflared tunnel route dns infiniradio radio.example.com

sudo tee /etc/cloudflared/config.yml <<EOF
tunnel: infiniradio
credentials-file: /home/YOU/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: radio.example.com
    service: http://localhost:8000
  - service: http_status:404
EOF

sudo cloudflared service install               # runs it as its own service
sudo systemctl enable --now cloudflared
```

Behavior notes:

- **HLS works through the tunnel unchanged.** Playlists (`.m3u8`) are not
  cached by default, and segment URLs are unique and immutable per encoder
  generation, so nothing stale can be served. If you ever suspect caching,
  add a Cache Rule that bypasses cache for `*/live/live.m3u8`.
- **Listener counting keeps working**: cloudflared connects from localhost
  and forwards the visitor's IP in `X-Forwarded-For`, which uvicorn trusts
  from `127.0.0.1` by default — the app still sees distinct listener IPs.
- **What the app already hardens**: security headers on every response,
  `Secure` cookie flags when the request arrived over HTTPS, login
  throttling (5 attempts, then a 5-minute lockout per IP), per-listener and
  per-IP vote rate limits, and a request-body size cap. On top of that:
  use a strong admin password, and treat the tunnel as the only way in —
  no port forwards to the origin.
- **The admin panel goes wherever the app goes.** Use a strong admin
  password; if you want it fenced off, put Cloudflare Access (Zero Trust) in
  front of `radio.example.com/admin`.
- Plain reverse proxy (nginx/caddy) + port forwarding works too; for larger
  audiences serve `/live` statically with caching and keep the rest on the
  app.

---

## 7. Operating

**Logs** — `journalctl -u infiniradio -f`.

**Updates** — deploy new code, then restart (database migrations run at
boot):

```bash
sudo -u infiniradio git pull                     # or re-run the §4 rsync
sudo -u infiniradio /opt/infiniradio/.venv/bin/pip install -e .
sudo systemctl restart infiniradio
```

**Deploying from a dev machine** — rsyncing as your own ssh user hits the
§2 ownership: the tree belongs to the service account. Either grant your
user group write access once (the setgid directories make new files
inherit it, so this keeps working):

```bash
sudo usermod -aG infiniradio <you>                # then log out and back in
sudo chmod -R g+rwX /opt/infiniradio
sudo find /opt/infiniradio -type d -exec chmod g+s {} +
```

...and plain rsync as your user works (`--no-g` matters: preserving the
dev machine's group tries to chgrp service-owned files and fails):

```bash
rsync -a --no-g --no-o --exclude .venv/ --exclude data/ --exclude config.toml \
    --exclude __pycache__/ ./ you@server:/opt/infiniradio/
```

or skip permissions entirely by staging through /tmp with sudo on the
receiving side:

```bash
rsync -a --no-g --no-o --exclude .venv/ --exclude data/ --exclude config.toml \
    --exclude __pycache__/ ./ you@server:/tmp/infiniradio-stage/
ssh -t you@server 'sudo rsync -a --no-g --delete --exclude .venv/ --exclude data/ \
    --exclude config.toml --exclude __pycache__/ /tmp/infiniradio-stage/ \
    /opt/infiniradio/ && sudo systemctl restart infiniradio'
```

In both cases exclude the production-only state (`data/`, `config.toml`)
from the sync — and remember a restart is all code updates need (the
install is editable; rerun `pip install -e .` only when `pyproject.toml`
changed).

**Backups** — everything that matters is `config.toml` + `data/`. The
database can be snapshotted consistently while the server runs (needs the
sqlite3 CLI: `apt install sqlite3` / `dnf install sqlite` / `pacman -S sqlite`):

```bash
sudo -u infiniradio sqlite3 /opt/infiniradio/data/infiniradio.db \
  ".backup '/var/backups/infiniradio-$(date +%F).db'"
tar -C /opt/infiniradio -czf /var/backups/infiniradio-music-$(date +%F).tgz \
  data/music config.toml
```

**Restore** — stop the service, put the tree back, start. A restore is just
another migration (§4).

**Moving `data/` elsewhere** (bigger disk): point `library.data_dir` at an
absolute path in `config.toml` — but keep the process working directory at
the repo root; `names/`, `prompts/`, `workflow/` and `config.toml` itself
are resolved from there.

---

## 8. Troubleshooting

| Symptom | Meaning |
|---|---|
| Refuses to start: *another infiniradio instance is already using data/* | One is truly running — the guard is a kernel lock, it cannot go stale after a crash. Check `systemctl status infiniradio` / `ps aux | grep infiniradio`. |
| Refuses to start (exit 2), empty library | No generator backend configured — set `[generator] backend = "comfyui"` (or test with `--mockaudio`). |
| Config seemingly ignored / data dir in the wrong place | The process isn't running with the repo root as working directory (§3) — or point `--config` / `INFINIRADIO_CONFIG` at the file explicitly. |
| Stream silent, player fine | Check the admin dashboard: generation paused (disk quota?), playback paused, or ComfyUI unreachable (`curl $COMFYUI_URL/system_stats`). |
| Generation works but the GPU box's drive keeps filling | Sidecar down, unreachable, or token mismatch — check `journalctl -u infiniradio-sidecar` on the ComfyUI machine and `[comfyui] cleanup_url`/`cleanup_token` on the radio machine (§5). |
