"""Music library: track lifecycle, ingest pipeline, votes, prompt import.

Layout under data_dir:
  music/<track_id>/original.<ext>   as generated (flac or mp3)
  music/<track_id>/playout.m4a     loudness-normalized AAC used on air
  music/<track_id>/score.abc        ABC notation (also stored in the DB row)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tomllib
from pathlib import Path

from .config import Config, runtime_get
from .db import DB
from .generator.base import GenResult
from .util import human_bytes, iso, utcnow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- ffmpeg helpers

async def ffprobe(cfg: Config, path: Path) -> dict:
    cmd = [
        cfg.get("stream.ffprobe_path", "ffprobe"),
        "-v", "error", "-print_format", "json", "-show_format", str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {err.decode(errors='replace').strip()}")
    return json.loads(out.decode())


async def transcode_playout(cfg: Config, src: Path, dst: Path) -> None:
    loud = float(cfg.get("library.loudness_target", -16.0))
    bitrate = str(cfg.get("library.playout_bitrate", "192k"))
    cmd = [
        cfg.get("stream.ffmpeg_path", "ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(src),
        "-af", f"loudnorm=I={loud}:TP=-1.5:LRA=11",
        "-ar", str(int(cfg.get("stream.sample_rate", 48000))), "-ac", "2",
        "-c:a", "aac", "-b:a", bitrate, str(dst),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"playout transcode failed: {err.decode(errors='replace').strip()}")


# ------------------------------------------------------------------ ABC parsing

_ABC_KEY = re.compile(r"^K:\s*(\S+)", re.MULTILINE)
_ABC_TEMPO = re.compile(r"^Q:\s*1/4\s*=\s*(\d+)", re.MULTILINE)


def parse_abc_header(text: str) -> tuple[str, int]:
    key = tempo = ""
    m = _ABC_KEY.search(text)
    if m:
        key = m.group(1)
    m = _ABC_TEMPO.search(text)
    if m:
        tempo = m.group(1)
    return key, int(tempo) if tempo else 0


# ---------------------------------------------------------------- track lifecycle

def music_dir(cfg: Config) -> Path:
    return cfg.path("library.data_dir") / "music"


def track_dir(cfg: Config, track_id: int) -> Path:
    return music_dir(cfg) / str(track_id)


def create_track(db: DB, cfg: Config, *, prompt_id: int | None, parent_track_id: int | None,
                 style_text: str, lyrics_snapshot: str, seed: int, cot_mode: str) -> int:
    track_id = db.execute(
        "INSERT INTO tracks (title, parent_track_id, prompt_id, style_text, lyrics_snapshot,"
        " seed, cot_mode, status, created_at)"
        " VALUES ('', ?, ?, ?, ?, ?, ?, 'generating', ?)",
        (parent_track_id, prompt_id, style_text, lyrics_snapshot, seed, cot_mode, iso(utcnow())),
    )
    track_dir(cfg, track_id).mkdir(parents=True, exist_ok=True)
    return track_id



async def ingest(db: DB, cfg: Config, track_id: int, result: GenResult) -> dict:
    """Complete a generating track: probe, transcode playout, finalize DB row."""
    tdir = track_dir(cfg, track_id)
    original = result.original_path
    if not original.is_absolute():
        original = tdir / original.name
    playout = tdir / "playout.m4a"
    await transcode_playout(cfg, original, playout)

    probe = await ffprobe(cfg, original)
    duration = float(probe.get("format", {}).get("duration", 0) or 0)
    size = original.stat().st_size + playout.stat().st_size

    row = db.one("SELECT * FROM tracks WHERE id = ?", (track_id,)) or {}
    prompt = db.one("SELECT * FROM prompts WHERE id = ?", (row.get("prompt_id"),)) \
        if row.get("prompt_id") else None
    parent = db.one("SELECT * FROM tracks WHERE id = ?", (row.get("parent_track_id"),)) \
        if row.get("parent_track_id") else None
    if parent:
        style_name = prompt["name"] if prompt else "custom"
        title = f"{parent['title'] or 'untitled'} \u00b7 {style_name}"
    else:
        title = f"{prompt['name']} #{track_id:04d}" if prompt else f"track #{track_id:04d}"

    key, tempo = parse_abc_header(result.abc_text or "")
    db.execute(
        "UPDATE tracks SET title = ?, duration_s = ?, size_bytes = ?, music_key = ?, tempo = ?,"
        " original_path = ?, playout_path = ?, abc_text = ?, status = 'ready'"
        " WHERE id = ?",
        (title, duration, size, key, tempo,
         f"{track_id}/{original.name}", f"{track_id}/playout.m4a",
         result.abc_text or "", track_id),
    )
    log.info("ingested track %s (%s, %.0fs, %s) %s", track_id, title, duration,
             human_bytes(size), "variation" if parent else "")
    return db.one("SELECT * FROM tracks WHERE id = ?", (track_id,)) or {}


def delete_track(db: DB, cfg: Config, track_id: int) -> bool:
    """Remove a track's files but keep the row (status=deleted) with ABC + stats.

    Never call this for unplayed tracks unless the user asked for it
    explicitly (admin action): the janitor only evicts played tracks.
    """
    row = db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    if not row or row["status"] == "deleted":
        return False
    tdir = track_dir(cfg, track_id)
    if tdir.is_dir():
        shutil.rmtree(tdir, ignore_errors=True)
    db.execute(
        "UPDATE tracks SET status = 'deleted', original_path = '', playout_path = '',"
        " size_bytes = 0 WHERE id = ?", (track_id,)
    )
    log.info("deleted track %s (%s)", track_id, row["title"])
    return True


def fail_track(db: DB, track_id: int) -> None:
    """Mark a still-generating track failed.

    Guarded on status so a stale cancellation cannot clobber a track that
    concurrently finished ingesting (or was already deleted).
    """
    db.execute("UPDATE tracks SET status = 'failed'"
               " WHERE id = ? AND status = 'generating'", (track_id,))


# ---------------------------------------------------------------- votes & plays

def apply_vote(db: DB, cfg: Config, track_id: int, listener_id: str, value: int) -> dict:
    if value not in (-1, 0, 1):
        raise ValueError("vote value must be -1, 0 or +1")
    now = iso(utcnow())
    db.execute(
        "INSERT INTO votes (track_id, listener_id, value, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(track_id, listener_id) DO UPDATE SET value = excluded.value,"
        " updated_at = excluded.updated_at",
        (track_id, listener_id, value, now, now),
    )
    counts = db.one(
        "SELECT COALESCE(SUM(value = 1), 0) AS up, COALESCE(SUM(value = -1), 0) AS down"
        " FROM votes WHERE track_id = ?", (track_id,)
    ) or {"up": 0, "down": 0}
    db.execute(
        "UPDATE tracks SET vote_up = ?, vote_down = ? WHERE id = ?",
        (counts["up"], counts["down"], track_id),
    )
    # Auto-protect popular tracks; protection is sticky until an admin clears it.
    threshold = runtime_get(cfg, db, "dj.protect_threshold")
    if counts["up"] - counts["down"] >= threshold:
        db.execute("UPDATE tracks SET protected = 1 WHERE id = ?", (track_id,))
    return db.one("SELECT * FROM tracks WHERE id = ?", (track_id,)) or {}


def mark_aired(db: DB, track_id: int, airtime_iso: str,
               prog_offset: float | None = None, epoch: int | None = None) -> None:
    db.execute("INSERT INTO play_log (track_id, started_at, prog_offset, epoch)"
               " VALUES (?, ?, ?, ?)", (track_id, airtime_iso, prog_offset, epoch))
    db.execute(
        "UPDATE tracks SET play_count = play_count + 1, last_played_at = ? WHERE id = ?",
        (airtime_iso, track_id),
    )


def my_votes(db: DB, listener_id: str) -> dict[int, int]:
    if not listener_id:
        return {}
    rows = db.query("SELECT track_id, value FROM votes WHERE listener_id = ?", (listener_id,))
    return {r["track_id"]: r["value"] for r in rows}


# ---------------------------------------------------------------- disk usage

def disk_usage(db: DB) -> int:
    row = db.one("SELECT COALESCE(SUM(size_bytes), 0) AS n FROM tracks WHERE status != 'deleted'")
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------- prompt seeds

def import_prompts(db: DB, cfg: Config, only_new: bool = True) -> int:
    """Import TOML prompt files from prompts.seed_dir. Skips existing names."""
    seed_dir = cfg.path("prompts.seed_dir")
    if not seed_dir.is_dir():
        return 0
    imported = 0
    for f in sorted(seed_dir.glob("*.toml")):
        try:
            data = tomllib.loads(f.read_text(encoding="utf-8"))
            name = str(data.get("name") or f.stem).strip()
            style = str(data.get("style") or "").strip()
            song = str(data.get("song") or "").strip()
            if not name or not style:
                log.warning("prompt seed %s: missing name or style, skipped", f.name)
                continue
            exists = db.one("SELECT id FROM prompts WHERE name = ?", (name,))
            if exists and only_new:
                continue
            if exists:
                db.execute(
                    "UPDATE prompts SET style_text = ?, song_text = ?, cot_mode = ? WHERE id = ?",
                    (style, song, str(data.get("cot") or "full"), exists["id"]),
                )
            else:
                db.execute(
                    "INSERT INTO prompts (name, style_text, song_text, cot_mode, enabled, created_at)"
                    " VALUES (?, ?, ?, ?, 1, ?)",
                    (name, style, song, str(data.get("cot") or "full"), iso(utcnow())),
                )
            imported += 1
        except Exception as exc:
            log.warning("prompt seed %s failed: %s", f.name, exc)
    if imported:
        log.info("imported %d prompt(s) from %s", imported, seed_dir)
    return imported
