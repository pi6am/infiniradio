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
from datetime import timedelta
from pathlib import Path

from . import naming
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
                 style_text: str, lyrics_snapshot: str, seed: int, cot_mode: str,
                 workflow_id: str | None = None, title: str = "",
                 theme_title: str = "", style_id: int | None = None) -> int:
    """Placeholder row for a generating track.

    title: set (LLM-extracted song name) when the workflow produced one;
    empty for a random naming title assigned at ingest. theme_title: the
    seed title that themed the lyrics workflow — shown as the display
    subtitle for workflow tracks. style_id: the style the DJ picked for
    this job (lineage + per-style vote aggregation).
    """
    track_id = db.execute(
        "INSERT INTO tracks (title, parent_track_id, prompt_id, workflow_id, style_id,"
        " theme_title, style_text, lyrics_snapshot, seed, cot_mode, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'generating', ?)",
        (title, parent_track_id, prompt_id, workflow_id, style_id, theme_title or "",
         style_text, lyrics_snapshot, seed, cot_mode, iso(utcnow())),
    )
    track_dir(cfg, track_id).mkdir(parents=True, exist_ok=True)
    return track_id



async def ingest(db: DB, cfg: Config, track_id: int, result: GenResult,
                 tags: list[str] | None = None) -> dict:
    """Complete a generating track: probe, transcode playout, finalize DB row.

    tags (from a lyrics workflow's outputs) are applied via the vocabulary
    after the row is finalized.
    """
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
    parent = db.one("SELECT * FROM tracks WHERE id = ?", (row.get("parent_track_id"),)) \
        if row.get("parent_track_id") else None
    # Workflow tracks keep their LLM-extracted title (assigned at creation);
    # everything else gets a random generated title. The prompt/lineage
    # context stays in the DB (admin/player metadata).
    title = (row.get("title") or "").strip() or naming.unique_title(db)

    key, tempo = parse_abc_header(result.abc_text or "")
    db.execute(
        "UPDATE tracks SET title = ?, duration_s = ?, size_bytes = ?, music_key = ?, tempo = ?,"
        " original_path = ?, playout_path = ?, abc_text = ?, status = 'ready'"
        " WHERE id = ?",
        (title, duration, size, key, tempo,
         f"{track_id}/{original.name}", f"{track_id}/playout.m4a",
         result.abc_text or "", track_id),
    )
    # Tags from the lyrics workflow (static + LLM-picked), validated against
    # the vocabulary: unknown tags are added or dropped per lyrics.allow_new_tags.
    if tags:
        applied = apply_track_tags(db, cfg, track_id, tags)
        if applied:
            log.info("track %s tagged: %s", track_id, ", ".join(applied))
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


# ------------------------------------------------------------- deletion buffer

def schedule_delete(db: DB, cfg: Config, track_id: int) -> bool:
    """Condemn a track: move it to the trash (status 'trashed', audio kept,
    never aired again, still counting toward quota) with delete_after = now
    + library.deletion_buffer_hours. The janitor — never this call — turns
    trashed into a real deletion, and only once the deadline has passed AND
    the track is not currently airing: trashing is always safe, even for
    the song on air right now (it simply finishes its airing). Buffer 0
    means the trash expires immediately; the janitor collects it on its
    next tick. Failed/generating placeholders are deleted outright — no
    audio to review, never airing.
    """
    row = db.one("SELECT status FROM tracks WHERE id = ?", (track_id,))
    if not row or row["status"] in ("deleted", "trashed"):
        return False
    db.execute("DELETE FROM play_queue WHERE track_id = ?", (track_id,))
    buffer_h = float(runtime_get(cfg, db, "library.deletion_buffer_hours"))
    if row["status"] == "ready":
        delete_after = iso(utcnow() + timedelta(hours=buffer_h))
        db.execute("UPDATE tracks SET status = 'trashed', delete_after = ? WHERE id = ?",
                   (delete_after, track_id))
        log.info("track %s trashed until %s (%.0fh buffer)",
                 track_id, delete_after, buffer_h)
        return True
    return delete_track(db, cfg, track_id)


def undelete_track(db: DB, track_id: int) -> bool:
    """Admin rescue: a trashed track returns to ready (votes and stats
    untouched — the decision to keep it is human)."""
    row = db.one("SELECT status FROM tracks WHERE id = ?", (track_id,))
    if not row or row["status"] != "trashed":
        return False
    db.execute("UPDATE tracks SET status = 'ready', delete_after = NULL WHERE id = ?",
               (track_id,))
    log.info("track %s restored (was trashed)", track_id)
    return True


def trashed_bytes(db: DB) -> int:
    return int(db.one(
        "SELECT COALESCE(SUM(size_bytes), 0) AS n FROM tracks WHERE status = 'trashed'"
    )["n"] or 0)


def finalize_deletions(db: DB, cfg: Config, active_ids: set[int] = frozenset()) -> int:
    """Empty the trash: hard-delete trashed tracks whose deadline has
    passed AND that are not currently airing (the file stays valid for the
    feeder until the airing ends; deleting under it would also fail on
    platforms that lock open files). Runs on the janitor's clock and ONLY
    on the deadline — quota pressure never fast-tracks it (that would
    defeat the anti-griefing purpose: a vote wave plus a full disk could
    still destroy tracks before review)."""
    now = iso(utcnow())
    rows = db.query("SELECT id FROM tracks WHERE status = 'trashed'"
                    " AND delete_after IS NOT NULL AND delete_after <= ?", (now,))
    n = 0
    for r in rows:
        if r["id"] in active_ids:
            continue
        if delete_track(db, cfg, r["id"]):
            n += 1
    return n


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
               prog_offset: float | None = None, epoch: int | None = None,
               played_s: float | None = None,
               resume_row_id: int | None = None) -> None:
    if resume_row_id is not None:
        # Restart resume: the airing cut off by the restart continues from
        # its remainder. Update the existing row in place — one airing, one
        # play count — with the fresh airtime and how much program it now
        # occupies (played_s) so the now-playing timeline stays exact.
        db.execute(
            "UPDATE play_log SET started_at = ?, prog_offset = ?, epoch = ?,"
            " played_s = ? WHERE id = ?",
            (airtime_iso, prog_offset, epoch, played_s, resume_row_id))
        db.execute("UPDATE tracks SET last_played_at = ? WHERE id = ?",
                   (airtime_iso, track_id))
        return
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


# ---------------------------------------------------------------- tags

def normalize_tag(name: str) -> str:
    return name.strip().strip("`\"'").strip().lower()


def _tag_id(db: DB, name: str, *, allow_new: bool = True) -> int | None:
    """Get-or-create a vocabulary tag; None when the name is unusable or
    creation is disabled. Curated content (tags.toml seeds, prompt/style
    tags from imports and the admin panel) always creates — allow_new
    governs only RUNTIME picks (lyrics.allow_new_tags), never the
    seeders: gating those made a fresh station with
    allow_new_tags = false unable to build any vocabulary at all."""
    name = normalize_tag(name)
    if not name:
        return None
    row = db.one("SELECT id FROM tags WHERE name = ?", (name,))
    if row is not None:
        return row["id"]
    if not allow_new:
        return None
    return db.execute("INSERT INTO tags (name, created_at) VALUES (?, ?)",
                      (name, iso(utcnow())))


def load_tag_seeds(cfg: Config) -> tuple[list[str], list[str]]:
    """(llm_tags, static_tags) from the configured seed file — the pure
    loader, usable without a DB (the audition CLI uses it too)."""
    path = cfg.path("tags.seed_file")
    if not path.is_file():
        return [], []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"cannot read tag seed file {path}: {exc}") from exc
    llm = [normalize_tag(t) for t in data.get("llm_tags") or []]
    static = [normalize_tag(t) for t in data.get("static_tags") or []]
    llm = [t for t in llm if t and t not in static]
    return llm, [t for t in static if t]


def seed_tags(db: DB, cfg: Config) -> int:
    """Seed the tag vocabulary from tags.toml at every startup: insert
    missing names and refresh the llm_pickable flags. The file curates the
    approved list; the DB accumulates usage and per-tag stats. Returns the
    number of LLM-pickable tags."""
    llm, static = load_tag_seeds(cfg)
    if not llm and not static:
        return 0
    for name in llm:
        tag_id = _tag_id(db, name)   # curated: always creatable
        if tag_id:
            db.execute("UPDATE tags SET llm_pickable = 1 WHERE id = ?", (tag_id,))
    for name in static:
        tag_id = _tag_id(db, name)
        if tag_id:
            db.execute("UPDATE tags SET llm_pickable = 0 WHERE id = ?", (tag_id,))
    if llm:
        log.info("tag vocabulary: %d LLM-pickable, %d static", len(llm), len(static))
    return len(llm)


def apply_track_tags(db: DB, cfg: Config, track_id: int, names: list[str]) -> list[str]:
    """Link tags to a track. This is the one RUNTIME-gated path: unknown
    names are created only when lyrics.allow_new_tags is true (engine-side
    sanitization already limits picks to the vocabulary, so this mostly
    covers workflow static tags the vocabulary hasn't adopted yet)."""
    allow_new = bool(cfg.get("lyrics.allow_new_tags", True))
    applied: list[str] = []
    for name in names:
        tag_id = _tag_id(db, name, allow_new=allow_new)
        if tag_id is None:
            continue
        db.execute("INSERT OR IGNORE INTO track_tags (track_id, tag_id) VALUES (?, ?)",
                   (track_id, tag_id))
        applied.append(normalize_tag(name))
    return applied


def apply_prompt_tags(db: DB, cfg: Config, prompt_id: int, names: list[str]) -> list[str]:
    """Same for prompts (static tags from the seed files / admin panel —
    curated, so tag creation is never blocked here)."""
    applied: list[str] = []
    for name in names:
        tag_id = _tag_id(db, name)
        if tag_id is None:
            continue
        db.execute("INSERT OR IGNORE INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)",
                   (prompt_id, tag_id))
        applied.append(normalize_tag(name))
    return applied


def track_tags(db: DB, track_id: int) -> list[str]:
    return [r["name"] for r in db.query(
        "SELECT t.name FROM track_tags tt JOIN tags t ON t.id = tt.tag_id"
        " WHERE tt.track_id = ? ORDER BY t.name", (track_id,))]


def prompt_tags(db: DB, prompt_id: int) -> list[str]:
    return [r["name"] for r in db.query(
        "SELECT t.name FROM prompt_tags pt JOIN tags t ON t.id = pt.tag_id"
        " WHERE pt.prompt_id = ? ORDER BY t.name", (prompt_id,))]


def style_tags(db: DB, style_id: int) -> list[str]:
    return [r["name"] for r in db.query(
        "SELECT t.name FROM style_tags st JOIN tags t ON t.id = st.tag_id"
        " WHERE st.style_id = ? ORDER BY t.name", (style_id,))]


def apply_style_tags(db: DB, cfg: Config, style_id: int, names: list[str]) -> list[str]:
    """Style tags (seed files + admin panel — curated, always creatable)."""
    applied: list[str] = []
    for name in names:
        tag_id = _tag_id(db, name)
        if tag_id is None:
            continue
        db.execute("INSERT OR IGNORE INTO style_tags (style_id, tag_id) VALUES (?, ?)",
                   (style_id, tag_id))
        applied.append(normalize_tag(name))
    return applied


def tag_vocabulary(db: DB) -> list[str]:
    """LLM-pickable tag names — what a workflow's tagging stage is offered.
    Forced/static tags (vocal, english, prompt genres) are never in this
    list; presenting them as choices is what makes the LLM dither."""
    return [r["name"] for r in db.query(
        "SELECT name FROM tags WHERE llm_pickable = 1 ORDER BY name")]


# ---------------------------------------------------------------- style seeds

def import_styles(db: DB, cfg: Config, only_new: bool = True) -> int:
    """Import TOML style seeds from styles.seed_dir. Skips existing names
    unless only_new is false (the admin "re-import seeds" button), which
    updates definitions in place. The prompts lifecycle: after first boot
    the database is the source of truth."""
    from . import styles as styles_mod

    seed_dir = cfg.path("styles.seed_dir")
    if not seed_dir.is_dir():
        return 0
    imported = 0
    for f in sorted(seed_dir.glob("*.toml")):
        try:
            style = styles_mod.load_style_file(f)
        except styles_mod.StyleError as exc:
            log.warning("style seed %s failed: %s", f.name, exc)
            continue
        exists = db.one("SELECT id FROM styles WHERE name = ?", (style.name,))
        if exists and only_new:
            continue
        vars_json = json.dumps(style.vars, ensure_ascii=False)
        weighted_json = json.dumps({k: [[w, v] for w, v in entries]
                                    for k, entries in style.weighted.items()},
                                   ensure_ascii=False)
        if exists:
            db.execute(
                "UPDATE styles SET description = ?, vars_json = ?, weighted_json = ?,"
                " enabled = ? WHERE id = ?",
                (style.description, vars_json, weighted_json, int(style.enabled), exists["id"]))
            style_id = exists["id"]
        else:
            style_id = db.execute(
                "INSERT INTO styles (name, description, vars_json, weighted_json,"
                " enabled, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (style.name, style.description, vars_json, weighted_json,
                 int(style.enabled), iso(utcnow())))
        if style.tags:
            apply_style_tags(db, cfg, style_id, list(style.tags))
        imported += 1
    if imported:
        log.info("imported %d style(s) from %s", imported, seed_dir)
    return imported


# ---------------------------------------------------------------- prompt seeds

def import_prompts(db: DB, cfg: Config, only_new: bool = True) -> int:
    """Import TOML prompt files from prompts.seed_dir. Skips existing names.

    Also seeds each prompt's tags: the file's ``tags`` list when present,
    else (backfill for pre-tag stations) the prompt name split on
    dashes/underscores — "chiptune-upbeat" -> chiptune, upbeat.
    """
    seed_dir = cfg.path("prompts.seed_dir")
    if not seed_dir.is_dir():
        return 0
    imported = 0
    file_tags: dict[str, list[str]] = {}
    for f in sorted(seed_dir.glob("*.toml")):
        try:
            data = tomllib.loads(f.read_text(encoding="utf-8"))
            name = str(data.get("name") or f.stem).strip()
            style = str(data.get("style") or "").strip()
            song = str(data.get("song") or "").strip()
            if not name or not style:
                log.warning("prompt seed %s: missing name or style, skipped", f.name)
                continue
            file_tags[name] = [str(t) for t in data.get("tags") or []]
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

    # Tag seeding / backfill (idempotent: prompts that already have tags
    # keep them — after first boot the database is the source of truth).
    seeded = 0
    for p in db.query("SELECT id, name FROM prompts"):
        if db.one("SELECT 1 FROM prompt_tags WHERE prompt_id = ? LIMIT 1", (p["id"],)):
            continue
        names = file_tags.get(p["name"])
        if not names:
            names = [part for part in re.split(r"[-_]+", p["name"].lower()) if part]
        if apply_prompt_tags(db, cfg, p["id"], names):
            seeded += 1
    if seeded:
        log.info("seeded tags for %d prompt(s)", seeded)
    return imported
