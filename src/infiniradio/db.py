"""SQLite access layer: one shared connection guarded by a lock (WAL mode).

All writes are small and fast; serializing them behind an RLock is well within
budget for a single-node radio station.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .util import iso, utcnow

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE prompts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      style_text TEXT NOT NULL,
      song_text TEXT NOT NULL,
      cot_mode TEXT NOT NULL DEFAULT 'full',
      enabled INTEGER NOT NULL DEFAULT 1,
      times_used INTEGER NOT NULL DEFAULT 0,
      last_used_at TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE tracks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL DEFAULT '',
      parent_track_id INTEGER REFERENCES tracks(id),
      prompt_id INTEGER REFERENCES prompts(id),
      style_text TEXT NOT NULL DEFAULT '',
      lyrics_snapshot TEXT NOT NULL DEFAULT '',
      seed INTEGER NOT NULL DEFAULT 0,
      cot_mode TEXT NOT NULL DEFAULT 'full',
      status TEXT NOT NULL DEFAULT 'generating',
      protected INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      duration_s REAL NOT NULL DEFAULT 0,
      size_bytes INTEGER NOT NULL DEFAULT 0,
      play_count INTEGER NOT NULL DEFAULT 0,
      last_played_at TEXT,
      vote_up INTEGER NOT NULL DEFAULT 0,
      vote_down INTEGER NOT NULL DEFAULT 0,
      music_key TEXT NOT NULL DEFAULT '',
      tempo INTEGER NOT NULL DEFAULT 0,
      original_path TEXT NOT NULL DEFAULT '',
      playout_path TEXT NOT NULL DEFAULT '',
      abc_text TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX idx_tracks_status ON tracks(status);
    CREATE INDEX idx_tracks_prompt ON tracks(prompt_id);

    CREATE TABLE votes (
      track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
      listener_id TEXT NOT NULL,
      value INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (track_id, listener_id)
    );

    CREATE TABLE generation_jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      prompt_id INTEGER REFERENCES prompts(id),
      parent_track_id INTEGER REFERENCES tracks(id),
      style_override TEXT,
      seed INTEGER,
      status TEXT NOT NULL DEFAULT 'queued',
      comfy_prompt_id TEXT,
      error TEXT,
      track_id INTEGER REFERENCES tracks(id),
      submitted_at TEXT,
      finished_at TEXT,
      created_at TEXT NOT NULL
    );
    CREATE INDEX idx_jobs_status ON generation_jobs(status);

    CREATE TABLE play_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      track_id INTEGER NOT NULL REFERENCES tracks(id),
      started_at TEXT NOT NULL
    );
    CREATE INDEX idx_play_log_start ON play_log(started_at);

    CREATE TABLE settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    """,
    2: """
    CREATE TABLE play_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
      queued_at TEXT NOT NULL
    );
    """,
    3: """
    -- Frame-accurate now-playing: program offset of each airing within the
    -- current encoder life, matching the HLS EXT-X-PROGRAM-DATE-TIME timeline.
    ALTER TABLE play_log ADD COLUMN prog_offset REAL;
    ALTER TABLE play_log ADD COLUMN epoch INTEGER;
    """,
    4: """
    -- Listener telemetry: unique listeners per UTC minute (rows only for
    -- minutes with activity; hourly-averaged compaction can come later)
    -- and cumulative unique listeners per track airing.
    CREATE TABLE listener_minutes (
      minute TEXT PRIMARY KEY,
      listeners INTEGER NOT NULL
    );
    ALTER TABLE tracks ADD COLUMN listener_count INTEGER NOT NULL DEFAULT 0;
    """,
    5: """
    -- Restart resume: seconds of program an airing actually occupies.
    -- NULL means the track's full duration (every normal airing); a resumed
    -- airing (server restart mid-song) carries only its remainder so the
    -- now-playing timeline stays frame-accurate.
    ALTER TABLE play_log ADD COLUMN played_s REAL;
    """,
}


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=10000")
        self.migrate()

    def migrate(self) -> None:
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            for version in sorted(MIGRATIONS):
                if version <= current:
                    continue
                self.conn.executescript(MIGRATIONS[version])
                self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                self.conn.commit()

    # -- helpers -----------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        with self.lock:
            cur = self.conn.execute(sql, params)
            r = cur.fetchone()
            return dict(r) if r is not None else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid or 0

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        with self.lock:
            self.conn.executemany(sql, seq)
            self.conn.commit()

    # -- settings ----------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def touch(self, table: str, row_id: int, column: str, value: Any) -> None:
        assert column.isidentifier() and table.isidentifier()
        self.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (value, row_id))


def now_iso() -> str:
    return iso(utcnow())
