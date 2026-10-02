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
    6: """
    -- LLM lyrics workflows (optional; see infiniradio/lyrics.py).
    -- Jobs gain a two-phase lifecycle: phase 'lyrics' (LLM pipeline) then
    -- 'render' (audio). progress holds the workflow engine's resumable
    -- state (completed stages, saved variables, conversation threads) plus
    -- the final outputs; job_stage_log is the append-only diagnostic trace
    -- of every LLM call — prompt sent, reply, gate verdicts, token usage.
    ALTER TABLE generation_jobs ADD COLUMN workflow_id TEXT;
    ALTER TABLE generation_jobs ADD COLUMN phase TEXT NOT NULL DEFAULT 'render';
    ALTER TABLE generation_jobs ADD COLUMN progress TEXT NOT NULL DEFAULT '{}';

    CREATE TABLE job_stage_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id INTEGER NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
      stage_id TEXT NOT NULL,
      attempt INTEGER NOT NULL DEFAULT 1,
      prompt_text TEXT NOT NULL DEFAULT '',
      response_text TEXT NOT NULL DEFAULT '',
      vars_json TEXT NOT NULL DEFAULT '{}',
      gate_json TEXT NOT NULL DEFAULT '[]',
      model TEXT NOT NULL DEFAULT '',
      usage_json TEXT NOT NULL DEFAULT '{}',
      duration_s REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );
    CREATE INDEX idx_stage_log_job ON job_stage_log(job_id);

    -- Tracks from a workflow keep their lineage: the workflow slug, the
    -- seed title that themed the brainstorm (shown as the display subtitle:
    -- "The Shape of a Dream · Gauzy and Stated"), and tags for cross-cutting
    -- vote aggregation (static workflow tags + LLM-picked dynamic tags).
    ALTER TABLE tracks ADD COLUMN workflow_id TEXT;
    ALTER TABLE tracks ADD COLUMN theme_title TEXT;

    CREATE TABLE tags (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    CREATE TABLE track_tags (
      track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
      tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
      PRIMARY KEY (track_id, tag_id)
    );
    CREATE TABLE prompt_tags (
      prompt_id INTEGER NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
      tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
      PRIMARY KEY (prompt_id, tag_id)
    );
    CREATE INDEX idx_tracks_workflow ON tracks(workflow_id);
    """,
    7: """
    -- Tag vocabulary curation: tags.toml seeds the vocabulary at every
    -- startup and decides which tags are offered to lyrics workflows'
    -- tagging stage (llm_pickable). Genre/vocal tags are pickable;
    -- workflow-forced tags ("vocal", "english") and prompt statics
    -- ("instrumental") are vocabulary-only. Runtime-added tags default to
    -- not pickable until the seed file adopts them.
    ALTER TABLE tags ADD COLUMN llm_pickable INTEGER NOT NULL DEFAULT 0;
    """,
    8: """
    -- Song styles: named variable-override bundles the DJ picks with the
    -- same weighted-bandit rule as prompts/workflows (votes steer styles).
    -- vars_json/weighted_json hold the [vars] fixed overrides and the
    -- [weighted] choices resolved by seeded RNG at commission time; the
    -- resolved values travel per-job in progress JSON. Prompts lifecycle:
    -- styles/*.toml seeds imported on first boot, DB source of truth after.
    CREATE TABLE styles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL UNIQUE,
      description TEXT NOT NULL DEFAULT '',
      vars_json TEXT NOT NULL DEFAULT '{}',
      weighted_json TEXT NOT NULL DEFAULT '{}',
      enabled INTEGER NOT NULL DEFAULT 1,
      times_used INTEGER NOT NULL DEFAULT 0,
      last_used_at TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE style_tags (
      style_id INTEGER NOT NULL REFERENCES styles(id) ON DELETE CASCADE,
      tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
      PRIMARY KEY (style_id, tag_id)
    );
    ALTER TABLE generation_jobs ADD COLUMN style_id INTEGER REFERENCES styles(id);
    ALTER TABLE tracks ADD COLUMN style_id INTEGER REFERENCES styles(id);
    CREATE INDEX idx_tracks_style ON tracks(style_id);
    """,
    9: """
    -- Deletion buffer: a condemned track first moves to status 'trashed'
    -- (audio kept on disk, still counting toward quota, never aired) until
    -- delete_after passes — protection against a wave of malicious
    -- downvotes deleting good music before anyone can review it. An admin
    -- can restore it while trashed. hold: admin-set "never air this track"
    -- regardless of votes or status.
    ALTER TABLE tracks ADD COLUMN delete_after TEXT;
    ALTER TABLE tracks ADD COLUMN hold INTEGER NOT NULL DEFAULT 0;
    """,
    10: """
    -- Job phase timing diagnostics: when each phase started/ended, so the
    -- admin job view can break a job's lifetime into waiting / lyrics /
    -- waiting / rendering components (quantifying, e.g., whether a new
    -- LLM endpoint is actually faster). Timestamps are (re)set at each
    -- claim, so they describe the most recent run of each phase.
    ALTER TABLE generation_jobs ADD COLUMN lyrics_started_at TEXT;
    ALTER TABLE generation_jobs ADD COLUMN lyrics_finished_at TEXT;
    ALTER TABLE generation_jobs ADD COLUMN render_started_at TEXT;
    """,
    11: """
    -- Re-compose: a variation that DISCARDS the parent's saved ABC and
    -- re-plans it from (admin-edited) lyrics — the repair path for songs
    -- whose ABC was built from mangled lyrics. The job carries its own
    -- outputs (edited lyrics, title, style) in progress, like a finished
    -- lyrics job; render-time abc=None routes through audited planning.
    ALTER TABLE generation_jobs ADD COLUMN recompose INTEGER NOT NULL DEFAULT 0;
    """,
    12: """
    -- Reproducibility: the seed actually submitted to ComfyUI for the
    -- render. Usually equals the job's requested seed, EXCEPT when the
    -- ABC audit rejected the first plan(s) and a retry seed was drawn —
    -- those were previously unrecorded, making failed renders impossible
    -- to replay. (Plan-pass seeds live in the server log.)
    ALTER TABLE generation_jobs ADD COLUMN used_seed INTEGER;
    """,
    13: """
    -- Data repair: play_count stopped incrementing for a window (a code
    -- regression left the increment unreachable while the play_log INSERT
    -- kept working), so airings from that window have rows but no count.
    -- play_log is the complete airing record — ratchet each track's count
    -- up to its number of rows. The MAX() direction matters: counts may
    -- legitimately exceed row counts (the old skip behavior deleted a
    -- flushed airing's row without decrementing), so never lower one.
    UPDATE tracks SET play_count = (
      SELECT COUNT(*) FROM play_log WHERE play_log.track_id = tracks.id
    )
    WHERE play_count < (
      SELECT COUNT(*) FROM play_log WHERE play_log.track_id = tracks.id
    );
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
