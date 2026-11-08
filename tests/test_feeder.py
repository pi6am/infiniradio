"""In-place skip: a skip is an early end-of-track, not an encoder flush.

The old behavior restarted the encoder on skip (new stream generation,
purge, manifest gap, every listener rebuilding its session across the seam)
and deleted the airing's play_log row. The new behavior cuts on the same
program timeline: the encoder epoch is unchanged, no restart, the airing
row stays (clamped via played_s to what actually aired so now-playing
hands over at the cut), and the next airing's prog_offset continues at the
cut position.
"""

from __future__ import annotations

import asyncio
import shutil

import _helpers

_helpers.ensure_repo()

from infiniradio import library
from infiniradio.db import DB
from infiniradio.generator.base import GenRequest
from infiniradio.generator.mock import MockGenerator
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import load_workflows
from infiniradio.streamer.feeder import Feeder

from _helpers import RT, isolated_config


class FakeEncoder:
    """Encoder stand-in: paced writes (so skips land deterministically
    mid-track) and a restart counter that must stay at zero."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 7
        self.written = 0
        self.restarts = 0

    @property
    def bytes_per_second(self) -> int:
        return int(self.cfg.get("stream.sample_rate", 48000)) * 2 * 2

    def program_seconds(self) -> float:
        return self.written / self.bytes_per_second

    def queued_seconds(self) -> float:
        return 0.0                      # never pace-throttled

    async def write(self, pcm: bytes) -> None:
        self.written += len(pcm)
        await asyncio.sleep(0.02)       # a 40s track takes ~2.4s to feed

    async def restart(self) -> None:
        self.restarts += 1


async def _air_and_skip(feeder: Feeder, track: dict) -> None:
    task = asyncio.create_task(feeder._feed_track(track))
    await asyncio.sleep(0.5)            # squarely mid-track
    assert feeder.request_skip(), "track should be feeding"
    assert await task is True


async def _main(rt, cfg) -> None:
    db = rt.db
    # a real playable track: mock generator renders a ~40s ffmpeg sine
    req = GenRequest(style="s", lyrics="", seed=101)
    tid = library.create_track(db, cfg, prompt_id=None, parent_track_id=None,
                               style_text="s", lyrics_snapshot="", seed=req.seed,
                               cot_mode="full")
    result = await rt.generator.generate(req, library.track_dir(cfg, tid))
    await library.ingest(db, cfg, tid, result)
    track = db.one("SELECT * FROM tracks WHERE id = ?", (tid,))
    assert track and track["status"] == "ready" and track["playout_path"]
    duration = track["duration_s"]

    feeder = Feeder(rt)
    fake = FakeEncoder(rt.cfg)
    feeder.encoder = fake

    await _air_and_skip(feeder, track)

    # same encoder life: no restart, no new generation, real audio written
    assert fake.restarts == 0 and fake.epoch == 7 and fake.written > 0

    # the airing row stays (a flush used to delete it) and is clamped to
    # the program the track actually occupied before the cut
    row = db.one("SELECT * FROM play_log WHERE track_id = ?", (tid,))
    assert row is not None
    assert row["epoch"] == 7 and row["prog_offset"] == 0.0
    assert row["played_s"] is not None and 0.0 < row["played_s"] < duration
    head_at_cut = fake.program_seconds()
    assert abs(row["played_s"] - head_at_cut) < 0.01
    print(f"in-place skip: aired {row['played_s']:.1f}s of {duration:.0f}s, "
          f"epoch {fake.epoch} unchanged, no encoder restart")

    # the next airing continues on the SAME timeline: its prog_offset
    # starts exactly at the cut, so the schedule hands over at the seam
    await _air_and_skip(feeder, track)
    rows = db.query("SELECT * FROM play_log WHERE track_id = ? ORDER BY id", (tid,))
    assert len(rows) == 2
    assert abs(rows[1]["prog_offset"] - head_at_cut) < 0.01
    assert rows[1]["played_s"] and 0.0 < rows[1]["played_s"] < duration
    assert fake.restarts == 0 and fake.epoch == 7
    print(f"continuity: next airing starts at prog {rows[1]['prog_offset']:.1f}s"
          f" (the cut), aired {rows[1]['played_s']:.1f}s more")

    # every airing counts a play (a regression once made this UPDATE dead
    # code while the play_log INSERT kept working — tracks replayed during
    # that window froze at their old counts)
    assert db.one("SELECT play_count AS pc FROM tracks WHERE id = ?",
                  (tid,))["pc"] == 2
    print("play_count: both airings counted")

    # mark_aired hands back the row id (the clamp above depends on it)
    rid = library.mark_aired(db, tid, "2026-01-01T00:00:00",
                             prog_offset=1.0, epoch=7)
    assert rid == db.one("SELECT MAX(id) AS m FROM play_log")["m"]

    # migration 13 repairs the frozen window from play_log: counts ratchet
    # UP to the airing rows, never down (counts above the row count are
    # legitimate — the old skip behavior deleted a flushed airing's row).
    # Replay = run 13's SQL directly and re-stamp it (and every LATER
    # one-shot migration, e.g. 14's ALTERs) as already applied.
    from infiniradio.db import MIGRATIONS

    def _replay_13():
        later = [r["v"] for r in db.query(
            "SELECT version AS v FROM schema_version WHERE version > 13")]
        db.execute("DELETE FROM schema_version WHERE version >= 13")
        db.conn.executescript(MIGRATIONS[13])
        for v in (13, *later):
            db.execute("INSERT INTO schema_version (version) VALUES (?)", (v,))

    db.execute("UPDATE tracks SET play_count = 1 WHERE id = ?", (tid,))
    _replay_13()
    assert db.one("SELECT play_count AS pc FROM tracks WHERE id = ?",
                  (tid,))["pc"] == 3   # three airing rows now
    db.execute("UPDATE tracks SET play_count = 99 WHERE id = ?", (tid,))
    _replay_13()
    assert db.one("SELECT play_count AS pc FROM tracks WHERE id = ?",
                  (tid,))["pc"] == 99
    print("migration 13: ratchets frozen counts up to airing rows, never lowers")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")
        rt = RT(cfg, db, MockGenerator(cfg), MockLLMClient(), load_workflows(cfg))
        asyncio.run(_main(rt, cfg))
        print("test_feeder: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
