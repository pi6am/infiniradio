"""Deletion buffer + hold: downvote eviction -> trash (audio kept, quota
charged, never aired), admin restore, deadline-only finalization that never
fires mid-airing, buffer-0 semantics, quota effective-usage eviction, and
hold excluding tracks from autoplay while manual queueing still works.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import timedelta
from types import SimpleNamespace

import _helpers

_helpers.ensure_repo()

from infiniradio import dj, janitor, library
from infiniradio.db import DB
from infiniradio.generator.base import GenRequest
from infiniradio.generator.mock import MockGenerator
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import load_workflows
from infiniradio.streamer.feeder import Feeder
from infiniradio.util import iso, utcnow

from _helpers import RT, isolated_config


def _make_track(rt, db, cfg, plays=2, up=0, down=5) -> int:
    req = GenRequest(style="s", lyrics="", seed=100)
    tid = library.create_track(db, cfg, prompt_id=None, parent_track_id=None,
                               style_text="s", lyrics_snapshot="", seed=req.seed,
                               cot_mode="full")
    result = asyncio.run(rt.generator.generate(req, library.track_dir(cfg, tid)))
    asyncio.run(library.ingest(db, cfg, tid, result))
    db.execute("UPDATE tracks SET play_count = ?, vote_up = ?, vote_down = ?,"
               " last_played_at = ? WHERE id = ?",
               (plays, up, down, iso(utcnow() - timedelta(hours=12)), tid))
    return tid


def _main(rt, cfg) -> None:
    db = rt.db
    db.set_setting("library.deletion_buffer_hours", 72.0)
    db.set_setting("dj.runway_minutes", 999)  # keep commissioning out of the way

    # 1. the anti-grief scenario: a wave-downvoted good track
    tid = _make_track(rt, db, cfg, plays=3, up=4, down=7)
    before = library.disk_usage(db)
    files_kept = (library.track_dir(cfg, tid) / "playout.m4a").is_file()
    db.execute("INSERT INTO play_queue (track_id, queued_at) VALUES (?, ?)",
               (tid, iso(utcnow())))
    summary = janitor.run_janitor(db, cfg, active_ids=set())
    row = db.one("SELECT * FROM tracks WHERE id = ?", (tid,))
    assert summary["downvote_evicted"] == 1 and row["status"] == "trashed"
    assert row["delete_after"] and row["delete_after"] > iso(utcnow())
    assert (library.track_dir(cfg, tid) / "playout.m4a").is_file() is files_kept
    assert library.disk_usage(db) == before                    # counts toward quota
    assert db.one("SELECT COUNT(*) AS n FROM play_queue WHERE track_id = ?",
                  (tid,))["n"] == 0                            # queue purged
    assert tid not in [t["id"] for t in dj.eligible_tracks(db, cfg)]
    print(f"1. downvote eviction -> trashed (until {row['delete_after']}): audio kept, "
          "quota charged, queue purged, not eligible")

    # 2. admin rescue
    assert library.undelete_track(db, tid)
    row = db.one("SELECT status, delete_after FROM tracks WHERE id = ?", (tid,))
    assert row["status"] == "ready" and row["delete_after"] is None
    assert tid in [t["id"] for t in dj.eligible_tracks(db, cfg)]
    assert not library.undelete_track(db, tid)  # only trashed tracks
    print("2. restore -> ready, airs again")

    # 3. deadline finalization — never early, never mid-airing
    library.schedule_delete(db, cfg, tid)
    assert library.finalize_deletions(db, cfg) == 0  # deadline is 72h out
    db.execute("UPDATE tracks SET delete_after = ? WHERE id = ?",
               (iso(utcnow() - timedelta(seconds=1)), tid))
    assert library.finalize_deletions(db, cfg, active_ids={tid}) == 0  # on air
    assert db.one("SELECT status FROM tracks WHERE id = ?", (tid,))["status"] == "trashed"
    assert (library.track_dir(cfg, tid) / "playout.m4a").is_file()
    assert library.finalize_deletions(db, cfg) == 1  # airing ended -> collected
    row = db.one("SELECT status, size_bytes FROM tracks WHERE id = ?", (tid,))
    assert row["status"] == "deleted" and row["size_bytes"] == 0
    assert not library.track_dir(cfg, tid).exists()
    print("3. finalize: nothing before the deadline, never mid-airing, "
          "hard delete once free")

    # 4. buffer 0 -> trash with immediate expiry; the janitor collects
    db.set_setting("library.deletion_buffer_hours", 0.0)
    tid2 = _make_track(rt, db, cfg)
    assert library.schedule_delete(db, cfg, tid2)
    row = db.one("SELECT status, delete_after FROM tracks WHERE id = ?", (tid2,))
    assert row["status"] == "trashed" and row["delete_after"] <= iso(utcnow())
    assert library.finalize_deletions(db, cfg) == 1
    assert not library.track_dir(cfg, tid2).exists()
    print("4. buffer 0 -> trashed with expired deadline, janitor collects")

    # 5. quota eviction marks trashed and stops at the effective target
    #    (trashed bytes count as already-freed; real usage still pauses gen)
    db.set_setting("library.deletion_buffer_hours", 72.0)
    db.set_setting("library.quota_gb", 0.00001)   # ~10 KiB: always over quota
    for i in range(5):
        _make_track(rt, db, cfg, plays=2, down=9 - i)
    summary = janitor.run_janitor(db, cfg, active_ids=set())
    ready = db.one("SELECT COUNT(*) AS n FROM tracks WHERE status = 'ready'")["n"]
    trashed = db.one("SELECT COUNT(*) AS n FROM tracks WHERE status = 'trashed'")["n"]
    assert summary["downvote_evicted"] + summary["quota_evicted"] >= 1
    assert janitor.generation_paused(db, cfg)[0] is True  # trashed still counts
    eff = library.disk_usage(db) - library.trashed_bytes(db)
    assert eff == 0 or ready == 0, (eff, ready)  # evicted until nothing left
    assert library.finalize_deletions(db, cfg) == 0  # quota never fast-tracks
    print(f"5. quota eviction: {trashed} trashed, {ready} ready left, "
          "generation paused (real usage), no fast-track")

    # 6. hold: the DJ never picks held tracks, but manual queueing is
    #    deliberate admin intent and still plays them
    db.set_setting("library.quota_gb", 50.0)
    tid3 = _make_track(rt, db, cfg, plays=1)
    db.execute("UPDATE tracks SET hold = 1 WHERE id = ?", (tid3,))
    assert tid3 not in [t["id"] for t in dj.eligible_tracks(db, cfg)]
    assert dj.pick_next(db, cfg) is None  # the only ready track is held
    assert dj.variation_parent(db, cfg) is None or \
        dj.variation_parent(db, cfg)["id"] != tid3
    db.execute("INSERT INTO play_queue (track_id, queued_at) VALUES (?, ?)",
               (tid3, iso(utcnow())))
    popped = Feeder._pop_queue(SimpleNamespace(rt=rt))
    assert popped is not None and popped["id"] == tid3
    db.execute("UPDATE tracks SET hold = 0 WHERE id = ?", (tid3,))
    db.execute("INSERT INTO play_queue (track_id, queued_at) VALUES (?, ?)",
               (tid3, iso(utcnow())))
    assert Feeder._pop_queue(SimpleNamespace(rt=rt))["id"] == tid3
    print("6. hold: DJ/variations excluded, manual queue still plays it, reversible")

    # 7. failed placeholders skip the buffer even when it is configured
    fid = library.create_track(db, cfg, prompt_id=None, parent_track_id=None,
                               style_text="s", lyrics_snapshot="", seed=1, cot_mode="full")
    library.fail_track(db, fid)
    library.schedule_delete(db, cfg, fid)
    assert db.one("SELECT status FROM tracks WHERE id = ?", (fid,))["status"] == "deleted"
    print("7. failed placeholders delete immediately (nothing to review)")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")
        rt = RT(cfg, db, MockGenerator(cfg), MockLLMClient(), load_workflows(cfg))
        _main(rt, cfg)
        print("test_deletion: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
