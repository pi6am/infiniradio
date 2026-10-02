"""DJ track selection: the quality x novelty x uncertainty weight, and the
starvation relaxation that keeps weighted picking alive when the repeat
windows empty out.

Pinned properties: votes and (weakly) listeners count as evidence, plays
never do; the novelty factor halves at dj.novelty_plays_scale recent plays
and forgets plays older than dj.novelty_window_hours (no permanent fatigue
ratchet); strict eligibility — what commissioning's runway measures — is
never relaxed by the picker.
"""

from __future__ import annotations

import random
import shutil
from datetime import timedelta

import _helpers

_helpers.ensure_repo()

from infiniradio import dj
from infiniradio.db import DB
from infiniradio.util import iso, utcnow

from _helpers import isolated_config


def _add(db, title: str, **kw) -> int:
    cols = {"title": title, "status": "ready",
            "created_at": "2026-01-01T00:00:00", **kw}
    db.execute(
        f"INSERT INTO tracks ({','.join(cols)})"
        f" VALUES ({','.join('?' * len(cols))})",
        tuple(cols.values()))
    return db.one("SELECT id FROM tracks WHERE title = ?", (title,))["id"]


def _air(db, track_id: int, age_h: float) -> None:
    at = iso(utcnow() - timedelta(hours=age_h))
    db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
               (track_id, at))
    db.execute("UPDATE tracks SET play_count = play_count + 1,"
               " last_played_at = ? WHERE id = ?", (at, track_id))


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")

        def w(up=0, down=0, listeners=0, plays=0, recent=0):
            track = {"vote_up": up, "vote_down": down,
                     "listener_count": listeners, "play_count": plays}
            return dj._pick_weight(cfg, db, track, recent)

        # 1. evidence: nobody heard it -> keeps the exploration bonus;
        #    30 people heard and ignored it -> fades toward baseline
        assert w() > w(listeners=1) > w(listeners=30)
        assert w() > w(down=5) > w(down=50)      # equal quality, more evidence
        assert w(up=50) > w(up=1) > w()          # quality still steers
        # total plays count for NOTHING (the old fatigue ratchet is gone)
        assert w(plays=0) == w(plays=500)
        print("1. evidence = votes + weak listeners; plays irrelevant")

        # 2. novelty: X / (X + recent plays); halves at X recent plays
        w0 = w()
        assert abs(w(recent=1) / w0 - 0.5) < 1e-9
        assert abs(w(recent=2) / w0 - 1 / 3) < 1e-9
        db.set_setting("dj.novelty_plays_scale", 2.0)
        assert abs(w(recent=1) / w0 - 2 / 3) < 1e-9
        assert abs(w(recent=2) / w0 - 0.5) < 1e-9   # half-point at X = 2
        db.set_setting("dj.novelty_plays_scale", 1.0)
        print("2. novelty halves at the plays scale, exactly")

        # 3. the novelty window forgets old plays (recovery, not a ratchet)
        recent_id = _add(db, "recent", play_count=9, listener_count=100)
        old_id = _add(db, "old", listener_count=100)
        _air(db, recent_id, age_h=1.0)
        _air(db, old_id, age_h=100.0)              # beyond the 48h window
        counts = dj.recent_play_counts(db, cfg)
        assert counts.get(recent_id) == 1 and old_id not in counts
        print("3. plays older than the window stop counting")

        # 4. starvation relaxation on its own station: five tracks, all
        #    aired (so strict windows are empty), airings inserted in
        #    chronological order like production play_log rows
        cfg3, tmp3 = isolated_config()
        try:
            db3 = DB(tmp3 / "infiniradio.db")
            a = _add(db3, "A", vote_up=200, vote_down=0, listener_count=50)
            b = _add(db3, "B", listener_count=100)
            c = _add(db3, "C")
            d = _add(db3, "D")
            e = _add(db3, "E")
            for tid, age in ((a, 5.0), (b, 4.5), (c, 0.5), (d, 0.4), (e, 0.3)):
                _air(db3, tid, age_h=age)
            assert dj.eligible_tracks(db3, cfg3) == []   # all in last-25 airs

            # the walk stops at the FIRST non-empty level: (3, 45m), whose
            # last-3-rows window excludes C/D/E and leaves the two rested
            # tracks -- A (the crowd favorite) and B
            relaxed = dj._relaxed_eligible(db3, cfg3, set())
            assert {t["id"] for t in relaxed} == {a, b}
            random.seed(20261002)
            picks = [dj.pick_next(db3, cfg3)["id"] for _ in range(40)]
            assert picks.count(a) >= 30, picks.count(a)
            # p(A) ~ 0.93 by weight; score-blind picking would land near
            # 20/40 (the old LRP fallback always took the oldest, too)
            print(f"4. starvation keeps scores: 'A' picked {picks.count(a)}/40"
                  f" of a relaxed {{{len(relaxed)}}}-track pool")
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        # 5. empty / held libraries still go silent, never crash
        cfg2, tmp2 = isolated_config()
        try:
            db2 = DB(tmp2 / "infiniradio.db")
            assert dj.pick_next(db2, cfg2) is None          # empty library
            _add(db2, "held", hold=1)
            assert dj.pick_next(db2, cfg2) is None          # only held tracks
            print("5. empty and held-only libraries -> None (silence)")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("test_pick_next: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
