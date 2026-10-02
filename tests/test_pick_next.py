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
            relaxed, win_t, win_h = dj._relaxed_eligible(db3, cfg3, set())
            assert {t["id"] for t in relaxed} == {a, b}
            assert (win_t, win_h) == (3, 0.75)       # the stopping level
            random.seed(20261002)
            picks = [dj.pick_next(db3, cfg3)["id"] for _ in range(40)]
            assert picks.count(a) >= 30, picks.count(a)
            # p(A) ~ 0.93 by weight; score-blind picking would land near
            # 20/40 (the old LRP fallback always took the oldest, too)
            print(f"4. starvation keeps scores: 'A' picked {picks.count(a)}/40"
                  f" of a relaxed {{{len(relaxed)}}}-track pool")

            # 6. the debug distribution mirrors the picker exactly:
            # probabilities sum to 1, the window level is reported, and
            # the fresh-share mixture matches the two-branch draw
            dist = dj.pick_distribution(db3, cfg3, set())
            assert not dist["strict"] and dist["windows"] == (3, 0.75)
            assert abs(sum(e["prob"] for e in dist["entries"]) - 1.0) < 1e-9
            # no fresh tracks here: p = w / sum(w), no share multiplier
            ws = {e["id"]: e["weight"] for e in dist["entries"]}
            tot = sum(ws.values())
            for e in dist["entries"]:
                assert abs(e["prob"] - ws[e["id"]] / tot) < 1e-12
            print("5. distribution without fresh: p = w/sum(w), level reported")
            cfg4, tmp4 = isolated_config()
            try:
                db4 = DB(tmp4 / "infiniradio.db")
                for i in range(5):
                    _add(db4, f"n{i}")
                # all-fresh library: the uniform branch always fires
                dist = dj.pick_distribution(db4, cfg4, set())
                assert dist["strict"]                     # never aired: windows moot
                assert all(abs(e["prob"] - 0.2) < 1e-9 for e in dist["entries"])

                # mixed: one seasoned track strictly eligible (aired 40h ago,
                # outside the last-2 rows) beside the five fresh ones —
                # fresh gets share/n_fresh + (1-share)*w/tot, seasoned
                # gets (1-share)*w/tot
                db4.set_setting("dj.repeat_window_tracks", 2)
                s1 = _add(db4, "s1", play_count=1)
                s2 = _add(db4, "s2", play_count=1, listener_count=8)
                s3 = _add(db4, "s3", play_count=1)
                for tid, age in ((s2, 40.0), (s1, 30.0), (s3, 20.0)):
                    _air(db4, tid, age_h=age)     # chronological row order
                dist = dj.pick_distribution(db4, cfg4, set())
                assert dist["fresh_n"] == 5 and dist["share"] == 0.35
                assert {e["id"] for e in dist["entries"]} == \
                    {s2} | {db4.one("SELECT id FROM tracks WHERE title = ?", (f"n{i}",))["id"] for i in range(5)}
                ws = {e["id"]: e["weight"] for e in dist["entries"]}
                tot = sum(ws.values())
                fe = next(e for e in dist["entries"] if e["title"] == "n0")
                se = next(e for e in dist["entries"] if e["id"] == s2)
                assert abs(fe["prob"] - (0.35 / 5 + 0.65 * ws[fe["id"]] / tot)) < 1e-12
                assert abs(se["prob"] - 0.65 * ws[s2] / tot) < 1e-12
                assert abs(sum(e["prob"] for e in dist["entries"]) - 1.0) < 1e-9
                print("6. mixed distribution: exact fresh-share mixture")
            finally:
                shutil.rmtree(tmp4, ignore_errors=True)
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)

        # 7. empty / held libraries still go silent, never crash
        cfg2, tmp2 = isolated_config()
        try:
            db2 = DB(tmp2 / "infiniradio.db")
            assert dj.pick_next(db2, cfg2) is None          # empty library
            _add(db2, "held", hold=1)
            assert dj.pick_next(db2, cfg2) is None          # only held tracks
            print("7. empty and held-only libraries -> None (silence)")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("test_pick_next: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
