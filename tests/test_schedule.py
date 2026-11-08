"""Schedule payload: pre-filled upcoming entry + PDT anchor correctness.

The player swaps the On Air card locally at the upcoming entry's
start_pdt_ms, so the payload must (a) include that entry whenever the
feeder has logged an airing beyond the listener's position, with the same
rich fields (lyrics/art) the live card renders, and (b) put the PDT
boundaries on ffmpeg's timeline — anchored at the encoder's FIRST WRITE
(pdt_origin), not process spawn — or every swap fires the spawn-to-first-
write gap early. The clamp rules (played_s) must apply to the upcoming
entry's end too, since a skip can cut the track currently being written.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone

import _helpers

_helpers.ensure_repo()

from infiniradio import library
from infiniradio.db import DB
from infiniradio.util import iso, utcnow
from infiniradio.web.routes_public import _schedule_payload

from _helpers import isolated_config

EPOCH = 7


class FakeEncoder:
    """Just the schedule-relevant surface of a live encoder."""

    def __init__(self, pdt_origin: datetime, spawn_gap_s: float = 1.0):
        self.epoch = EPOCH
        self.pdt_origin = pdt_origin
        self.started_wall = pdt_origin - timedelta(seconds=spawn_gap_s)

    def program_seconds(self) -> float:
        return 400.0


def _track(db, tid: int, title: str, duration_s: float) -> None:
    db.execute(
        "INSERT INTO tracks (id, title, original_path, playout_path, status,"
        " created_at, duration_s, lyrics_snapshot)"
        " VALUES (?, ?, ?, ?, 'ready', '2026-01-01T00:00:00', ?, ?)",
        (tid, title, f"{tid}/original.flac", f"{tid}/playout.m4a", duration_s,
         "[Verse]\nla la la\n"))


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")
        _track(db, 1, "First", 120.0)
        _track(db, 2, "Second", 90.0)

        origin = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc) \
            .replace(tzinfo=None)                    # naive UTC, like the encoder
        origin_ms = origin.replace(tzinfo=timezone.utc).timestamp() * 1000.0
        enc = FakeEncoder(origin, spawn_gap_s=1.0)

        def payload(at_s: float | None):
            at = None if at_s is None else origin_ms + at_s * 1000.0
            return _schedule_payload(db, cfg, "listener", at, enc)

        # nothing aired yet for this position -> clock self-heal (pre-existing)
        # air two tracks: first starts at prog 0, second already queued at 100
        # (airtimes are wall-recent so the clock-mode window check can hold)
        library.mark_aired(db, 1, iso(utcnow() - timedelta(minutes=3)),
                           prog_offset=0.0, epoch=EPOCH)
        library.mark_aired(db, 2, iso(utcnow() - timedelta(seconds=30)),
                           prog_offset=100.0, epoch=EPOCH)

        # mid-first-track: upcoming carries the queued second airing
        s = payload(50.0)
        assert s["sync"] == "program", s["sync"]
        assert s["current"]["id"] == 1
        assert s["current"]["start_pdt_ms"] == round(origin_ms), \
            "boundaries must anchor at pdt_origin (first write), not spawn"
        assert s["current"]["end_pdt_ms"] == round(origin_ms + 120_000)
        up = s["upcoming"]
        assert up and up["id"] == 2, up
        assert up["start_pdt_ms"] == round(origin_ms + 100_000)
        assert up["end_pdt_ms"] == round(origin_ms + 190_000)
        assert "lyrics" in up and "art" in up, "the card swap needs the rich fields"
        assert up["lyrics"], "upcoming lyrics ride along like the live entry's"
        print("mid-track: current + upcoming (rich fields) on the pdt_origin timeline")

        # mid-second-track: nothing queued beyond -> no upcoming, both in recent
        s = payload(150.0)
        assert s["current"]["id"] == 2
        assert s["upcoming"] is None
        assert [r["id"] for r in s["recent"]] == [2, 1]
        print("past the seam: upcoming empty, both airings recent")

        # clock mode (no at / no anchor): no timeline fields, no upcoming
        # (checked before the clamp below shrinks windows against old airtimes)
        s = payload(None)
        assert s["sync"] == "clock"
        assert s["current"]["id"] == 2
        assert s["current"]["start_pdt_ms"] is None
        assert s["upcoming"] is None
        print("clock mode: unchanged fallback (no timeline, no upcoming)")

        # a skip clamps the airing being written: its queued end shrinks too
        db.execute("UPDATE play_log SET played_s = 40.0 WHERE track_id = 2")
        s = payload(120.0)                      # inside the clamped 100..140 window
        assert s["current"]["end_pdt_ms"] == round(origin_ms + 140_000), \
            "played_s must clamp the live window (in-place cut)"
        print("skip clamp: played_s shrinks the live entry's end")

        # clamped track as the UPCOMING one (cut landed while queued ahead)
        s = payload(50.0)
        assert s["upcoming"]["end_pdt_ms"] == round(origin_ms + 140_000), \
            "the upcoming entry must obey the same clamp rule"
        print("skip clamp: upcoming entry obeys the same window rule")

        print("test_schedule: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
