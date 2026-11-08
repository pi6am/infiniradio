"""Lyrics on the public player: section headers stripped from the display
text, instrumental placeholders exposed as None (the has-authored-lyrics
test), and lyrics attached to the current schedule entry + /api/track only.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import timedelta
from pathlib import Path

import _helpers

_helpers.ensure_repo()

from infiniradio import library
from infiniradio.db import DB
from infiniradio.llm.images import MOCK_PNG
from infiniradio.util import iso, utcnow
from infiniradio.web import routes_public
from _helpers import RT


PLACEHOLDER = "[Intro]\n\n[Instrumental]\n\n[Outro]"
VOCAL = "[verse 1]\nline one\n\n[Chorus]\nline two\n\n[VERSE 2]\nline three\n"
VOCAL_CLEAN = "line one\n\nline two\n\nline three"


def _seed(db: DB) -> tuple[int, int]:
    """A vocal track currently on air and an instrumental that just aired."""
    now = utcnow()
    voc = db.execute(
        "INSERT INTO tracks (title, lyrics_snapshot, duration_s, status, created_at)"
        " VALUES ('vocal song', ?, 180, 'ready', ?)",
        (VOCAL, iso(now - timedelta(hours=2))))
    ins = db.execute(
        "INSERT INTO tracks (title, lyrics_snapshot, duration_s, status, created_at)"
        " VALUES ('instrumental', ?, 180, 'ready', ?)",
        (PLACEHOLDER, iso(now - timedelta(hours=2, seconds=1))))
    db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
               (voc, iso(now - timedelta(seconds=60))))    # still on air
    db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
               (ins, iso(now - timedelta(seconds=600))))   # finished
    return voc, ins


def run() -> None:
    cfg, tmp = _helpers.isolated_config()
    try:
        # 1. lyrics_public: instrumental placeholders vanish — LF, CRLF, and
        #    the malformed doubled-[Instrumental] variant seen in real data
        assert routes_public.lyrics_public(PLACEHOLDER) is None
        assert routes_public.lyrics_public(
            "[Intro]\r\n\r\n[Instrumental]\r\n\r\n[Outro]") is None
        assert routes_public.lyrics_public(
            "[Intro]\n\n[Instrumental]\n\n[Instrumental]\n\n[Outro]") is None
        assert routes_public.lyrics_public("") is None
        assert routes_public.lyrics_public(None) is None
        print("1. instrumental placeholders -> None")

        # 2. lyrics_public: headers stripped in any casing/numbering, lyric
        #    text intact; mid-line brackets are lyrics, not headers
        assert routes_public.lyrics_public(VOCAL) == VOCAL_CLEAN
        assert routes_public.lyrics_public(
            "said [softly] to me") == "said [softly] to me"
        assert routes_public.lyrics_public("[v]\na\n\n\n\nb") == "a\n\nb"
        print("2. section headers stripped, text kept")

        # 3. schedule payload: lyrics on the current entry only
        db = DB(Path(tmp) / "infiniradio.db")
        voc, ins = _seed(db)
        payload = routes_public._schedule_payload(db, cfg, "listener-1", None, None)
        assert payload["current"]["id"] == voc
        assert payload["current"]["lyrics"] == VOCAL_CLEAN
        assert {r["id"] for r in payload["recent"]} == {voc, ins}
        assert all("lyrics" not in r for r in payload["recent"]), \
            "recent rows must stay lean"
        print("3. /api/schedule: lyrics on current entry only")

        # 4. instrumental takes over -> lyrics None (player hides the panel)
        db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
                   (ins, iso(utcnow() - timedelta(seconds=30))))
        payload = routes_public._schedule_payload(db, cfg, "listener-1", None, None)
        assert payload["current"]["id"] == ins
        assert payload["current"]["lyrics"] is None
        print("4. instrumental on air -> lyrics None")

        # 5. /api/track/{id} carries cleaned lyrics per track
        rt = RT(cfg, db, None, None, {})
        fake_app = type("A", (), {})()
        fake_app.state = type("S", (), {})()
        fake_app.state.rt = rt
        req = type("R", (), {"cookies": {}, "app": fake_app})()
        v = json.loads(asyncio.run(routes_public.api_track(req, voc)).body)
        i = json.loads(asyncio.run(routes_public.api_track(req, ins)).body)
        assert v["lyrics"] == VOCAL_CLEAN
        assert i["lyrics"] is None
        print("5. /api/track: cleaned lyrics per track")

        # 5b. /api/track/{id} also carries the current art pick's links
        # (None when the track has none). Files are written directly —
        # art_file_map reads disk, so no ffmpeg is needed here.
        assert v["art"] is None
        aid = db.execute(
            "INSERT INTO track_art (track_id, workflow_id, created_at)"
            " VALUES (?, 'cover', ?)", (voc, iso(utcnow())))
        adir = library.art_dir(cfg, voc)
        adir.mkdir(parents=True, exist_ok=True)
        (adir / f"{aid}.png").write_bytes(MOCK_PNG)
        (adir / f"{aid}_512.jpg").write_bytes(b"fake-thumb")
        db.execute("UPDATE tracks SET art_id = ? WHERE id = ?", (aid, voc))
        v2 = json.loads(asyncio.run(routes_public.api_track(req, voc)).body)
        assert v2["art"]["master"] == f"/art/{voc}/{aid}/full.png"
        assert v2["art"]["card"] == f"/art/{voc}/{aid}/512.jpg"
        assert v2["art"]["card_px"] == 512
        print("5b. /api/track: art links for the current pick")

        print("test_lyrics_public: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
