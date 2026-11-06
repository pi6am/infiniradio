"""Banned lyric phrases end to end: the shared normalizer (search, phrase
analysis and the generation gate all match the same word sequences), the
workflow engine's banned gate (fresh re-roll on a match, feedback names the
phrase, exhaustion fails the run), the DJ's live-setting pass-through, the
/admin/lyrics view, and the admin bulk actions.
"""

from __future__ import annotations

import asyncio
import shutil

import _helpers

_helpers.ensure_repo()

from starlette.datastructures import FormData

from infiniradio import dj, phrases
from infiniradio.config import load_config
from infiniradio.db import DB
from infiniradio.llm.base import Reply
from infiniradio.llm.mock import SCRIPT
from infiniradio.lyrics import load_workflows, run_workflow
from infiniradio.web import routes_admin
from infiniradio.web.routes_admin import _library_view, _lyrics_view

from _helpers import RT, isolated_config

VOCAB = ["pop", "rock", "synthwave", "female vocal"]

# The canned songwriter reply and a variant carrying a banned line — the
# extract shape (heading + ```style/```lyrics fences) is untouched, so the
# ONLY thing failing the variant is the banned phrase.
_SONGWRITER = next(r for needle, r in SCRIPT if needle == "Write a song about")
_SONGWRITER_BAD = _SONGWRITER.replace("Nothing fancy, nothing cured",
                                      "I'm a functioning alcoholic")

_TRANSLATE = next(r for needle, r in SCRIPT if needle == "Translate the following song")
_TRANSLATE_BAD = _TRANSLATE.replace("In a quiet room I unfold a map",
                                    "I'm a functioning alcoholic in the quiet room")


class _ScriptedLLM:
    """Replies from a queue, one per chat() call; records every prompt."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []

    async def chat(self, messages, *, model=None, temperature=None,
                   max_tokens=None, extra=None):
        self.calls += 1
        self.prompts.append(messages[-1].content)
        return Reply(content=self.replies.pop(0), model=model or "scripted",
                     usage={"prompt_tokens": 10, "completion_tokens": 10})


def _canned(needle: str) -> str:
    return next(r for n, r in SCRIPT if n == needle)


class _Req:
    """Minimal Request stand-in for route handlers (test_jobs pattern)."""

    def __init__(self, rt, data: FormData, referer="/admin/library"):
        app = type("A", (), {})()
        app.state = type("S", (), {})()
        app.state.rt = rt
        self.app = app
        self.cookies = {}
        self.headers = {"referer": referer}
        self._form = data

    async def form(self):
        return self._form


def _add(db, title: str, **kw) -> int:
    cols = {"title": title, "status": "ready", "created_at": "2026-01-01T00:00:00", **kw}
    return db.execute(
        f"INSERT INTO tracks ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        tuple(cols.values()))


async def _main(cfg) -> None:
    specs = load_workflows(load_config())
    sw = specs["songwriter"]
    tf = specs["title-first"]

    # 1. normalizer: headers dropped, case/punctuation folded, curly
    #    apostrophes straightened; whole-word-phrase matching
    assert phrases.normalize("[Verse 1]\nI'm a functioning alcoholic,") == \
        "i'm a functioning alcoholic"
    assert phrases.normalize("don\u2019t \u2014 STOP!") == "don't stop"
    assert phrases.contains_phrase(
        "She said I'm a functioning-alcoholic mess", "functioning alcoholic")
    assert not phrases.contains_phrase(  # word boundaries: no substring hits
        "we are functioning alcoholics anonymous", "functioning alcoholic")
    assert not phrases.contains_phrase("anything", "")   # empty phrase never matches
    print("1. normalize + whole-word-phrase matching")

    # 2. phrase analysis: EXCLUSIVE counting — an occurrence inside a
    #    listed longer phrase belongs to the longer form
    rows = phrases.top_phrases([
        "[verse 1]\nI'm a functioning alcoholic\nliving in a dream",
        "[chorus]\nShe said I'm a functioning alcoholic\ntonight we drive",
        "nothing shared here at all",
    ])
    assert rows == [{"phrase": "i'm a functioning alcoholic",
                     "songs": 2, "pct": 66.7, "total": 2}], rows
    # every "i was born" occurrence extends "i was born in": nothing left
    # for the fragment, only the maximal form is listed
    rows = phrases.top_phrases([
        "i was born in a town",
        "i was born in the middle of the city",
        "i was dancing alone",
    ])
    assert rows == [{"phrase": "i was born in", "songs": 2,
                     "pct": 66.7, "total": 2}], rows
    # ...but a shorter phrase whose occurrences do NOT all extend keeps
    # its row with the exclusive count (the 1000/100 case, scaled)
    rows = phrases.top_phrases([
        "i was a functioning mess indeed",
        "i was a functioning wreck indeed",
        "i was a functioning joke indeed",
        "i was a functioning alcoholic tonight",
        "i was a functioning alcoholic forever",
    ])
    assert rows == [{"phrase": "i was a functioning", "songs": 3,
                     "pct": 60.0, "total": 3},
                    {"phrase": "i was a functioning alcoholic", "songs": 2,
                     "pct": 40.0, "total": 2}], rows
    # no length cap: a memorized 14-word line surfaces ONCE at its full
    # shared length, never as sliding 8-word windows
    line = "i was supposed to be sent away but they forgot to come and get me"
    rows = phrases.top_phrases([
        f"[verse]\n{line}\nthen more words follow here",
        f"[chorus]\n{line}\nother tail words entirely",
        "completely unrelated words only",
    ])
    assert rows == [{"phrase": line, "songs": 2, "pct": 66.7, "total": 2}], rows
    # only DISPLAYED phrases absorb: with a one-row table, the longer
    # variant is cut and the visible row keeps its full count — the
    # number the library search reports when you click it
    rows = phrases.top_phrases([
        "i was supposed to leave tonight",
        "i was supposed to leave forever",
        "i was supposed to dance instead",
        "i was supposed to sing along",
        "i was supposed to run away",
    ], limit=1)
    assert rows == [{"phrase": "i was supposed to", "songs": 5,
                     "pct": 100.0, "total": 5}], rows
    print("2. top_phrases: exclusive counts, unbounded length")

    # 3. engine gate, songwriter shape (extracted lyrics, fresh re-rolls):
    #    a banned phrase fails attempt 1 (and only that), the clean re-roll
    #    passes — banned phrase casing is normalized away
    mock = _ScriptedLLM([_SONGWRITER_BAD, _SONGWRITER])
    r = await run_workflow(mock, sw, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB, banned=["Functioning Alcoholic"])
    assert r.ok, r.error
    assert mock.calls == 2
    fails = r.attempts[0].failures
    assert any(f.code == "forbidden_present" and f.check == "banned"
               and "Functioning Alcoholic" in f.detail for f in fails), fails
    assert "functioning alcoholic" not in r.lyrics.lower()
    print("3. songwriter: banned attempt re-rolled clean (2 calls)")

    # 4. exhaustion: every attempt banned -> run fails with the code
    mock = _ScriptedLLM([_SONGWRITER_BAD] * 3)
    r = await run_workflow(mock, sw, initial_vars={"title": "T"},
                           tag_vocabulary=VOCAB, banned=["functioning alcoholic"])
    assert not r.ok and mock.calls == 3
    assert "forbidden_present" in r.error and "banned" in r.error, r.error
    print("4. songwriter: exhaustion fails the run with forbidden_present")

    # 5. title-first shape (song-block stages + feedback retries): the
    #    banned phrase in the translate stage fails, the feedback retry
    #    NAMES the phrase (failure details render into the prompt), and the
    #    whole pipeline still completes
    mock = _ScriptedLLM([
        _canned("Brainstorm ideas"),
        _canned("Use this guide to write a song"),
        _TRANSLATE_BAD, _TRANSLATE,                    # translate: bad, then feedback
        _canned("English name"),
        _canned("Write a Style Prompt"),
        _canned("Rewrite the words of this song"),
        _canned("tags"),
    ])
    r = await run_workflow(mock, tf, initial_vars={"title": "A Window Far Away"},
                           tag_vocabulary=VOCAB,
                           banned=["functioning alcoholic"])
    assert r.ok, r.error
    assert "functioning alcoholic" in mock.prompts[3].lower()  # the feedback turn
    assert "functioning alcoholic" not in r.lyrics.lower()
    print("5. title-first: feedback retry names the banned phrase")

    # 6. dj pass-through: the live setting reaches the engine; attempts are
    #    logged to job_stage_log with the banned failure
    cfg2, tmp2 = isolated_config()
    try:
        db = DB(tmp2 / "infiniradio.db")
        db.set_setting("banned_lyric_phrases", ["functioning alcoholic"])
        job_id = dj.enqueue_job(db, workflow_id="songwriter")
        job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
        mock = _ScriptedLLM([_SONGWRITER_BAD, _SONGWRITER])
        rt = RT(cfg2, db, None, mock, {"songwriter": sw})
        await dj._run_lyrics_phase(rt, job)
        row = db.one("SELECT status, phase FROM generation_jobs WHERE id = ?", (job_id,))
        assert row["status"] == "queued" and row["phase"] == "render", row
        assert mock.calls == 2
        logged = db.query("SELECT gate_json FROM job_stage_log WHERE job_id = ?"
                          " ORDER BY id", (job_id,))
        assert len(logged) == 2 and "forbidden_present" in logged[0]["gate_json"]
        print("6. dj lyrics phase: live setting passed through, attempts logged")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        asyncio.run(_main(cfg))

        # 7. /admin/lyrics view: vocal corpus counts, placeholder excluded,
        #    banned list round-trips through the setting
        db = DB(tmp / "infiniradio.db")
        _add(db, "v1", lyrics_snapshot="[verse 1]\nI'm a functioning alcoholic\nhere")
        _add(db, "v2", lyrics_snapshot="[chorus]\nI'm a functioning alcoholic\ntagain")
        _add(db, "v3", lyrics_snapshot="[Intro]\n\n[Instrumental]\n\n[Outro]")
        _add(db, "gone", status="deleted",
             lyrics_snapshot="[verse]\nI'm a functioning alcoholic\ndeleted")
        v = _lyrics_view(db)
        assert v["vocal_count"] == 2          # placeholder + deleted excluded
        assert v["rows"][0]["phrase"] == "i'm a functioning alcoholic"
        assert v["banned"] == [] and v["banned_text"] == ""
        print("7. lyrics view: corpus counts, placeholder/deleted excluded")

        # 8. library q-filter: normalized whole-word matching inside the tab,
        #    q rides along in every URL
        view = _library_view(db, "ready", "id", "desc", 1, q="Functioning Alcoholic")
        assert [t["title"] for t in view["tracks"]] == ["v2", "v1"]
        assert "q=Functioning+Alcoholic" in view["columns"][0]["url"]
        assert "q=" in view["keep"] and view["pager"]["total"] == 2
        assert _library_view(db, "ready", "id", "desc", 1, q="nothing matches")["tracks"] == []
        print("8. library q filter: match, URL state, empty result")

        # 9. bulk actions via the route handler: delete trashes through the
        #    buffer and purges the queue; hold/protect set flags; junk input
        #    is rejected or skipped
        real_admin, real_csrf = routes_admin._is_admin, routes_admin._csrf_ok
        routes_admin._is_admin = lambda r: True
        routes_admin._csrf_ok = lambda r, s: True
        try:
            a, b, c = _add(db, "del-a"), _add(db, "del-b"), _add(db, "flag-c")
            _add(db, "held")  # untouched control
            db.execute("INSERT INTO play_queue (track_id, queued_at) VALUES (?, ?)",
                       (a, "2026-01-01T00:00:00"))
            rt = RT(cfg, db, None, None, {})
            req = _Req(rt, FormData([("action", "delete"), ("csrf", "x"),
                                     ("track_ids", str(a)), ("track_ids", str(b)),
                                     ("track_ids", "junk"), ("track_ids", "99999")]))
            resp = asyncio.run(routes_admin.library_bulk(req))
            assert resp.status_code == 303
            assert db.one("SELECT status FROM tracks WHERE id = ?", (a,))["status"] == "trashed"
            assert db.one("SELECT delete_after FROM tracks WHERE id = ?", (b,))["delete_after"]
            assert db.one("SELECT COUNT(*) AS n FROM play_queue WHERE track_id = ?", (a,))["n"] == 0

            req = _Req(rt, FormData([("action", "hold"), ("csrf", "x"),
                                     ("track_ids", str(c))]))
            asyncio.run(routes_admin.library_bulk(req))
            assert db.one("SELECT hold FROM tracks WHERE id = ?", (c,))["hold"] == 1
            req = _Req(rt, FormData([("action", "unprotect"), ("csrf", "x"),
                                     ("track_ids", str(c))]))
            asyncio.run(routes_admin.library_bulk(req))
            assert db.one("SELECT protected FROM tracks WHERE id = ?", (c,))["protected"] == 0

            bad = _Req(rt, FormData([("action", "launch"), ("csrf", "x"),
                                     ("track_ids", str(c))]))
            assert asyncio.run(routes_admin.library_bulk(bad)).status_code == 400
            none = _Req(rt, FormData([("action", "delete"), ("csrf", "x")]))
            assert asyncio.run(routes_admin.library_bulk(none)).status_code == 400
            print("9. bulk actions: delete/hold/flags, junk rejected")

            # 10. banned-list save: newline list -> JSON setting (deduped,
            #     trimmed); the lyrics view reflects it
            req = _Req(rt, FormData([("csrf", "x"),
                                     ("phrases", "  functioning alcoholic \n\nmelt it\n"
                                                 "melt it\n")]),
                       referer="/admin/lyrics")
            resp = asyncio.run(routes_admin.lyrics_banned_save(req))
            assert resp.status_code == 303
            assert db.get_setting("banned_lyric_phrases") == \
                ["functioning alcoholic", "melt it"]
            assert _lyrics_view(db)["banned"] == ["functioning alcoholic", "melt it"]
            print("10. banned-list save: trimmed, deduped, live in the view")
        finally:
            routes_admin._is_admin, routes_admin._csrf_ok = real_admin, real_csrf

        print("test_lyrics_banned: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
