"""Styles: seed import (prompts lifecycle), definition TOML round-trip,
validation (including the weight-0/empty-value editing idiom), seeded
resolution and weight honoring, commission-time delivery of overrides into
workflow templates, bandit stats, and the live workflow enable/disable +
lyrics-share semantics.
"""

from __future__ import annotations

import asyncio
import json
import shutil

import _helpers

_helpers.ensure_repo()

from infiniradio import dj, library
from infiniradio.db import DB
from infiniradio.generator.mock import MockGenerator
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import load_workflows
from infiniradio.styles import (StyleError, parse_definition, render_definition,
                                resolve)

from _helpers import RT, isolated_config


async def _render_phase(db, cfg, rt, jid):
    await dj._run_lyrics_phase(rt, db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid,)))


async def _main(cfg, rt) -> None:
    db = rt.db
    library.import_prompts(db, cfg)
    library.seed_tags(db, cfg)
    n = library.import_styles(db, cfg)
    assert n == 6, n
    assert library.import_styles(db, cfg) == 0  # only_new: re-import is a no-op
    assert len(db.query("SELECT * FROM styles")) == 6
    assert db.one("SELECT COUNT(*) AS n FROM style_tags")["n"] >= 6
    print(f"1. imported {n} styles (+ tags linked); re-import is a no-op")

    # definition round-trip: DB row -> TOML -> parse -> same values
    row = db.one("SELECT * FROM styles WHERE name = 'jpop-anime'")
    text = render_definition(json.loads(row["vars_json"]),
                             {k: [tuple(e) for e in v]
                              for k, v in json.loads(row["weighted_json"]).items()})
    v2, w2 = parse_definition(text, "round-trip")
    assert v2 == json.loads(row["vars_json"])
    assert {k: [list(e) for e in entries] for k, entries in w2.items()} == \
        json.loads(row["weighted_json"])
    print("2. definition TOML round-trips (admin form <-> DB)")

    # validation rejects bad definitions; the editing idiom is supported
    for bad in [
        '[vars]\nTitle = "no"',                              # reserved/case
        '[vars]\nstyle_guide = ""',                          # empty vars value
        '[weighted]\nx = [{ weight = 1 }]',                  # missing value
        '[vars]\nx = "a"\n[weighted]\nx = [{ weight = 1, value = "b" }]',  # collision
        '[weighted]\nx = []',                                # no entries
        '[weighted]\nx = [{ weight = -1, value = "y" }]',    # negative weight
    ]:
        try:
            parse_definition(bad, "bad")
            raise AssertionError(f"accepted: {bad!r}")
        except StyleError:
            pass
    v, w = parse_definition(
        '[weighted]\ng = [{ weight = 0, value = "off" }, { weight = 2, value = "" },'
        ' { weight = 1, value = "on" }]', "idiom")
    assert resolve({}, w, 7) == {"g": "on"}
    v2, w2 = parse_definition('[weighted]\ng = [{ weight = 0, value = "a" }]', "idiom")
    assert resolve({}, w2, 7) == {}  # all disabled -> default applies
    print("3. definition validation ok (weight-0/empty = disabled entries)")

    # commission: enqueue resolves overrides deterministically from the seed
    jid = dj.enqueue_job(db, workflow_id="jp-title-first",
                         style_id=row["id"], seed=1234)
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid,))
    assert job["style_id"] == row["id"] and job["seed"] == 1234
    ov = json.loads(job["progress"])["overrides"]
    expect = resolve(json.loads(row["vars_json"]),
                     {k: [tuple(e) for e in v]
                      for k, v in json.loads(row["weighted_json"]).items()}, 1234)
    assert ov == expect and "style_guide" in ov and "structure_guide" in ov
    assert db.one("SELECT times_used FROM styles WHERE id = ?",
                  (row["id"],))["times_used"] == 1
    dist = {resolve({}, {k: [tuple(e) for e in v]
                         for k, v in json.loads(row["weighted_json"]).items()},
                    s)["structure_guide"]
            for s in range(200)}
    assert len(dist) == 3  # all three structures reachable
    print("4. enqueue resolves overrides from the job seed; all options reachable")

    # weighted choice respects the weights
    picks = [resolve({}, {"g": [(3, "a"), (1, "b")]}, s)["g"] for s in range(4000)]
    ratio = picks.count("a") / len(picks)
    assert 0.70 < ratio < 0.80, ratio
    print(f"5. weights honored: 3:1 -> {ratio:.2f} 'a' over 4000 draws")

    # the lyrics phase renders the overrides into the stage prompts, and
    # they survive every progress write
    await _render_phase(db, cfg, rt, jid)
    traces = db.query("SELECT * FROM job_stage_log WHERE job_id = ? ORDER BY id", (jid,))
    assert "energetic J-pop-style" in traces[0]["prompt_text"]
    assert "energetic J-pop-style" in traces[1]["prompt_text"]
    assert ov["structure_guide"][:40] in traces[1]["prompt_text"]
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid,))
    assert job["phase"] == "render"
    final = json.loads(job["progress"])
    assert final["theme_title"] and final["overrides"] == ov
    print("6. overrides rendered into prompts; survived every progress write")

    # track lineage + air stats feed the bandit
    req = dj.build_request(db, job)
    track_id = library.create_track(
        db, cfg, prompt_id=None, parent_track_id=None, workflow_id=job["workflow_id"],
        style_id=job["style_id"], title=final["outputs"]["title"],
        theme_title=final["outputs"]["theme_title"], style_text=req.style,
        lyrics_snapshot=req.lyrics, seed=req.seed, cot_mode="full")
    db.execute("UPDATE tracks SET status = 'ready', play_count = 1,"
               " vote_up = 8, vote_down = 0 WHERE id = ?", (track_id,))
    assert dj.style_air_stats(db, job["style_id"])["aired"] == 1
    other = db.one("SELECT id FROM styles WHERE name = 'folk-quiet'")["id"]
    assert dj.style_weight(db, cfg, job["style_id"]) > dj.style_weight(db, cfg, other)
    assert dj.pick_style(rt) is not None
    print("7. track lineage + style air stats: upvoted style outweighs unaired")

    # styled workflows + share semantics
    spec = rt.workflows["jp-title-first"]
    assert spec.styled and "rhyme_guide" in spec.vars
    assert dj.pick_generator(rt) is not None
    assert len(dj.enabled_workflows(rt)) == 2
    db.set_setting("workflows_disabled", ["zh-title-first"])
    assert [w.name for w in dj.enabled_workflows(rt)] == ["jp-title-first"]
    db.set_setting("dj.lyrics_share", 1.0)
    assert {dj.pick_generator(rt)[0] for _ in range(30)} == {"workflow"}
    db.set_setting("dj.lyrics_share", 0.0)
    assert {dj.pick_generator(rt)[0] for _ in range(30)} == {"prompt"}
    db.set_setting("dj.lyrics_share", 0.5)
    assert {dj.pick_generator(rt)[0] for _ in range(60)} == {"prompt", "workflow"}
    db.set_setting("workflows_disabled", ["jp-title-first", "zh-title-first"])
    db.set_setting("dj.lyrics_share", 1.0)
    assert dj.pick_generator(rt)[0] == "prompt"  # empty pool falls through
    db.set_setting("workflows_disabled", [])
    db.execute("DELETE FROM settings WHERE key = 'dj.lyrics_share'")
    print("8. live enable/disable + lyrics_share semantics ok")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        # Regression: allow_new_tags = false must not block the SEEDERS —
        # a fresh station with it set used to end up with an empty (and
        # therefore broken) tag vocabulary.
        object.__setattr__(cfg, "_d", {**cfg._d, "lyrics":
                                       {**cfg._d["lyrics"], "allow_new_tags": False}})
        db = DB(tmp / "infiniradio.db")
        library.import_prompts(db, cfg)
        library.seed_tags(db, cfg)
        assert len(library.tag_vocabulary(db)) >= 30, library.tag_vocabulary(db)
        assert {"chiptune", "ambient"} <= {t for p in db.query("SELECT * FROM prompts")
                                           for t in library.prompt_tags(db, p["id"])}
        # runtime picks stay gated: an unknown tag is NOT created on ingest
        assert library.apply_track_tags(db, cfg, 1, ["brand-new-genre"]) == []
        assert db.one("SELECT COUNT(*) AS n FROM tags WHERE name = 'brand-new-genre'"
                      )["n"] == 0
        print("0. allow_new_tags=false seeds the vocabulary, gates only runtime picks")

        rt = RT(cfg, db, MockGenerator(cfg), MockLLMClient(), load_workflows(cfg))
        asyncio.run(_main(cfg, rt))
        print("test_styles: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
