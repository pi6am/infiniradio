"""Track art pipeline: workflow loading + weighted picks, the two job
phases (LLM image-prompt workflow -> images render), storage/versioning/
thumbnails, GPU alternation with music renders, auto-enqueue for new
tracks, backfill budgeting, and the versioned /art URL builder.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

import _helpers

_helpers.ensure_repo()

from infiniradio import art, dj, library
from infiniradio.config import runtime_get
from infiniradio.db import DB
from infiniradio.llm.images import MOCK_PNG, MockImagesClient
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import WorkflowError
from infiniradio.web import routes_public

from _helpers import RT, isolated_config


def _make_track(db, cfg, title="Glass Cathedral", lyrics="[Verse]\nNeon rain\n"):
    track_id = library.create_track(
        db, cfg, prompt_id=None, parent_track_id=None, style_text="dreamy synthwave",
        lyrics_snapshot=lyrics, seed=7, cot_mode="full", title=title)
    db.execute("UPDATE tracks SET status = 'ready', duration_s = 120 WHERE id = ?",
               (track_id,))
    return track_id


def _mkrt(cfg, db, llm=None):
    rt = RT(cfg, db, None, llm or MockLLMClient(), {})
    rt.images = MockImagesClient()
    rt.artworkflows = art.load_art_workflows(cfg)
    db.set_setting("art.enabled", True)
    return rt


async def _main(cfg, tmp) -> None:
    db = DB(cfg.path("library.data_dir") / "infiniradio.db")

    # ---- loading ----------------------------------------------------------
    wf = art.load_art_workflows(cfg)
    assert set(wf) == {"cover", "cover-illustrated", "cover-photographic", "lyric-scene"}
    assert wf["cover"].abstract and not wf["cover-illustrated"].abstract
    # inheritance resolved: children carry the base's two stages
    assert [s.id for s in wf["cover-illustrated"].stages] == ["concept", "prompt"]
    assert wf["cover-illustrated"].weight == 2.0
    # the duck-typed output contract maps onto the shared engine
    assert wf["cover"].output.style == "image_prompt" and wf["cover"].output.lyrics is None
    print(f"loaded {len(wf)} art workflows (inheritance + weights + output contract)")

    # a broken workflow dir is a startup error, never a silent skip
    bad = tmp / "bad_art_workflows" / "broken"
    bad.mkdir(parents=True)
    (bad / "workflow.toml").write_text('name = "broken"\nno_such_key = 1\n')
    object.__setattr__(cfg, "_d", {**cfg._d, "art": {**cfg._d["art"],
                                    "workflows_dir": str(tmp / "bad_art_workflows")}})
    try:
        art.load_art_workflows(cfg)
        raise AssertionError("broken art workflow did not fail validation")
    except WorkflowError:
        pass
    object.__setattr__(cfg, "_d", {**cfg._d, "art": {**cfg._d["art"],
                                    "workflows_dir": "art_workflows"}})
    print("broken workflow dir rejected at load")

    rt = _mkrt(cfg, db)

    # ---- weighted picks + requires filter -----------------------------------
    db.set_setting("art_workflow_weights", {"cover-illustrated": 0.0,
                                            "cover-photographic": 0.0})
    db.set_setting("art_workflows_disabled", [])
    lyric_track = {"lyrics_snapshot": "[Verse]\nNeon rain on empty streets\n"}
    instr_track = {"lyrics_snapshot": "[Intro]\n\n[Instrumental]\n\n[Outro]"}
    assert wf["lyric-scene"].requires == {"lyrics": True}
    assert art.workflow_applies(wf["lyric-scene"], lyric_track)
    assert not art.workflow_applies(wf["lyric-scene"], instr_track)
    assert art.workflow_applies(wf["cover-illustrated"], instr_track)  # no predicate
    for _ in range(20):
        assert art.pick_art_workflow(rt, lyric_track).name == "lyric-scene"
    # an instrumental track filters lyric-scene out entirely — the pick
    # falls to the applicable workflows even with their weights zeroed
    for _ in range(20):
        assert art.pick_art_workflow(rt, instr_track).name in \
            ("cover-illustrated", "cover-photographic")
    # the override retunes, overriding the file weight (file says 2.0)
    db.set_setting("art_workflow_weights", {"cover-illustrated": 5.0})
    assert art.effective_weight(db, rt.artworkflows["cover-illustrated"]) == 5.0
    db.set_setting("art_workflow_weights", {})
    # disabled via the admin list
    db.set_setting("art_workflows_disabled", ["lyric-scene"])
    assert {w.name for w in art.enabled_art_workflows(rt)} == \
        {"cover-illustrated", "cover-photographic"}
    db.set_setting("art_workflows_disabled", [])
    print("picks filter [requires] first, then weight; overrides and the "
          "disabled list honored")

    # unknown [requires] keys are startup errors (the names/ rule)
    bad2 = tmp / "bad_art_workflows" / "badreq"
    bad2.mkdir(parents=True)
    (bad2 / "workflow.toml").write_text(
        'name = "badreq"\n[requires]\nnope = true\n')
    object.__setattr__(cfg, "_d", {**cfg._d, "art": {**cfg._d["art"],
                                    "workflows_dir": str(tmp / "bad_art_workflows")}})
    try:
        art.load_art_workflows(cfg)
        raise AssertionError("unknown [requires] key did not fail validation")
    except WorkflowError as exc:
        assert "requires" in str(exc), exc   # badreq sorts before broken
    object.__setattr__(cfg, "_d", {**cfg._d, "art": {**cfg._d["art"],
                                    "workflows_dir": "art_workflows"}})
    print("unknown [requires] key rejected at load")

    # ---- enqueue -----------------------------------------------------------
    t1 = _make_track(db, cfg)
    # explicit workflow: the weighted pick below must stay deterministic here
    job_id = art.enqueue_art_job(rt, t1, workflow_id="cover-illustrated")
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["kind"] == "art" and job["phase"] == "art" and job["status"] == "queued"
    assert job["track_id"] == t1 and job["workflow_id"] in {
        "cover-illustrated", "cover-photographic", "lyric-scene"}
    # idempotent while queued
    assert art.enqueue_art_job(rt, t1) == job_id
    # non-ready tracks are rejected
    db.execute("UPDATE tracks SET status = 'failed' WHERE id = ?", (t1,))
    t2 = _make_track(db, cfg, title="Second")
    assert art.enqueue_art_job(rt, t2, workflow_id="nope") is None
    db.execute("UPDATE tracks SET status = 'ready' WHERE id = ?", (t1,))
    # the auto pick respects [requires]: an instrumental track never gets
    # lyric-scene; an explicit admin pick bypasses the filter
    instr = _make_track(db, cfg, title="Instrumental",
                        lyrics="[Intro]\n\n[Instrumental]\n\n[Outro]")
    for _ in range(10):
        j = art.enqueue_art_job(rt, instr)
        assert j is not None and db.one(
            "SELECT workflow_id FROM generation_jobs WHERE id = ?", (j,)
        )["workflow_id"] != "lyric-scene"
        db.execute("DELETE FROM generation_jobs WHERE id = ?", (j,))
    forced = art.enqueue_art_job(rt, instr, workflow_id="lyric-scene")
    assert forced is not None   # explicit = the admin's call to make
    db.execute("DELETE FROM generation_jobs WHERE id = ?", (forced,))
    print(f"enqueue: kind/phase/track linkage, idempotency, ready-only, "
          f"requires-filtered auto picks, explicit bypass, job #{job_id}")

    # ---- phase 1: the LLM image prompt -------------------------------------
    await art.run_art_phase(rt, job)
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["status"] == "queued" and job["phase"] == "art_render", \
        (job["status"], job["phase"])
    outputs = json.loads(job["progress"])["outputs"]
    assert "cathedral" in outputs["image_prompt"] and len(outputs["image_prompt"]) > 60
    stages = db.query("SELECT * FROM job_stage_log WHERE job_id = ? ORDER BY id", (job_id,))
    assert stages and all(r["response_text"] for r in stages)
    assert job["lyrics_started_at"] and job["lyrics_finished_at"]
    print(f"art phase: prompt extracted ({len(outputs['image_prompt'])} chars), "
          f"{len(stages)} exchange(s) logged")

    # ---- phase 2: the images render + storage ------------------------------
    # (dj.job_runner marks the job running before calling the phase)
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (job_id,))
    await art.run_art_render(rt, job)
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["status"] == "done" and job["used_seed"] == job["seed"]
    row = db.one("SELECT * FROM tracks WHERE id = ?", (t1,))
    assert row["art_version"] == 1 and row["art_workflow_id"] == job["workflow_id"]
    assert row["art_prompt"] == outputs["image_prompt"]
    tdir = library.track_dir(cfg, t1)
    assert (tdir / "art.png").is_file()
    assert (tdir / "art_512.jpg").is_file() and (tdir / "art_192.jpg").is_file()
    size1 = row["size_bytes"]
    assert size1 > 0
    print(f"art render: master + 2 jpeg thumbs stored, size_bytes={size1}, v1")

    # ---- links + versioning + regeneration ---------------------------------
    links = routes_public.art_links(cfg, t1, row["art_version"])
    assert links["card"].startswith(f"/art/{t1}/512.jpg?v=1")
    assert links["icon"].startswith(f"/art/{t1}/192.jpg?v=1")
    assert links["master"].startswith(f"/art/{t1}/full.png?v=1")
    assert links["card_px"] == 512 and links["icon_px"] == 192
    assert routes_public.art_links(cfg, t1, 0) is None
    # regenerate: same track, second job, version bumps (URLs change)
    j2 = art.enqueue_art_job(rt, t1, workflow_id="lyric-scene")
    job2 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (j2,))
    await art.run_art_phase(rt, job2)
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (j2,))
    await art.run_art_render(rt, db.one("SELECT * FROM generation_jobs WHERE id = ?", (j2,)))
    row = db.one("SELECT * FROM tracks WHERE id = ?", (t1,))
    assert row["art_version"] == 2 and row["art_workflow_id"] == "lyric-scene"
    assert routes_public.art_links(cfg, t1, row["art_version"])["card"].endswith("?v=2")
    print("links: versioned card/icon/master URLs; regeneration bumps the version")

    # unknown/odd names resolve to nothing (route 404s; no traversal)
    for bad_name in ("full.png/../../db", "art.png", "512.gif", "x512.png", "full"):
        assert routes_public._art_file(cfg, t1, bad_name) is None, bad_name
    assert routes_public._art_file(cfg, t1, "512.jpg").name == "art_512.jpg"
    assert routes_public._art_file(cfg, t1, "full.png").name == "art.png"

    # remove
    assert library.remove_art(db, cfg, t1) is True
    row = db.one("SELECT * FROM tracks WHERE id = ?", (t1,))
    assert row["art_version"] == 0 and row["art_prompt"] is None
    assert not (tdir / "art.png").exists() and not (tdir / "art_512.jpg").exists()
    assert row["size_bytes"] == 0  # art was the only thing counted
    assert routes_public.art_links(cfg, t1, row["art_version"]) is None
    print("remove: files gone, columns reset, quota bytes returned")

    # jpeg input normalization through ffmpeg
    jpg = tmp / "in.jpg"
    png = tmp / "in.png"
    png.write_bytes(MOCK_PNG)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(png), str(jpg)], check=True)
    await library.store_art(db, cfg, t1, jpg.read_bytes(), seed=1,
                            workflow_id="lyric-scene", prompt="jpeg in, png out")
    assert (tdir / "art.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    print("jpeg provider bytes normalized to a PNG master")

    # ---- GPU alternation ----------------------------------------------------
    db.execute("DELETE FROM generation_jobs")
    library.import_prompts(db, cfg)
    from infiniradio.generator.mock import MockGenerator
    rt.generator = MockGenerator(cfg)   # claims require a music backend
    prompt_id = db.one("SELECT id FROM prompts LIMIT 1")["id"]
    music1 = dj.enqueue_job(db, prompt_id=prompt_id)
    art1 = art.enqueue_art_job(rt, _make_track(db, cfg, title="Third"))
    # simulate the LLM phase having finished: the GPU queue sees phase
    # 'art_render' (run_art_phase flips it in production)
    db.execute("UPDATE generation_jobs SET phase = 'art_render' WHERE id = ?", (art1,))
    # both queued: claims alternate, music first (the restart default)
    first = dj._claim_gpu_job(rt, "art_render")
    second = dj._claim_gpu_job(rt, first["phase"])
    assert first["id"] == music1 and second["id"] == art1, (first["id"], second["id"])
    third = dj._claim_gpu_job(rt, second["phase"])
    assert third["id"] == music1  # no more art queued -> music again
    # master switch off: art jobs are unclaimable (they pause, not fail)
    db.set_setting("art.enabled", False)
    assert dj._claim_gpu_job(rt, "art_render")["id"] == music1
    db.set_setting("art.enabled", True)
    # a queue of art jobs never blocks music commissioning's budget
    art.enqueue_art_job(rt, _make_track(db, cfg, title="Fourth"))
    art.enqueue_art_job(rt, _make_track(db, cfg, title="Fifth"))
    queued_music = db.one("SELECT COUNT(*) AS n FROM generation_jobs"
                          " WHERE status = 'queued' AND kind = 'music'")["n"]
    assert queued_music == 1
    print("GPU claims alternate (music first), art pauses when disabled, "
          "music budget counts only music jobs")

    # ---- auto-enqueue for new tracks + backfill -----------------------------
    db.execute("DELETE FROM generation_jobs")
    db.set_setting("art.enabled", False)
    fresh = _make_track(db, cfg, title="Fresh")
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 0
    db.set_setting("art.enabled", True)
    db.set_setting("art.new_tracks", False)
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 0
    db.set_setting("art.new_tracks", True)
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 1
    # backfill: oldest artless track first, within budget; failed jobs are
    # never retried by the trickle
    db.execute("DELETE FROM generation_jobs")
    db.execute("UPDATE tracks SET art_version = 0")
    db.set_setting("art.backfill", True)
    oldest = db.one("SELECT MIN(id) AS m FROM tracks WHERE status = 'ready'")["m"]
    art.backfill_tick(rt)
    job = db.one("SELECT * FROM generation_jobs")
    assert job["track_id"] == oldest and job["kind"] == "art"
    art.backfill_tick(rt)   # budget 1 already used by the queued job
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 1
    db.execute("UPDATE generation_jobs SET status = 'failed'")
    art.backfill_tick(rt)   # the failed track is never re-enqueued...
    job2 = db.one("SELECT * FROM generation_jobs ORDER BY id DESC LIMIT 1")
    assert job2["track_id"] != oldest, (job2["track_id"], oldest)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 2
    print(f"auto-enqueue gated by enabled/new_tracks; backfill oldest-first "
          f"(track {oldest}), budgeted, failed tracks skipped not retried")

    # ---- schedule payload carries art ---------------------------------------
    db.execute("DELETE FROM generation_jobs")
    await library.store_art(db, cfg, fresh, MOCK_PNG, seed=3,
                            workflow_id="lyric-scene", prompt="x" * 80)
    from infiniradio.util import iso, utcnow
    db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
               (fresh, iso(utcnow())))
    payload = routes_public._schedule_payload(db, cfg, "listener-1", None, None)
    assert payload["current"]["id"] == fresh
    assert payload["current"]["art"]["card"].startswith(f"/art/{fresh}/512.jpg?v=")
    assert payload["recent"][0].get("art", "absent") == "absent"  # current-only
    print("schedule payload: current track carries versioned art links")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        asyncio.run(_main(cfg, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
