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
    return rt


async def _main(cfg, tmp) -> None:
    db = DB(cfg.path("library.data_dir") / "infiniradio.db")

    # ---- loading ----------------------------------------------------------
    wf = art.load_art_workflows(cfg)
    assert set(wf) == {"cover", "cover-illustrated", "cover-photographic",
                       "lyric-scene", "analysis-abstract", "analysis-typographic"}
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
                                            "cover-photographic": 0.0,
                                            "analysis-abstract": 0.0,
                                            "analysis-typographic": 0.0})
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
    db.set_setting("art_workflows_disabled", ["lyric-scene", "analysis-abstract",
                                              "analysis-typographic"])
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
    # a lyrics-requiring workflow; an explicit admin pick bypasses the filter
    instr = _make_track(db, cfg, title="Instrumental",
                        lyrics="[Intro]\n\n[Instrumental]\n\n[Outro]")
    for _ in range(10):
        j = art.enqueue_art_job(rt, instr)
        assert j is not None and db.one(
            "SELECT workflow_id FROM generation_jobs WHERE id = ?", (j,)
        )["workflow_id"] in ("cover-illustrated", "cover-photographic")
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
    assert row["art_id"] is not None
    art_row = db.one("SELECT * FROM track_art WHERE id = ?", (row["art_id"],))
    assert art_row["workflow_id"] == job["workflow_id"]
    assert art_row["job_id"] == job_id and art_row["prompt"] == outputs["image_prompt"]
    adir = library.art_dir(cfg, t1)
    assert (adir / f"{row['art_id']}.png").is_file()
    assert (adir / f"{row['art_id']}_512.jpg").is_file()
    assert (adir / f"{row['art_id']}_192.jpg").is_file()
    first_art_id = row["art_id"]
    size1 = row["size_bytes"]
    assert size1 > 0
    print(f"art render: master + 2 jpeg thumbs stored as art #{first_art_id}")

    # the analysis workflows: single stage, Prompt:-anchored extraction
    j3 = art.enqueue_art_job(rt, t2, workflow_id="analysis-abstract")
    job3 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (j3,))
    await art.run_art_phase(rt, job3)
    job3 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (j3,))
    assert job3["phase"] == "art_render" and job3["status"] == "queued"
    out3 = json.loads(job3["progress"])["outputs"]["image_prompt"]
    assert out3.startswith("Album art of") and len(out3) > 60, out3[:80]
    print("analysis workflow: single stage, Prompt:-anchored extraction ok")

    # ---- template whitespace discipline ---------------------------------------
    # trim_blocks eats the newline after every block tag: tags at the ENDS
    # of content lines glue lines together (seen in prod). Art templates
    # keep tags at line starts and fence multi-line values — assert both.
    spec = wf["cover-illustrated"]
    base = art.track_context(db, db.one("SELECT * FROM tracks WHERE id = ?", (t1,)))
    txt = spec.env.get_template("concept.j2.md").render(
        **{**dict(spec.vars), **base, "params": spec.params})
    assert "\n- Music: " in txt and txt.count("- Title: ") == 1
    assert "```\n[Verse]" in txt and "\n```" in txt      # lyrics fenced
    assert "\n\nDescribe ONE cover" in txt               # separation kept
    instrumental = dict(base, theme_title="", track_tags=[], genre="",
                        tempo=0, music_key="", duration_s=0.0,
                        lyrics="[Intro]\n\n[Instrumental]\n\n[Outro]",
                        has_lyrics=False)
    txt2 = spec.env.get_template("concept.j2.md").render(
        **{**dict(spec.vars), **instrumental, "params": spec.params})
    assert "- No lyrics — an instrumental track." in txt2
    assert "- Lyrics:" not in txt2 and "```" not in txt2   # no skeleton fence
    assert "- Theme:" not in txt2 and "- Tags:" not in txt2 and "- Tempo:" not in txt2
    assert "\n- Music: " in txt2   # optional bullets gone without gluing
    scene = wf["lyric-scene"].env.get_template("scene.j2.md").render(
        **{**dict(wf["lyric-scene"].vars), **base, "params": {}})
    assert "Lyrics:\n```\n[Verse]" in scene
    print("templates: bullets on their own lines, lyrics fenced, no glue")

    # ---- links, history, current pick, delete -------------------------------
    links = routes_public.art_links(db, cfg, t1)
    assert links["card"].startswith(f"/art/{t1}/{first_art_id}/512.jpg")
    assert links["icon"].startswith(f"/art/{t1}/{first_art_id}/192.jpg")
    assert links["master"].startswith(f"/art/{t1}/{first_art_id}/full.png")
    assert links["card_px"] == 512 and links["icon_px"] == 192
    # regenerate: a SECOND immutable image — the first is untouched
    j2 = art.enqueue_art_job(rt, t1, workflow_id="lyric-scene")
    job2 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (j2,))
    await art.run_art_phase(rt, job2)
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (j2,))
    await art.run_art_render(rt, db.one("SELECT * FROM generation_jobs WHERE id = ?", (j2,)))
    row = db.one("SELECT * FROM tracks WHERE id = ?", (t1,))
    second_art_id = row["art_id"]
    assert second_art_id != first_art_id
    assert (adir / f"{first_art_id}.png").is_file()   # history kept
    rows = library.track_art_rows(db, t1)
    assert [r["id"] for r in rows] == [second_art_id, first_art_id]  # newest first
    assert routes_public.art_links(db, cfg, t1)["card"].startswith(
        f"/art/{t1}/{second_art_id}/512.jpg")          # newest is current
    # re-pick: the admin's choice changes the URLs the player gets
    assert library.set_current_art(db, t1, first_art_id) is True
    assert routes_public.art_links(db, cfg, t1)["master"] == \
        f"/art/{t1}/{first_art_id}/full.png"
    assert library.set_current_art(db, t2, first_art_id) is False  # not that track's
    library.set_current_art(db, t1, second_art_id)
    print("history: two immutable images, newest current, re-pick switches URLs")

    # unknown/odd names resolve to nothing (route 404s; no traversal)
    for bad_name in ("full.png/../../db", "art.png", "512.gif", "x512.png", "full"):
        assert routes_public._art_file(cfg, t1, second_art_id, bad_name) is None, bad_name
    assert routes_public._art_file(cfg, t1, second_art_id, "512.jpg").name == \
        f"{second_art_id}_512.jpg"
    assert routes_public._art_file(cfg, t1, second_art_id, "full.png").name == \
        f"{second_art_id}.png"

    # delete: a non-current image leaves the current pick alone
    assert library.remove_art(db, cfg, first_art_id) is True
    assert not (adir / f"{first_art_id}.png").exists()
    assert db.one("SELECT COUNT(*) AS n FROM track_art WHERE id = ?",
                  (first_art_id,))["n"] == 0
    assert row["art_id"] == second_art_id or db.one(
        "SELECT art_id FROM tracks WHERE id = ?", (t1,))["art_id"] == second_art_id
    # deleting the CURRENT image leaves the track artless
    assert library.remove_art(db, cfg, second_art_id) is True
    row = db.one("SELECT * FROM tracks WHERE id = ?", (t1,))
    assert row["art_id"] is None
    assert routes_public.art_links(db, cfg, t1) is None
    assert row["size_bytes"] == 0  # art bytes returned to the quota
    print("delete: non-current kept, current deletion clears the pick, bytes returned")

    # jpeg input normalization through ffmpeg
    jpg = tmp / "in.jpg"
    png = tmp / "in.png"
    png.write_bytes(MOCK_PNG)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(png), str(jpg)], check=True)
    norm_id = await library.store_art(db, cfg, t1, jpg.read_bytes(), seed=1,
                                      workflow_id="lyric-scene",
                                      prompt="jpeg in, png out")
    assert (adir / f"{norm_id}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    print("jpeg provider bytes normalized to a PNG master")

    # ---- legacy (v14) layout relocation --------------------------------------
    from infiniradio.util import iso, utcnow
    t_legacy = _make_track(db, cfg, title="Legacy")
    legacy_id = db.execute(
        "INSERT INTO track_art (track_id, workflow_id, seed, prompt, created_at)"
        " VALUES (?, 'cover', 1, 'p', ?)", (t_legacy, iso(utcnow())))
    tdir = library.track_dir(cfg, t_legacy)
    (tdir / "art.png").write_bytes(MOCK_PNG)
    (tdir / "art_512.jpg").write_bytes(b"thumb")
    db.execute("UPDATE tracks SET art_id = ? WHERE id = ?", (legacy_id, t_legacy))
    assert library.migrate_legacy_art(db, cfg) == 1
    assert (tdir / "art" / f"{legacy_id}.png").is_file()
    assert (tdir / "art" / f"{legacy_id}_512.jpg").is_file()
    assert not (tdir / "art.png").exists()
    assert library.migrate_legacy_art(db, cfg) == 0   # idempotent
    print("legacy v14 art relocated onto its row (idempotent)")

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
    # claims are gated by CONFIGURATION only (pipeline_ready): with the
    # images backend gone, art jobs wait — no runtime switch ever blocks
    # an already-queued job
    rt.images = None
    assert dj._claim_gpu_job(rt, "render")["id"] == music1
    rt.images = MockImagesClient()
    # a queue of art jobs never blocks music commissioning's budget
    art.enqueue_art_job(rt, _make_track(db, cfg, title="Fourth"))
    art.enqueue_art_job(rt, _make_track(db, cfg, title="Fifth"))
    queued_music = db.one("SELECT COUNT(*) AS n FROM generation_jobs"
                          " WHERE status = 'queued' AND kind = 'music'")["n"]
    assert queued_music == 1
    print("GPU claims alternate (music first), unconfigured art waits, "
          "music budget counts only music jobs")

    # ---- auto-enqueue for new tracks + backfill -----------------------------
    db.execute("DELETE FROM generation_jobs")
    rt.images = None                   # pipeline not configured: nothing
    fresh = _make_track(db, cfg, title="Fresh")   # automatic ever enqueues
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 0
    rt.images = MockImagesClient()
    db.set_setting("art.new_tracks", False)
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 0
    db.set_setting("art.new_tracks", True)
    art.maybe_enqueue_for_track(rt, fresh)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 1
    # backfill: fills the budget each tick, oldest first; failed jobs are
    # never retried by the trickle
    db.execute("DELETE FROM generation_jobs")
    db.execute("UPDATE tracks SET art_id = NULL")
    db.set_setting("art.backfill", True)
    db.set_setting("art.backfill_max_queued", 1)
    oldest = db.one("SELECT MIN(id) AS m FROM tracks WHERE status = 'ready'")["m"]
    art.backfill_tick(rt)
    job = db.one("SELECT * FROM generation_jobs")
    assert job["track_id"] == oldest and job["kind"] == "art"
    art.backfill_tick(rt)   # budget 1 already used by the queued job
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 1
    db.set_setting("art.backfill_max_queued", 3)
    art.backfill_tick(rt)   # fills ALL headroom, not one per tick
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs"
                  " WHERE status = 'queued'")["n"] == 3
    db.execute("UPDATE generation_jobs SET status = 'failed'")
    art.backfill_tick(rt)   # the failed tracks are never re-enqueued...
    job2 = db.one("SELECT * FROM generation_jobs ORDER BY id DESC LIMIT 1")
    assert job2["track_id"] != oldest, (job2["track_id"], oldest)
    assert db.one("SELECT COUNT(*) AS n FROM generation_jobs")["n"] == 6
    print(f"auto-enqueue gated by pipeline/new_tracks; backfill fills the "
          f"budget per tick (oldest first, track {oldest}), failed tracks skipped")

    # ---- images client: model is optional ------------------------------------
    from infiniradio.config import Config as _Config
    from infiniradio.llm.images import ImagesClient
    from pathlib import Path as _Path
    base = {"art": {"api": {"base_url": "http://x/v1"}}}
    c = ImagesClient(_Config(base, _Path(".")))
    assert "model" not in c._payload("p", "1024x1024", 5)
    full = {"art": {"api": {"base_url": "http://x/v1", "model": "sd",
                            "send_seed": True}}}
    assert ImagesClient(_Config(full, _Path(".")))._payload("p", "512x512", 7) == {
        "prompt": "p", "size": "512x512", "n": 1,
        "response_format": "b64_json", "model": "sd", "seed": 7}
    print("images payload: model omitted when unset, sent (with seed) when set")

    # the admin regen route refuses while the master switch is off — a
    # queued art job is unclaimable then, which reads as a hang
    from infiniradio.web import routes_admin
    real_admin, real_csrf = routes_admin._is_admin, routes_admin._csrf_ok
    routes_admin._is_admin = lambda r: True
    routes_admin._csrf_ok = lambda r, c: True
    fake = type("R", (), {"app": type("A", (), {})()})()
    fake.app.state = type("S", (), {})()
    fake.app.state.rt = rt
    try:
        # manual regen queues whenever the pipeline is configured — no
        # runtime switch (the Generate-page rule)
        ok = await routes_admin.track_art_regen(fake, t2, workflow="", csrf="")
        assert ok.status_code == 303
        j4 = db.one("SELECT * FROM generation_jobs ORDER BY id DESC LIMIT 1")
        assert j4["kind"] == "art" and j4["track_id"] == t2
        db.execute("DELETE FROM generation_jobs WHERE id = ?", (j4["id"],))
        db.set_setting("generation_paused", True)   # the pause never blocks
        ok = await routes_admin.track_art_regen(fake, t2, workflow="", csrf="")
        assert ok.status_code == 303
        db.execute("DELETE FROM generation_jobs")   # the manual job
        db.set_setting("generation_paused", False)
    finally:
        routes_admin._is_admin, routes_admin._csrf_ok = real_admin, real_csrf
    print("regen route: manual queueing waits for no runtime switch or pause")

    # ---- schedule payload carries art ---------------------------------------
    db.execute("DELETE FROM generation_jobs")
    fresh_id = await library.store_art(db, cfg, fresh, MOCK_PNG, seed=3,
                                       workflow_id="lyric-scene", prompt="x" * 80)
    from infiniradio.util import iso, utcnow
    db.execute("INSERT INTO play_log (track_id, started_at) VALUES (?, ?)",
               (fresh, iso(utcnow())))
    payload = routes_public._schedule_payload(db, cfg, "listener-1", None, None)
    assert payload["current"]["id"] == fresh
    assert payload["current"]["art"]["card"] == f"/art/{fresh}/{fresh_id}/512.jpg"
    assert payload["recent"][0].get("art", "absent") == "absent"  # current-only
    print("schedule payload: current track carries the current pick's links")

    # ---- the gallery grid: current picks only, newest first -----------------
    gallery = routes_admin._gallery_entries(db, cfg, 50)
    ids = [g["track_id"] for g in gallery]
    assert fresh in ids and t2 not in ids       # t2 has no stored art
    entry = next(g for g in gallery if g["track_id"] == fresh)
    assert entry["card"] == f"/art/{fresh}/{fresh_id}/512.jpg"
    assert [g["art_id"] for g in gallery] == \
        sorted((g["art_id"] for g in gallery), reverse=True)
    print("gallery: current picks only, newest image first")

    # the gallery page renders, including the backfill status line (the
    # trickle is held silently by the generation pause otherwise)
    class _Sess:
        def validate(self, c): return True
        def csrf_token(self, c): return "x"
    fake.app.state.sessions = _Sess()
    from infiniradio.app import WEB_DIR
    from fastapi.templating import Jinja2Templates
    fake.app.state.templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    fake.cookies = {}
    fake.url = type("U", (), {"path": "/admin/gallery"})()
    resp = await routes_admin.gallery_page(fake)
    html = resp.body.decode()
    assert "backfill on ·" in html and "waiting" in html
    db.set_setting("generation_paused", True)
    held = (await routes_admin.gallery_page(fake)).body.decode()
    assert "held (paused from the admin panel)" in held
    db.set_setting("generation_paused", False)
    print("gallery page: backfill status line shown, held reason when paused")

    # ---- same-prompt re-roll: fresh seed, no LLM pass -------------------------
    # (route first, then the functional path; `fresh` carries the current
    #  pick — prompt "x"*80, seed 3 — and t2 has none)
    routes_admin._is_admin = lambda r: True
    routes_admin._csrf_ok = lambda r, c: True
    try:
        ok = await routes_admin.track_art_reroll(fake, fresh, csrf="")
        assert ok.status_code == 303
        rjr = db.one("SELECT * FROM generation_jobs ORDER BY id DESC LIMIT 1")
        assert rjr["kind"] == "art" and rjr["phase"] == "art_render"
        assert rjr["track_id"] == fresh and rjr["status"] == "queued"
        assert json.loads(rjr["progress"])["outputs"]["image_prompt"] == "x" * 80
        assert (await routes_admin.track_art_reroll(fake, t2, csrf="")).status_code == 400
        db.execute("DELETE FROM generation_jobs WHERE id = ?", (rjr["id"],))
    finally:
        routes_admin._is_admin, routes_admin._csrf_ok = real_admin, real_csrf
    print("reroll route: queues straight to art_render; no current art -> 400")

    before = library.current_art(db, fresh)
    assert before["prompt"] == "x" * 80 and before["seed"] == 3
    rj = art.enqueue_art_reroll(rt, fresh)
    assert rj is not None
    rjob = db.one("SELECT * FROM generation_jobs WHERE id = ?", (rj,))
    assert rjob["phase"] == "art_render" and rjob["workflow_id"] == "lyric-scene"
    assert rjob["seed"] not in (None, 3)          # a FRESH seed, never the same
    assert json.loads(rjob["progress"])["outputs"]["image_prompt"] == "x" * 80
    assert art.enqueue_art_reroll(rt, fresh) == rj          # idempotent while queued
    assert art.enqueue_art_reroll(rt, t2) is None           # nothing current
    # the GPU claim gates art renders on the images endpoint only — a
    # re-roll carries no LLM work, so no llm/workflow requirement may hold it
    assert dj._claim_gpu_job(rt, "render")["id"] == rj
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (rj,))
    await art.run_art_render(rt, rjob)
    row = db.one("SELECT * FROM tracks WHERE id = ?", (fresh,))
    assert row["art_id"] not in (None, before["id"])        # a NEW image, now current
    after = db.one("SELECT * FROM track_art WHERE id = ?", (row["art_id"],))
    assert after["prompt"] == "x" * 80 and after["seed"] == rjob["seed"]
    assert after["job_id"] == rj and after["workflow_id"] == "lyric-scene"
    assert (library.art_dir(cfg, fresh) / f"{row['art_id']}.png").is_file()
    print(f"reroll: same prompt + fresh seed -> new image #{row['art_id']}, "
          f"history and old files untouched")

    # ---- gallery pagination (first/prev/next/last + page indicator) ----------
    pool = [t2, instr] + [_make_track(db, cfg, title=f"Pageable {i}") for i in range(9)]
    for tid in pool:
        await library.store_art(db, cfg, tid, MOCK_PNG, seed=1,
                                workflow_id="cover", prompt="paged")
    # fresh + the 11 above = 12 current picks -> two pages at n=10
    page1 = (await routes_admin.gallery_page(fake, n=10)).body.decode()
    assert "page 1 / 2" in page1 and page1.count('<img src="/art/') == 10
    assert 'href="/admin/gallery?page=2&amp;n=10"' in page1    # next/last carry n
    assert 'href="/admin/gallery?page=1' not in page1          # at the first page
    page2 = (await routes_admin.gallery_page(fake, n=10, page=2)).body.decode()
    assert "page 2 / 2" in page2 and page2.count('<img src="/art/') == 2
    assert 'href="/admin/gallery?page=1&amp;n=10"' in page2    # first/prev enabled
    assert 'href="/admin/gallery?page=2&amp;n=10"' not in page2
    clamped = (await routes_admin.gallery_page(fake, n=10, page=99)).body.decode()
    assert "page 2 / 2" in clamped                             # over-range clamps
    print("gallery pager: first/prev/next/last + page indicator, page clamp ok")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        asyncio.run(_main(cfg, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
