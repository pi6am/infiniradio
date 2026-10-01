"""Two-phase job flow on a temp DB: lyrics workflow commissioning, stage
logging, render hand-off, ingest with tags/lineage, crash reconciliation,
and failure->retry resume (the expensive-LLM-stages-never-rerun guarantee).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import _helpers

_helpers.ensure_repo()

from infiniradio import dj, library
from infiniradio.db import DB
from infiniradio.generator.base import GenRequest, GenResult
from infiniradio.generator.mock import MockGenerator
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import load_workflows

from _helpers import RT, isolated_config


async def _main(cfg, rt) -> None:
    db = rt.db
    library.import_prompts(db, cfg)
    library.seed_tags(db, cfg)

    # tag vocabulary: only llm_tags are offered to workflows
    vocabs = library.tag_vocabulary(db)
    assert {"pop", "rock", "blues", "metal", "rap", "male vocal"} <= set(vocabs), vocabs
    assert "vocal" not in vocabs and "english" not in vocabs  # forced, never offered
    print(f"tag vocabulary: {len(vocabs)} LLM-pickable (forced tags excluded)")

    # prompt tags come ONLY from declared `tags` in seed files — never
    # derived from slug names (which produced '8' + 'bit' junk and ran over
    # admin-created prompts at every boot)
    db.execute("INSERT INTO prompts (name, style_text, song_text, cot_mode, enabled,"
               " created_at) VALUES ('8-bit', 's', '', 'full', 1, '2026-01-01T00:00:00')")
    library.import_prompts(db, cfg)
    assert library.prompt_tags(db, db.one(
        "SELECT id FROM prompts WHERE name = '8-bit'")["id"]) == []
    assert db.one("SELECT COUNT(*) AS n FROM tags WHERE name IN ('8', 'bit')")["n"] == 0
    # declared tags still apply on import (fresh stations get them)
    assert library.prompt_tags(db, db.one(
        "SELECT id FROM prompts WHERE name = 'chiptune-upbeat'")["id"]) == \
        ["chiptune", "instrumental", "upbeat"]
    print("prompt tags: declared-only (no slug derivation; seeds still apply)")

    # commission a lyrics workflow job
    job_id = dj.enqueue_job(db, workflow_id="jp-title-first")
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["phase"] == "lyrics" and job["status"] == "queued"
    theme = json.loads(job["progress"])["theme_title"]
    assert theme
    print(f"job {job_id} queued (phase=lyrics, theme={theme!r})")

    # run the lyrics phase exactly as the runner does
    await dj._run_lyrics_phase(rt, job)
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["status"] == "queued" and job["phase"] == "render", \
        (job["status"], job["phase"])
    outputs = json.loads(job["progress"])["outputs"]
    assert outputs["title"] == "A Window Far Away" and outputs["lyrics"]
    stage_rows = db.query("SELECT * FROM job_stage_log WHERE job_id = ? ORDER BY id",
                          (job_id,))
    assert len(stage_rows) == 7 and all(r["response_text"] for r in stage_rows)
    assert all(json.loads(r["usage_json"]).get("finish_reason") == "stop"
               for r in stage_rows)  # diagnostics recorded per exchange
    print(f"lyrics phase done: 7 stages logged (+ finish_reasons), "
          f"title={outputs['title']!r}")

    # phase timings recorded; the detail-page breakdown computes from them
    from infiniradio.util import iso, parse_iso, utcnow
    from infiniradio.web.routes_admin import _job_timings
    assert job["lyrics_started_at"] and job["lyrics_finished_at"]
    assert parse_iso(job["created_at"]) <= parse_iso(job["lyrics_started_at"]) \
        <= parse_iso(job["lyrics_finished_at"])
    timings = dict(_job_timings(job))
    assert timings["waiting for lyrics"] and timings["lyrics"]
    assert "waiting for render" not in timings  # render not claimed yet
    assert "total" not in timings               # no finished_at, not running
    print(f"timings recorded: {timings}")

    # build the render request from the job row (the render runner path)
    req = dj.build_request(db, job)
    assert "dream pop" in req.style and "[Chorus]" in req.lyrics and req.abc is None

    # render phase body: create_track -> mock generate -> ingest
    track_id = library.create_track(
        db, cfg, prompt_id=None, parent_track_id=None, workflow_id=job["workflow_id"],
        style_id=job["style_id"], title=outputs["title"],
        theme_title=outputs["theme_title"], style_text=req.style,
        lyrics_snapshot=req.lyrics, seed=req.seed, cot_mode="full")
    result = await rt.generator.generate(req, library.track_dir(cfg, track_id))
    await library.ingest(db, cfg, track_id, result, tags=outputs["tags"])
    track = db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
    assert track["status"] == "ready" and track["title"] == "A Window Far Away"
    assert track["theme_title"] == theme and track["workflow_id"] == "jp-title-first"
    tag_names = library.track_tags(db, track_id)
    assert {"vocal", "english", "rock", "pop", "male vocal"} <= set(tag_names)
    print(f"track {track_id} ingested with tags {sorted(tag_names)}")

    # stats feed the commission bandit; reconcile fails running jobs
    assert dj.workflow_air_stats(db, "jp-title-first")["aired"] == 0
    assert dj.pick_generator(rt) is not None
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (job_id,))
    dj.reconcile_stale_jobs(db)
    assert db.one("SELECT status FROM generation_jobs WHERE id = ?",
                  (job_id,))["status"] == "failed"

    # legacy prompt jobs go straight to render
    pid = db.one("SELECT id FROM prompts LIMIT 1")["id"]
    jid2 = dj.enqueue_job(db, prompt_id=pid)
    j2 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid2,))
    assert j2["phase"] == "render" and j2["workflow_id"] is None
    print("prompt job enqueues straight to render (legacy behavior)")

    # prompt-job timings: single waiting row (no lyrics phase)
    db.execute("UPDATE generation_jobs SET render_started_at = ?, finished_at = ?,"
               " status = 'done' WHERE id = ?",
               (iso(utcnow()), iso(utcnow()), jid2))
    timings = dict(_job_timings(db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid2,))))
    assert timings["waiting"] and timings["rendering"] and timings["total"]
    assert "lyrics" not in timings and "waiting for lyrics" not in timings
    print(f"prompt-job timing rows: {list(timings)}")

    # re-compose: discards the parent's ABC, re-plans from edited lyrics,
    # keeps the title/theme/style lineage, and inherits the parent's tags
    fixed_lyrics = "[Intro]\n\n[Verse 1]\nrepaired lyrics\n\n[Chorus]\ncall the rain\n"
    jid4 = dj.enqueue_job(db, parent_track_id=track_id, recompose=True,
                          lyrics=fixed_lyrics, style_override="orchestral ballad, 70 BPM",
                          seed=77)
    j4 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid4,))
    assert j4["recompose"] == 1 and j4["phase"] == "render" and j4["workflow_id"] is None
    assert j4["style_id"] == job["style_id"]  # inherited from the parent
    out4 = json.loads(j4["progress"])["outputs"]
    assert out4["title"] == track["title"] and out4["theme_title"] == track["theme_title"]
    assert out4["lyrics"] == fixed_lyrics and "orchestral ballad" in out4["style"]
    req4 = dj.build_request(db, j4)
    assert req4.abc is None and req4.parent_track_id == track_id and req4.seed == 77
    assert req4.lyrics == fixed_lyrics and "orchestral ballad" in req4.style
    # the placeholder keeps the title; render copies the parent's tags over
    kid = library.create_track(
        db, cfg, prompt_id=None, parent_track_id=track_id, workflow_id=None,
        style_id=j4["style_id"], title=out4["title"], theme_title=out4["theme_title"],
        style_text=req4.style, lyrics_snapshot=req4.lyrics, seed=req4.seed,
        cot_mode="full")
    for name in library.track_tags(db, track_id):
        library.apply_track_tags(db, cfg, kid, [name])
    assert set(library.track_tags(db, kid)) == set(library.track_tags(db, track_id))
    kidrow = db.one("SELECT title FROM tracks WHERE id = ?", (kid,))
    assert kidrow["title"] == "A Window Far Away"  # same song, repaired
    # ...while a plain variation still re-renders the saved ABC
    jid5 = dj.enqueue_job(db, prompt_id=pid, parent_track_id=track_id)
    req5 = dj.build_request(db, db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid5,)))
    assert req5.abc == track["abc_text"] and req5.parent_track_id == track_id
    print("re-compose: abc=None + edited lyrics + lineage; variation unchanged")

    # failure persists completed stages; retry resumes, not restarts
    jid3 = dj.enqueue_job(db, workflow_id="jp-title-first")
    failing = RT(cfg, db, rt.generator, MockLLMClient(fail_first={"tags": 2}), rt.workflows)
    await dj._run_lyrics_phase(
        failing, db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid3,)))
    job3 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid3,))
    assert job3["status"] == "failed" and "tags" in (job3["error"] or "")
    mid = json.loads(job3["progress"])
    assert len(mid["completed"]) == 6 and mid["theme_title"]
    assert mid["vars"].get("song_name") == "A Window Far Away"
    print("failed run persisted 6/7 completed stages mid-run")
    db.execute("UPDATE generation_jobs SET status = 'queued', error = NULL,"
               " finished_at = NULL WHERE id = ? AND status = 'failed'", (jid3,))
    fresh = MockLLMClient()
    rt2 = RT(cfg, db, rt.generator, fresh, rt.workflows)
    await dj._run_lyrics_phase(
        rt2, db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid3,)))
    job3 = db.one("SELECT * FROM generation_jobs WHERE id = ?", (jid3,))
    assert job3["status"] == "queued" and job3["phase"] == "render"
    assert fresh.calls == 1, fresh.calls  # only the tags stage ran
    print("retry resumed at the tags stage only (1 LLM call)")

    # recompose through the REAL route handler (the form path) with a
    # recording generator, plus the real ComfyUI payload construction:
    # the edited lyrics must survive byte-for-byte to the workflow node
    from infiniradio.web import routes_admin

    mangled = fixed_lyrics + "\n\nWait, let me re-examine. partial thinking\n\n" + fixed_lyrics
    pid_parent = library.create_track(
        db, cfg, prompt_id=None, parent_track_id=None, workflow_id="jp-title-first",
        style_id=None, title="Disused Rounds", theme_title="Ember Puzzles",
        style_text="a slow art-pop song", lyrics_snapshot=mangled, seed=12345678,
        cot_mode="full")
    db.execute("UPDATE tracks SET status = 'ready', abc_text = 'X:1 old',"
               " playout_path = 'x', duration_s = 300 WHERE id = ?", (pid_parent,))

    captured: dict = {}

    class Recorder(MockGenerator):
        async def generate(self, req, workdir):
            captured["req"] = req
            return await super().generate(req, workdir)

    rt_rec = RT(cfg, db, Recorder(cfg), MockLLMClient(), rt.workflows)
    fake_app = type("A", (), {})()
    fake_app.state = type("S", (), {})()
    fake_app.state.rt = rt_rec
    FakeRequest = type("R", (), {"app": fake_app, "client": None})
    real_admin, real_csrf = routes_admin._is_admin, routes_admin._csrf_ok
    routes_admin._is_admin = lambda r: True
    routes_admin._csrf_ok = lambda r, c: True
    try:
        resp = await routes_admin.track_vary_submit(
            FakeRequest(), pid_parent, mode="recompose", style_prompt_id=0,
            custom_style="a slow art-pop song", lyrics=fixed_lyrics,
            title="Disused Rounds", seed=12345678, csrf="x")
        assert resp.headers["location"] == "/admin/jobs"
    finally:
        routes_admin._is_admin, routes_admin._csrf_ok = real_admin, real_csrf

    rjob = db.one("SELECT * FROM generation_jobs ORDER BY id DESC LIMIT 1")
    rout = json.loads(rjob["progress"])["outputs"]
    assert rjob["recompose"] == 1 and rout["lyrics"] == fixed_lyrics
    rreq = dj.build_request(db, rjob)
    assert rreq.lyrics == fixed_lyrics and rreq.abc is None and rreq.seed == 12345678
    rtrack = library.create_track(
        db, cfg, prompt_id=rjob["prompt_id"], parent_track_id=rjob["parent_track_id"],
        workflow_id=rjob["workflow_id"], style_id=rjob["style_id"],
        title=rout["title"], theme_title=rout["theme_title"], style_text=rreq.style,
        lyrics_snapshot=rreq.lyrics, seed=rreq.seed, cot_mode=rreq.cot_mode)
    result = await rt_rec.generator.generate(rreq, library.track_dir(cfg, rtrack))
    await library.ingest(db, cfg, rtrack, result, tags=None)
    assert captured["req"].lyrics == fixed_lyrics  # the generator got the edited text
    kid = db.one("SELECT title, lyrics_snapshot FROM tracks WHERE id = ?", (rtrack,))
    assert kid["title"] == "Disused Rounds" and kid["lyrics_snapshot"] == fixed_lyrics
    assert result.info["seed"] == 12345678  # effective seed travels in the result

    # job completion records the EFFECTIVE render seed (retry substitutions
    # are exactly what used to be unrecorded and un-replayable)
    db.execute("UPDATE generation_jobs SET status = 'running' WHERE id = ?", (rjob["id"],))
    dj._complete_job(db, rjob["id"], result)
    done = db.one("SELECT status, used_seed, comfy_prompt_id FROM generation_jobs"
                  " WHERE id = ?", (rjob["id"],))
    assert done["status"] == "done" and done["used_seed"] == 12345678
    db.execute("UPDATE generation_jobs SET status = 'running', used_seed = NULL"
               " WHERE id = ?", (rjob["id"],))
    dj._complete_job(db, rjob["id"], GenResult(
        original_path=Path("x.flac"), abc_text="",
        info={"seed": 999, "comfy_prompt_id": "p1"}))
    retry = db.one("SELECT used_seed FROM generation_jobs WHERE id = ?", (rjob["id"],))
    assert retry["used_seed"] == 999  # audit-retry substitution is captured
    print("job completion records the effective render seed (incl. retries)")

    # the real ComfyUI payload construction against the station's workflow
    # JSON (no HTTP): both the render pass and the pruned planning pass
    # must carry the edited lyrics in the lyrics node
    from infiniradio.generator.comfyui import ComfyUIGenerator
    cgen = ComfyUIGenerator(cfg)
    full_wf = cgen._patch_workflow(cgen._load_workflow(), rreq)
    assert cgen._node(full_wf, "lyrics")["inputs"]["value"] == fixed_lyrics
    plan_wf = cgen._patch_workflow(cgen._abc_only_subgraph(cgen._load_workflow()), rreq)
    assert cgen._node(plan_wf, "lyrics")["inputs"]["value"] == fixed_lyrics
    print("recompose route->render->comfy chain: edited lyrics intact end-to-end")

    # dispatch: recompose (abc=None + parent) takes the supervised single
    # pass — no audit, no reseeding; fresh songs keep the audit; variations
    # re-render the saved ABC directly
    from dataclasses import replace as _dc_replace

    calls: list[str] = []

    async def fake_full(req, workdir):
        calls.append("full")
        return GenResult(original_path=workdir / "x.flac", abc_text="X:1 T: K:C c")

    async def fake_audit(req, workdir):
        calls.append("audit")
        return GenResult(original_path=workdir / "x.flac", abc_text="X:1 T: K:C c")

    cgen._run_full = fake_full
    cgen._generate_with_audit = fake_audit
    import tempfile as _tf
    wd = Path(_tf.mkdtemp(prefix="infiniradio-dispatch-", dir=_tf.gettempdir()))
    await cgen.generate(rreq, wd)                                    # recompose
    await cgen.generate(_dc_replace(rreq, parent_track_id=None), wd)  # fresh song
    await cgen.generate(_dc_replace(rreq, abc="X:1 saved"), wd)       # variation
    assert calls == ["full", "audit", "full"], calls
    print("generator dispatch: recompose=single-pass, fresh=audited, "
          "variation=saved-abc")


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")
        rt = RT(cfg, db, MockGenerator(cfg), MockLLMClient(), load_workflows(cfg))
        asyncio.run(_main(cfg, rt))
        print("test_jobs: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
