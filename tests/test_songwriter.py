"""The songwriter workflow: one fine-tuned model call writes the title,
style prompt and lyrics in a trained reply shape (# heading + ```style /
```lyrics fences). Covers fresh-context retries, tags derived from the
style text (tags_in_text), load-time validation of the new keys, and the
jobs hand-off. Runs the real workflow definition against the scripted
mock LLM (temp data dir; no endpoint, no server)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import _helpers

_helpers.ensure_repo()

from infiniradio import dj, library
from infiniradio.config import load_config
from infiniradio.db import DB
from infiniradio.generator.mock import MockGenerator
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import (WorkflowError, load_workflows, run_workflow,
                                tags_in_text)

from _helpers import RT, isolated_config

VOCAB = ["pop", "rock", "synthwave", "female vocal"]
MODEL = "Suno-Song-Generator-gemma3-12B-HF.Q8_0"


def _spec():
    return load_workflows(load_config())["songwriter"]


async def _main(cfg, rt) -> None:
    spec = _spec()

    # 1. definition loads with the shape the fine-tune needs
    assert len(spec.stages) == 1 and spec.stages[0].id == "write"
    st = spec.stages[0]
    assert st.fresh_retries and st.thread == "new" and st.feedback_template is None
    assert st.model == MODEL and st.max_tokens == 8192
    assert spec.output.title == "song_title" and spec.output.tags is None
    assert spec.output.tags_from == "style_prompt" and spec.output.max_tags == 3
    assert not spec.styled and not spec.abstract and spec.enabled
    print("1. spec ok: 1 stage, fresh retries, per-stage model", MODEL)

    # 2. happy path: ONE LLM call yields title + style + lyrics; dynamic
    #    tags come out of the style text (pop, synthwave — and NOT
    #    "electropop", which is not in the vocabulary)
    mock = MockLLMClient()
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    assert mock.calls == 1
    assert r.title == "Velveeta"
    assert r.style.startswith("A cheesy, upbeat electropop track")
    assert r.lyrics.startswith("[verse 1]") and "Oh, Velveeta" in r.lyrics
    assert "```" not in r.lyrics and "#" not in r.lyrics
    assert r.tags == ["vocal", "english", "pop", "synthwave"], r.tags
    print("2. happy path ok: 1 call, 3 extracts, tags", r.tags)

    # 3. fresh retries: a malformed reply re-rolls the SAME prompt in a
    #    BLANK conversation — no feedback turn, no history, and the stage
    #    thread keeps only the winning exchange
    mock = MockLLMClient(bad_first={"Write a song about": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    assert mock.calls == 2
    assert [len(c) for c in mock.convs] == [1, 1]          # both blank slates
    assert mock.convs[0][0].content == mock.convs[1][0].content  # same prompt
    w = [t for t in r.attempts if t.stage_id == "write"]
    assert len(w) == 2 and not w[0].ok
    assert all(f.code == "extract_failed" for f in w[0].failures)
    conv = r.progress["threads"]["write"]
    assert len(conv) == 2 and conv[0][0] == "user" and conv[1][0] == "assistant"
    print("3. fresh retry ok: same prompt, blank conversation, clean thread")

    # 4. truncation mid-reply: the continuation completes the fences and
    #    the joined reply still parses (transport, not a content retry)
    mock = MockLLMClient(truncate_first={"Write a song about": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    w = [t for t in r.attempts if t.stage_id == "write"]
    assert len(w) == 1 and w[0].continuations == 1
    assert r.lyrics.startswith("[verse 1]") and r.title == "Velveeta"
    print("4. truncation continuation ok: fence closed across the seam")

    # 5. exhaustion: three fresh re-rolls, then the job fails
    mock = MockLLMClient(bad_first={"Write a song about": 3})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert not r.ok and "write" in r.error and "extract_failed" in r.error
    assert mock.calls == 3 and all(len(c) == 1 for c in mock.convs)
    print("5. exhaustion ok:", r.error)

    # 6. tags_in_text: word boundaries, plurals, no substring traps
    assert tags_in_text(["pop", "rap", "soul", "female vocal", "lo-fi"],
                        "Popular wrapper souls. Female vocals over lo-fi beats") \
        == ["soul", "female vocal", "lo-fi"]
    assert tags_in_text(VOCAB, "") == []
    assert tags_in_text([], "pop all day") == []
    assert tags_in_text(VOCAB, "SYNTHWAVE revival") == ["synthwave"]
    print("6. tags_in_text ok: boundaries, plurals, case-insensitive")

    # 7. jobs hand-off: enqueue → lyrics phase → render queue, one LLM
    #    exchange logged under the fine-tuned model's name
    db = rt.db
    library.seed_tags(db, cfg)
    job_id = dj.enqueue_job(db, workflow_id="songwriter")
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["phase"] == "lyrics" and job["status"] == "queued"
    theme = json.loads(job["progress"])["theme_title"]
    assert theme
    await dj._run_lyrics_phase(rt, job)
    job = db.one("SELECT * FROM generation_jobs WHERE id = ?", (job_id,))
    assert job["status"] == "queued" and job["phase"] == "render"
    outputs = json.loads(job["progress"])["outputs"]
    assert outputs["title"] == "Velveeta" and outputs["theme_title"] == theme
    assert outputs["style"].startswith("A cheesy, upbeat")
    assert outputs["lyrics"].startswith("[verse 1]")
    # derived against the real seeded vocabulary (alphabetical order)
    assert outputs["tags"] == ["vocal", "english", "pop", "synthwave"], outputs["tags"]
    req = dj.build_request(db, job)
    assert "synthwave" in req.style and req.lyrics.startswith("[verse 1]")
    rows = db.query("SELECT * FROM job_stage_log WHERE job_id = ? ORDER BY id",
                    (job_id,))
    assert len(rows) == 1 and rows[0]["stage_id"] == "write"
    assert rows[0]["model"] == MODEL   # per-stage routing reached the request
    assert rt.llm.calls == 1
    print(f"7. jobs ok: theme {theme!r} → track title {outputs['title']!r}, "
          f"stage log names the fine-tuned model")


# --------------------------------------------------------------- validation

_TEMPLATE = "Write a song about {{ title }}.\n"


def _broken_workflows(toml: str) -> Path:
    """A temp workflows dir holding one 'broken' workflow; validation of
    the RESOLVED spec is what the tests exercise."""
    root = Path(tempfile.mkdtemp(prefix="infiniradio-wf-"))
    wf = root / "broken"
    wf.mkdir()
    (wf / "workflow.toml").write_text(toml, encoding="utf-8")
    (wf / "write.j2.md").write_text(_TEMPLATE, encoding="utf-8")
    return root


def _expect_error(root: Path, needle: str) -> None:
    cfg = load_config()
    object.__setattr__(cfg, "_d", {**cfg._d,
                                   "lyrics": {**cfg._d["lyrics"],
                                              "workflows_dir": str(root)}})
    try:
        load_workflows(cfg)
    except WorkflowError as exc:
        assert needle in str(exc), (needle, str(exc))
        print(f"   rejected: {exc}")
        return
    raise AssertionError(f"expected WorkflowError mentioning {needle!r}")


def _validation_cases() -> list[Path]:
    roots: list[Path] = []

    root = _broken_workflows("""\
name = "broken"
description = "x"
[[stage]]
id = "write"
template = "write.j2.md"
fresh_retries = true
feedback_template = "write.j2.md"
""")
    _expect_error(root, "no feedback turn")
    roots.append(root)

    root = _broken_workflows("""\
name = "broken"
description = "x"
[[stage]]
id = "one"
template = "write.j2.md"
[[stage]]
id = "two"
template = "write.j2.md"
thread = "continue"
fresh_retries = true
""")
    _expect_error(root, "fresh_retries needs thread = 'new'")
    roots.append(root)

    root = _broken_workflows("""\
name = "broken"
description = "x"
[output]
style = "style_prompt"
lyrics = "final_lyrics"
tags = "track_tags"
tags_from = "style_prompt"

[[stage]]
id = "write"
template = "write.j2.md"
[stage.extract]
style_prompt = '(?s)```style\\n(.*?)\\n```'
final_lyrics = '(?s)```lyrics\\n(.*?)\\n```'
track_tags = '(.*)'
""")
    _expect_error(root, "mutually exclusive")
    roots.append(root)

    root = _broken_workflows("""\
name = "broken"
description = "x"
[output]
style = "style_prompt"
lyrics = "final_lyrics"
tags_from = "nowhere"

[[stage]]
id = "write"
template = "write.j2.md"
[stage.extract]
style_prompt = '(?s)```style\\n(.*?)\\n```'
final_lyrics = '(?s)```lyrics\\n(.*?)\\n```'
""")
    _expect_error(root, "not saved or extracted")
    roots.append(root)

    print("8. validation ok: fresh_retries/tags_from misuse rejected at load")
    return roots


def run() -> None:
    cfg, tmp = isolated_config()
    leftovers: list[Path] = []
    try:
        db = DB(tmp / "infiniradio.db")
        rt = RT(cfg, db, MockGenerator(cfg), MockLLMClient(), load_workflows(cfg))
        asyncio.run(_main(cfg, rt))
        leftovers.extend(_validation_cases())
        print("test_songwriter: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for p in leftovers:
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    run()
