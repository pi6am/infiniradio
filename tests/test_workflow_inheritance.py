"""Workflow inheritance: base chains deep-merge (child wins), stages merge
by id with positioned insertion and removal, templates resolve child-first
then ancestors, abstract bases load but are never commissioned — and every
mistake fails at startup. Verified against synthetic fixtures in a temp
workflows dir, plus the real title-first family.
"""

from __future__ import annotations

import asyncio
import shutil

import _helpers

_helpers.ensure_repo()

from infiniradio import dj
from infiniradio.db import DB
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import WorkflowError, load_workflows, run_workflow

from _helpers import RT, isolated_config

BASE_TOML = """
abstract = true
name = "base"
description = "the shared pipeline"
seed_title = true
styled = true
tags = ["vocal"]
[params]
greeting = "hello"
language = "jp"
[vars]
style_guide = ""
[output]
style = "s_out"
lyrics = "l_out"
title = "t_out"
max_tags = 3
[[stage]]
id = "one"
template = "one.j2.md"
saves = "t_out"
[[stage]]
id = "two"
template = "two.j2.md"
saves = "s_out"
max_attempts = 2
[[stage]]
id = "three"
template = "three.j2.md"
saves = "l_out"
"""

CHILD_TOML = """
name = "child"
base = "base"
description = "child override"
remove_stages = ["three"]
[params]
language = "en"
[[stage]]
id = "two"
max_attempts = 5
[[stage]]
id = "mid"
after = "one"
template = "mid.j2.md"
saves = "l_out"
"""

def _fixture_dir(tmp, files: dict[str, str]):
    wf = tmp / "wf"
    wf.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = wf / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return wf


def _load_from(cfg, wf_dir):
    object.__setattr__(cfg, "_d", {**cfg._d,
                                   "lyrics": {**cfg._d["lyrics"],
                                              "workflows_dir": str(wf_dir)}})
    return load_workflows(cfg)


def _real_family_checks(cfg) -> None:
    specs = load_workflows(cfg)
    assert set(specs) == {"title-first", "jp-title-first", "zh-title-first"}
    assert specs["title-first"].abstract
    assert not specs["jp-title-first"].abstract and specs["jp-title-first"].styled
    jp = specs["jp-title-first"]
    zh = specs["zh-title-first"]
    assert jp.params["language"] == "Japanese" and zh.params["language"] == "Chinese"
    assert [s.id for s in zh.stages] == [s.id for s in jp.stages]
    assert zh.output.max_tags == jp.output.max_tags == 5   # jp's authority
    refine = next(s for s in zh.stages if s.id == "refine")
    assert refine.max_attempts == 3 and refine.max_tokens == 12288
    # jp resolves byte-identically whether standalone history or inherited
    assert jp.directory.name == "jp-title-first" and len(jp.stages) == 7
    print("0. real family: title-first base + jp/zh children, jp-authority values")


async def _child_executes(spec) -> None:
    r = await run_workflow(MockLLMClient(), spec, initial_vars={"title": "T"},
                           tag_vocabulary=[])
    assert r.ok, r.error
    assert r.style and r.lyrics          # "(mock reply)" from the fallback script
    assert r.tags == ["vocal"]           # static inherited, picks sanitized to none
    assert r.title == "(mock reply)"


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        _real_family_checks(cfg)

        files = {
            "base/workflow.toml": BASE_TOML,
            "base/one.j2.md": "one {{ params.greeting }} {{ params.language }}",
            "base/two.j2.md": "two base template",
            "base/three.j2.md": "three",
            "base/feedback.j2.md": "fix it",
            "child/workflow.toml": CHILD_TOML,
            "child/two.j2.md": "two CHILD template",     # shadows the base's
            "child/mid.j2.md": "mid",
        }
        wf = _fixture_dir(tmp, files)
        specs = _load_from(cfg, wf)
        assert set(specs) == {"base", "child"} and specs["base"].abstract

        child = specs["child"]
        # scalars/params: child wins where set, inherits where not
        assert child.description == "child override"
        assert child.params == {"greeting": "hello", "language": "en"}
        assert child.static_tags == ("vocal",) and child.seed_title and child.styled
        assert child.output.max_tags == 3 and child.output.style == "s_out"
        assert "style_guide" in child.vars
        # stages: order one -> mid -> two; two overridden per-field; three gone
        assert [s.id for s in child.stages] == ["one", "mid", "two"]
        two = next(s for s in child.stages if s.id == "two")
        assert two.max_attempts == 5 and two.saves == "s_out"  # override + inherit
        assert two.template == "two.j2.md"                      # name resolves child-first
        # templates: child shadow + base fallback through the search path
        assert "two CHILD template" == child.env.get_template("two.j2.md").render()
        assert "one hello en" == child.env.get_template("one.j2.md").render(
            params=child.params)
        asyncio.run(_child_executes(child))
        print("1. merge: scalars/params/stages/templates + a resolved child executes")

        # abstract bases never commission
        db = DB(tmp / "wfdb")
        db.set_setting("dj.lyrics_share", 1.0)
        rt = RT(cfg, db, None, MockLLMClient(), specs)
        picks = {dj.pick_generator(rt)[0] for _ in range(20)}
        assert picks == {"workflow"}, picks
        assert {w.name for w in dj.enabled_workflows(rt)} == {"child"}
        print("2. abstract excluded from commissioning (enabled_workflows)")

        # every mistake fails at startup with a clear error
        counter = {"n": 0}

        def broken(child_toml: str | None, extra: dict | None, expect: str) -> None:
            counter["n"] += 1
            f = dict(files)
            if child_toml is not None:
                f["child/workflow.toml"] = child_toml
            if extra:
                f.update(extra)
            d = tmp / f"broken{counter['n']}"
            try:
                _load_from(cfg, _fixture_dir(d, f))
                raise AssertionError(f"accepted broken workflow: {expect}")
            except WorkflowError as exc:
                assert expect in str(exc), (expect, str(exc))
            finally:
                shutil.rmtree(d, ignore_errors=True)

        broken(CHILD_TOML.replace('base = "base"', 'base = "nope"'), None,
               "names no loaded workflow")
        broken(CHILD_TOML.replace(
            'remove_stages = ["three"]', 'remove_stages = ["nonexistent"]'), None,
            "unknown stage")
        broken(CHILD_TOML.replace('after = "one"', 'after = "ghost"'), None,
               "names no known stage")
        broken(CHILD_TOML.replace(
            'id = "two"\nmax_attempts = 5', 'id = "two"\nafter = "one"\nmax_attempts = 5'),
            None, "position is inherited")
        broken(None,
               {"cyc/workflow.toml": 'name = "cyc"\nbase = "cycbase"\n',
                "cycbase/workflow.toml": 'name = "cycbase"\nbase = "cyc"\n'
                '[[stage]]\nid = "a"\ntemplate = "one.j2.md"\n'},
               "cycle")
        print("3. errors: unknown base, cycle, bad remove, bad after, after-on-override")

        print("test_workflow_inheritance: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()