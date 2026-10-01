"""Lyrics workflow engine: happy path, gate/extract retries with feedback,
llm_error exhaustion, resume, guard cancellation, song-block extraction,
liberal name parsing, tag sanitization, and truncation continuation.

Runs against the scripted mock LLM (src/infiniradio/llm/mock.py) whose
replies are shaped like the real endpoint's failure modes — including the
thinking-leak and token-budget-truncation specimens observed in production.
"""

from __future__ import annotations

import asyncio

import _helpers

_helpers.ensure_repo()

from infiniradio.config import load_config
from infiniradio.llm.mock import MockLLMClient
from infiniradio.lyrics import (WorkflowCancelled, extract_song_block, load_workflows,
                                parse_tag_list, run_workflow)

VOCAB = ["pop", "rock", "jazz", "electronic", "folk", "male vocal", "female vocal"]


def _spec():
    return load_workflows(load_config())["jp-title-first"]


async def _main() -> None:
    spec = _spec()

    # 1. happy path (the mock's refine reply wraps the song in commentary +
    #    a fence and carries a [Title] section — the failure shape observed
    #    on the real endpoint)
    r = await run_workflow(MockLLMClient(), spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    assert r.title == "A Window Far Away" and "dream pop" in r.style
    assert "Key Improvements" not in r.lyrics and "[Title]" not in r.lyrics
    assert "Sparks of Silence" not in r.lyrics and r.lyrics.startswith("[Intro]")
    assert "iambic" not in r.lyrics  # commentary stripped, song kept
    assert r.tags == ["vocal", "english", "rock", "pop", "male vocal"], r.tags
    print("1. happy path ok; stages:",
          [(s.stage_id, s.attempts, s.skipped) for s in r.stages])

    # 2a. translate gate retry with feedback (block stage: garbage fails the
    #     section-header gate, not an extract)
    mock = MockLLMClient(bad_first={"Translate the following song": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    tr = [t for t in r.attempts if t.stage_id == "translate"]
    assert len(tr) == 2 and not tr[0].ok and tr[0].failures[0].code == "missing_any"
    assert "section header" in tr[1].prompt_text
    print("2a. translate gate retry ok; feedback mentions:",
          [f.code for f in tr[0].failures])

    # 2b. a bad Name line retries ONLY the cheap name continuation
    mock = MockLLMClient(bad_first={"English name": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    nm = [t for t in r.attempts if t.stage_id == "name"]
    assert len(nm) == 2 and not nm[0].ok and nm[0].failures[0].code == "extract_failed"
    assert "Name:" in nm[1].prompt_text
    assert len([t for t in r.attempts if t.stage_id == "translate"]) == 1
    print("2b. name retry ok — translation ran exactly once")

    # 3. transport errors consume attempts; exhaustion fails the job
    mock = MockLLMClient(fail_first={"Brainstorm ideas": 2})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert not r.ok and "brainstorm" in r.error and "llm_error" in r.error
    print("3. llm_error exhaustion fails job:", r.error)

    # 4. resume: run to a stage, then resume from the returned progress
    mock = MockLLMClient()
    r1 = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                            tag_vocabulary=VOCAB, stop_after="translate")
    assert r1.ok
    calls_before = mock.calls
    r2 = await run_workflow(mock, spec, tag_vocabulary=VOCAB, progress=r1.progress)
    assert r2.ok, r2.error
    assert {s.stage_id for s in r2.stages if s.skipped} == \
        {"brainstorm", "foreign_lyrics", "translate"}
    assert mock.calls == calls_before + 4  # name, style, refine, tags only
    assert r2.title == "A Window Far Away"
    print("4. resume ok — no stage re-ran")

    # 5. guard cancellation between stages
    state = {"stop": False}

    class FlipAfter2:
        name = "flip"
        calls = 0

        async def chat(self, messages, **kw):
            FlipAfter2.calls += 1
            if FlipAfter2.calls >= 2:
                state["stop"] = True
            return await mock.chat(messages, **kw)

        async def status(self):
            return await mock.status()

    mock = MockLLMClient()
    try:
        await run_workflow(FlipAfter2(), spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB, guard=lambda: not state["stop"])
        raise AssertionError("expected WorkflowCancelled")
    except WorkflowCancelled as exc:
        print("5. guard cancellation ok:", exc)

    # 6. song-block extraction shapes (output-style nondeterminism)
    commentary_fence = ("Preamble:\n```\nnotes only\n```\nmore talk\n```\n[Intro]\nline\n\n"
                        "[Chorus]\nline\n```\nafter")
    assert extract_song_block(commentary_fence) == "[Intro]\nline\n\n[Chorus]\nline"
    bare = "Preamble text.\n\n[Title]\nSome Name\n\n[Verse 1]\nwords\n\n[Outro]"
    assert extract_song_block(bare) == "[Verse 1]\nwords\n\n[Outro]"
    titled_fence = "talk\n```text\n[Title]\nName Here\n\n[Intro]\nwords\n```"
    assert extract_song_block(titled_fence) == "[Intro]\nwords"
    assert extract_song_block("just commentary, no headers at all") == \
        "just commentary, no headers at all"
    assert "    indented line kept" in extract_song_block(
        "```\n[Intro]\n    indented line kept\n```")
    # The rhyme-guide specimen: leaked WIP draft, rhyme deliberation, then
    # the final song — the LAST header run wins. (Prose lines preserved as
    # single long lines, exactly as the endpoint emitted them: the run
    # splitter keys on long-lined prose between headers.)
    specimen = """\
[Verse]
Wheel, glass, steel
Ribs showing through, no light
No blinking, no feel
Nowhere to hide, no sight

[Outro]
The engine fades
One beat, one beat, a heartbeat almost still
Still moving
White road
No shadow

Let me check the outro. Original:
"The engine fades / One beat, one beat, a heartbeat almost still / Still moving / White road / No shadow"

ABABC:
A: -ade (fades, blade, made, shade)
B: -ill (still, hill, chill, fill)
A: -ade
B: -ill
C: -ow (road, shadow, go, low)

Wait, C should rhyme with... in ABABC, C is the 5th line and doesn't need to rhyme with anything (it's the final line). Actually in ABABC: A rhymes with A (lines 1,3), B rhymes with B (lines 2,4), C is standalone (line 5).

Let me work with the original words:

[Verse]
Chrome and sunlight
Steel on the horizon
Chrome and sunlight
Night without end

[Outro]
The engine fades to blue
One beat, one beat, a heartbeat almost still
Still moving
White road
No shadow"""
    got = extract_song_block(specimen)
    assert got.startswith("[Verse]\nChrome and sunlight"), got[:80]
    assert "Let me check" not in got and "ABABC" not in got
    assert "Wheel, glass, steel" not in got
    assert got.endswith("No shadow"), got[-60:]
    plain = "[Verse 1]\nshort lines\nmore lines\n\n[Verse 2]\nmore\nlines\n\n[Outro]\nend"
    assert extract_song_block(plain) == plain
    # The continuation-seam specimen: a truncated reply's continuation
    # RESTARTED the song (leaving a short thinking fragment between the
    # copies) instead of resuming — a late [Intro] marks the restart, and
    # the last run (the model's final copy) wins.
    seam = """\
[Intro]
Break, break, it's okay to break
Break, break, it's okay to break

[Verse 1]
holding words of the first copy

[Outro]
Break, break, it's okay to break

Hmm, let me re-examine. The user wants "a fast

[Intro]
Break, break, it's okay to break
Break, break, it's okay to break

[Verse 1]
final words of the second copy

[Outro]
the true end"""
    got = extract_song_block(seam)
    assert got.count("[Intro]") == 1, got[:80]
    assert got.startswith("[Intro]\nBreak, break, it's okay to break")
    assert "let me re-examine" not in got and "first copy" not in got
    assert got.endswith("the true end"), got[-40:]
    print("6. extract_song_block shapes ok (draft+deliberation -> last run; "
          "continuation-seam restart -> late-[Intro] split)")

    # 7. liberal song-name parsing (the #1 retry source on the real endpoint)
    from infiniradio.lyrics import _evaluate
    name_stage = next(s for s in spec.stages if s.id == "name")
    name = "Blinding Truth, Don't Stop"
    variants = [
        f"Name: `{name}`", f'Name: "{name}"', f"Name: '{name}'", f"Name: {name}",
        f"**Name:** `{name}`", f"- Name: {name}",
        f"Sure!\n\nName: {name}\nHope it helps.", f"Name: \u201c{name}\u201d",
    ]
    for v in variants:
        failures, extracted = _evaluate(name_stage, v, v)
        assert not failures, (v, failures)
        assert extracted["song_name"] == name, (v, extracted)
    failures, _ = _evaluate(name_stage, "The name of the song is unclear.", "x")
    assert failures and failures[0].code == "extract_failed"
    print("7. liberal name parsing ok (8 variants + prose non-match)")

    # 8. tag sanitization: thinking-leak reply -> clean/static picks only;
    #    filter-then-truncate in most-relevant-first order
    assert parse_tag_list('1. Rock\n- "dream pop"\n* \u2018male vocal\u2019') == \
        ["rock", "dream pop", "male vocal"]
    mock = MockLLMClient(bad_first={"tags": 1})  # first tags reply = garbage
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    assert r.tags == ["vocal", "english"], r.tags  # garbage -> statics only
    assert spec.output.max_tags is None or spec.output.max_tags >= 1
    from infiniradio.llm import mock as mock_mod
    old_entry = mock_mod.SCRIPT[-1]
    mock_mod.SCRIPT[-1] = ("tags", "rock, pop, jazz, folk, trance, male vocal")
    try:
        r = await run_workflow(MockLLMClient(), spec,
                               initial_vars={"title": "Glass Ember"}, tag_vocabulary=VOCAB)
        limit = spec.output.max_tags or 99
        picks = ["rock", "pop", "jazz", "folk", "trance", "male vocal"]
        expected_dyn = [p for p in picks if p in set(VOCAB)][:limit]
        assert r.tags == ["vocal", "english"] + expected_dyn, (r.tags, limit)
    finally:
        mock_mod.SCRIPT[-1] = old_entry
    print("8. tag sanitization ok (tolerance, leak -> statics, filter+truncate)")

    # 9. truncation: client reports it (any casing — servers emit "LENGTH");
    #    engine continues; exhaustion fails with code "truncated" and the
    #    feedback retry recovers. Per-stage knobs reach the client.
    from infiniradio.llm.openai_compat import OpenAICompatClient
    reply = OpenAICompatClient._parse(
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
         "usage": {"completion_tokens": 100}})
    assert reply.truncated and reply.content == "partial" and \
        reply.finish_reason == "length"
    upper = OpenAICompatClient._parse(
        {"choices": [{"message": {"content": "cut"}, "finish_reason": "LENGTH"}]})
    assert upper.truncated and upper.finish_reason == "LENGTH"  # the prod bug
    assert not OpenAICompatClient._parse(
        {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}).truncated

    mock = MockLLMClient(truncate_first={"Rewrite the words of this song": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    tr = [t for t in r.attempts if t.stage_id == "refine"]
    assert len(tr) == 1 and tr[0].ok and tr[0].continuations == 1
    assert tr[0].finish_reason == "stop"  # the final (successful) call's value
    assert tr[0].usage["completion_tokens"] > 60  # both calls' tokens summed
    assert "[Outro]" in r.lyrics and "Here's the revised" not in r.lyrics
    assert len(r.progress["threads"]["refine"]) == 4  # user, partial, continue, full
    print("9a. truncation continuation ok (any-casing detection, tokens summed)")

    mock = MockLLMClient(truncate_first={"Rewrite the words of this song": 9})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB, max_continuations=0)
    assert not r.ok and "truncated" in r.error, r.error
    assert r.lyrics == ""  # a partial reply NEVER becomes the song
    mock = MockLLMClient(truncate_first={"Rewrite the words of this song": 1})
    r = await run_workflow(mock, spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB, max_continuations=0)
    assert r.ok  # attempt 1 truncated -> feedback retry -> clean
    tr = [t for t in r.attempts if t.stage_id == "refine"]
    assert len(tr) == 2 and "cut off" in tr[1].prompt_text
    print("9b. truncation exhaustion -> 'truncated'; feedback retry recovers")

    # 10. per-stage generation knobs reach the client: max_tokens,
    #     reasoning_effort and arbitrary extra body parameters
    from dataclasses import replace as _replace

    tuned = _replace(spec, stages=tuple(
        _replace(s, max_tokens=2048, reasoning_effort="low",
                 extra={"enable_thinking": False}) if s.id == "brainstorm" else s
        for s in spec.stages))
    seen: list[dict] = []
    inner = MockLLMClient()

    class Recorder:
        name = "recorder"

        async def chat(self, messages, **kw):
            seen.append(dict(kw))
            return await inner.chat(messages, **kw)

    r = await run_workflow(Recorder(), tuned, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=VOCAB)
    assert r.ok, r.error
    assert seen[0]["max_tokens"] == 2048
    assert seen[0]["extra"] == {"enable_thinking": False, "reasoning_effort": "low"}
    others = [kw for kw in seen[1:]]
    assert all(kw.get("max_tokens") is None or kw["max_tokens"] == 12288
               for kw in others if kw)  # refine carries its own budget now
    assert all("reasoning_effort" not in (kw.get("extra") or {}) for kw in others)
    print("10. per-stage max_tokens/reasoning_effort/extra reach the client")

    # 11. empty tag vocabulary: the prompt degrades coherently (no phantom
    #     "from this list only"), picks sanitize to nothing, statics survive
    r = await run_workflow(MockLLMClient(), spec, initial_vars={"title": "Glass Ember"},
                           tag_vocabulary=[])
    assert r.ok, r.error
    tags_prompt = next(t.prompt_text for t in r.attempts if t.stage_id == "tags")
    assert "from this list only" not in tags_prompt
    assert "genre tags" in tags_prompt
    assert r.tags == ["vocal", "english"], r.tags
    print("11. empty vocabulary -> coherent fallback prompt, statics only")


def run() -> None:
    asyncio.run(_main())
    print("test_lyrics_engine: ALL PASSED")


if __name__ == "__main__":
    run()
