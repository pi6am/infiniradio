"""Mock LLM client: canned, gate-shaped responses for pipeline testing.

Testing only — reachable exclusively through the --mockllm command line
flag (or the lyrics audition CLI's --mock), never through config, mirroring
the mock generator's --mockaudio rule. Exercising the whole lyrics pipeline
(DJ commissioning, stage retries, gates, job phases, ingestion) must not
require a GPU or an LLM endpoint.

Responses are selected by keyword-matching the last user message, so the
mock follows a workflow through its stages. The canned texts are shaped to
pass the reference workflows' gates and extractions; ``fail_first`` and
``bad_first`` script transport failures / bad outputs for the first N
attempts of a stage (keyed by the same keywords) to exercise the retry
paths deterministically.
"""

from __future__ import annotations

import random
from typing import Any

from .base import LLMClient, LLMError, Message, Reply

# (keyword in the last user message, reply) — first match wins. Keywords
# taken from the reference workflows' templates. The en-native "write"
# needle must precede the foreign-draft needle: the write prompt opens
# "Use this guide to write a song in English:", which the foreign needle
# would otherwise match first.
SCRIPT: list[tuple[str, str]] = [
    ("Brainstorm ideas", """\
Theme and progression (no lyrics):

The title is treated metaphorically: not a literal scene but a state of
mind — wanting something you can only see from a distance. Verse 1 sets the
ordinary world and the quiet dissatisfaction in it. Verse 2 introduces the
second character/force and the pull between them. The chorus is the emotional
spike: a request spoken too late. Bridge strips the arrangement down to the
one honest sentence the verses avoided. Outro returns to the opening image,
now understood differently."""),
    ("write a song in English", """\
[Intro]

[Verse 1]
In a quiet room I unfold a map
Searching for the name of a town I have never seen

[Chorus]
Beyond the window a light is swaying
Night breaks while it stays out of reach

[Bridge]
A small voice carried by the wind

[Outro]

[Instrumental]"""),
    ("Use this guide to write a song", """\
タイトル：遠い窓

[Intro]

[Verse 1]
静かな部屋で　地図を広げる
行ったことのない　街の名を探す

[Chorus]
窓の向こうに　灯りが揺れる
届かないまま　夜が明ける

[Verse 2]
時刻表には　書いていない場所
折り畳んだ地図に　印をつける

[Bridge]
風が運ぶ　小さな声

[Outro]

[Instrumental]"""),
    ("Translate the following song", """\
```text
[Intro]

[Verse 1]
In a quiet room I unfold a map
Searching for the name of a town I have never seen

[Chorus]
Beyond the window a light is swaying
Night breaks while it stays out of reach

[Verse 2]
A place the timetable never lists
I mark it on the folded map

[Bridge]
A small voice carried by the wind

[Outro]

[Instrumental]
```"""),
    ("English name", "Name: `A Window Far Away`"),
    ("Write a Style Prompt", """\
Style prompt:

```
English, warm female vocal, dream pop, 96 BPM, padded synths, rounded
electric bass, brushed drums, soft dynamics
```"""),
    ("Rewrite the words of this song", """\
Here's the revised version with improved iambic rhythm and evocative sparsity:

```
[Title]
Sparks of Silence

[Intro]
A quiet room. A map unfolds.

[Verse 1]
I hunt a town no road has named.

[Chorus]
Past the glass a lantern sways.

[Bridge]
A small voice rides the wind.

[Outro]
```

**Key Improvements:**
1. **Iambic Flow:** adjusted phrasing so emphasis follows the meter.
2. **Sparsity:** removed redundant words for a tighter rhythm.
The revision keeps the song's essence while sharpening its scan."""),
    # The single-stage songwriter workflow: the reply is the fine-tuned
    # model's trained shape (# heading + ```style/```lyrics fences). The
    # needle must not collide with the en-native/foreign draft needles
    # ("Write a song about" vs "write a song in English" / "Use this guide
    # to write a song") — checked, it does not.
    ("Write a song about", """\
# Velveeta

```style
A cheesy, upbeat electropop track with 80s-influenced synthwave textures and playful production. Features bright, processed synth pads, drum machines, and retro digital effects. The verses maintain a light, bouncy feel with minimal instrumentation, building into choruses with layered vocals and added synth elements. Bridge section incorporates pitch-shifted vocals and wider stereo imaging for enhanced nostalgia. Production intentionally references 80s commercial jingles with its polished, artificial quality while maintaining modern pop structure and arrangement.
```

```lyrics
[verse 1]
If you want a taste of heaven
A warm and gooey sensation
There's only one cheese that I've found
That's smooth, creamy, all around
Oh, Velveeta

[chorus]
Melt it, melt it, watch it flow
Golden river, soft and slow
Nothing fancy, nothing cured
Just the melt that I prefer
Oh, Velveeta

[verse 2]
Blue box glowing in the pantry light
Shelf-stable promise, forever bright
While the artisans age their wheel
I have got my copper-kettle seal

[bridge]
It's not cheese, they say
But it melts that way

[outro]
Oh, Velveeta
Oh, Velveeta
```"""),
    # Art workflows (art_workflows/): same keyword-follows-the-stage
    # scheme. Must precede the generic "tags" needle — the art templates
    # list track tags ("Tags: ...") and would otherwise resolve to the
    # lyrics tagging reply.
    ("art director of an internet radio", """\
A single stained-glass cathedral silhouette rising out of a neon-lit rain
puddle on an empty street at night; the reflection carries the color while
the building above stays dark. Cool blues and magentas, one warm amber
window. Centered, symmetrical, poster-like calm."""),
    ("Turn that concept into the final image-generation prompt", """\
```text
A dark cathedral silhouette in flat-color illustration with screen-print
texture, centered and symmetrical, rising from a rain puddle that mirrors
it in neon magenta and electric blue; one warm amber window glows at its
heart; limited palette of five inks, crisp edges, subtle grain, night
mood, square album-cover composition, no text or lettering
```"""),
    ("image-generation prompts for an internet radio", """\
```text
A rain-slicked empty street at night shot on 35mm: a lone figure under a
bus shelter, neon signs reflected as long streaks of magenta and cyan in
the puddles, distant traffic lights as warm bokeh, gentle film grain,
wide centered composition with the figure small against the glowing city,
cinematic night mood, no text or lettering
```"""),
    ("tags", "rock, pop, male vocal"),
]

FALLBACK = "(mock reply)"


class MockLLMClient(LLMClient):
    name = "mock"

    def __init__(self, fail_first: dict[str, int] | None = None,
                 bad_first: dict[str, int] | None = None,
                 truncate_first: dict[str, int] | None = None):
        self.fail_first = {k.lower(): n for k, n in (fail_first or {}).items()}
        self.bad_first = {k.lower(): n for k, n in (bad_first or {}).items()}
        # Stage keyword -> number of initial replies to return TRUNCATED
        # (a prefix of the canned text, finish_reason=length). The
        # continuation turn that follows resolves via the sticky keyword to
        # the full canned reply — exercising the engine's continuation path.
        self.truncate_first = {k.lower(): n for k, n in (truncate_first or {}).items()}
        self.calls = 0
        self.last_extra: dict = {}
        self.convs: list[list[Message]] = []  # message list per chat() call —
        # tests assert conversation shape (fresh vs continued retries)
        self._sticky: str | None = None  # last matched stage keyword

    def _keyword(self, messages: list[Message]) -> str | None:
        # Scan from the newest message backwards so retries (whose newest
        # turn is the feedback message) still resolve to their stage, and a
        # "continue" stage (whose thread contains the previous stage's
        # prompt) resolves to its own, newest prompt.
        for msg in reversed(messages):
            for needle, _ in SCRIPT:
                if needle.lower() in msg.content.lower():
                    return needle
        return None

    async def chat(self, messages: list[Message], *, model: str | None = None,
                   temperature: float | None = None,
                   max_tokens: int | None = None,
                   extra: dict | None = None) -> Reply:
        self.calls += 1
        self.last_extra = dict(extra or {})  # recorded for tests/diagnostics
        self.convs.append(list(messages))
        key = self._keyword(messages)
        if key is not None:
            self._sticky = key
        else:
            # Retry prompts (feedback turns) name no stage; stay on the one
            # the conversation is already in.
            key = self._sticky
        # Transport-level scripted failure: LLMError, as a dead endpoint would raise.
        if key is not None and self.fail_first.get(key.lower(), 0) > 0:
            self.fail_first[key.lower()] -= 1
            raise LLMError("mock: scripted transport failure")
        if key is not None and self.bad_first.get(key.lower(), 0) > 0:
            self.bad_first[key.lower()] -= 1
            return Reply(content="(mock garbage that fails gates)",
                         model=model or "mock",
                         usage={"prompt_tokens": 10, "completion_tokens": 6})
        if key is not None and self.truncate_first.get(key.lower(), 0) > 0:
            self.truncate_first[key.lower()] -= 1
            content = next((reply for needle, reply in SCRIPT if needle == key), FALLBACK)
            return Reply(content=content[:60], truncated=True, model=model or "mock",
                         finish_reason="length",
                         usage={"prompt_tokens": 10, "completion_tokens": 60})
        content = next((reply for needle, reply in SCRIPT if needle == key), FALLBACK)
        return Reply(content=content, model=model or "mock", finish_reason="stop",
                     usage={"prompt_tokens": len(messages[-1].content) // 4,
                            "completion_tokens": len(content) // 4
                            + random.randint(0, 20)})

    async def status(self) -> dict[str, Any]:
        return {"backend": self.name, "ok": True, "model": "mock",
                "calls": self.calls}
