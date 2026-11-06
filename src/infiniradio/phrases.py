"""Lyric-phrase utilities shared by the admin phrase analysis, the library
lyrics search, and the workflow engine's banned-phrase gate.

One normalizer everywhere: a phrase the analysis flags, the search box
finds, and the generation gate blocks all match the same word sequence,
regardless of case, punctuation, section headers, or curly-vs-straight
apostrophes.
"""

from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")


def normalize(text: str | None) -> str:
    """Lyrics reduced to a comparable word sequence: lowercased, section
    header lines dropped, curly apostrophes straightened, other punctuation
    stripped (apostrophes stay — they are part of words), whitespace
    collapsed.

    "I'm a functioning alcoholic," -> "i'm a functioning alcoholic"
    """
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if not _HEADER_RE.match(ln)]
    text = " ".join(lines).replace("\u2019", "'").replace("\u2018", "'").lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word-sequence containment on normalized text: "functioning
    alcoholic" matches "...I'm a functioning alcoholic tonight..." but not
    "...dysfunctioning alcoholic" or "...functioning alcoholics" (inflected
    variants need their own entry in a ban list).
    """
    hay, needle = normalize(text), normalize(phrase)
    if not needle:
        return False
    return f" {needle} " in f" {hay} "


def top_phrases(songs: list[str], *, n_min: int = 2, n_max: int = 8,
                min_docs: int = 2, limit: int = 60) -> list[dict]:
    """The most over-represented word n-grams across a lyric corpus.

    Ranked by document frequency — how many songs contain the phrase —
    because a phrase appearing in many songs is the tell of a memorized
    real-world line; occurrence totals break ties. Sub-grams collapse:
    when every song containing a phrase also contains the same longer
    phrase, only the longest maximal phrase is kept, so the table shows
    "i'm a functioning alcoholic" rather than each of its fragments.
    """
    docs = [w for w in (normalize(s).split() for s in songs) if w]
    if not docs:
        return []
    # grams[n]: phrase (tuple of words) -> [song indexes containing it,
    # total occurrences]
    grams: dict[int, dict[tuple[str, ...], list]] = {}
    for i, words in enumerate(docs):
        for n in range(n_min, n_max + 1):
            if len(words) < n:
                break
            level = grams.setdefault(n, {})
            for j in range(len(words) - n + 1):
                entry = level.setdefault(tuple(words[j:j + n]), [set(), 0])
                entry[0].add(i)
                entry[1] += 1
    # A phrase is subsumed when a one-word-longer phrase containing it
    # covers all of its songs: the longer phrase is the maximal form.
    subsumed: dict[int, set] = {n: set() for n in range(n_min, n_max + 1)}
    for n in range(n_min, n_max):
        upper, lower = grams.get(n + 1, {}), grams.get(n, {})
        for phrase, (docs_, _total) in upper.items():
            for sub in (phrase[1:], phrase[:-1]):
                entry = lower.get(sub)
                if entry and entry[0] <= docs_:
                    subsumed[n].add(sub)
    out: list[dict] = []
    for n in range(n_min, n_max + 1):
        for phrase, (docs_, total) in grams.get(n, {}).items():
            if phrase in subsumed[n] or len(docs_) < min_docs:
                continue
            out.append({"phrase": " ".join(phrase), "songs": len(docs_),
                        "pct": round(100.0 * len(docs_) / len(docs), 1),
                        "total": total})
    out.sort(key=lambda r: (-r["songs"], -len(r["phrase"].split()), r["phrase"]))
    return out[:limit]
