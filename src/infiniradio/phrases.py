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


def top_phrases(songs: list[str], *, n_min: int = 3, n_max: int = 8,
                min_docs: int = 2, limit: int = 60) -> list[dict]:
    """The most over-represented word n-grams across a lyric corpus.

    Ranked by document frequency — how many songs contain the phrase —
    with occurrence totals breaking ties. Counts are EXCLUSIVE: an
    occurrence that sits inside a listed longer phrase is attributed to
    the longer form. "i was a functioning" (1000 raw occurrences) keeps
    its row at 900 when 100 of them continue into "i was a functioning
    alcoholic" (listed at 100); a phrase whose every occurrence extends
    longer listed phrases has nothing left and drops out.
    """
    docs = [w for w in (normalize(s).split() for s in songs) if w]
    if not docs:
        return []
    # Pass 1: per-phrase doc sets and occurrence totals.
    grams: dict[tuple[str, ...], list] = {}
    for i, words in enumerate(docs):
        for n in range(n_min, n_max + 1):
            if len(words) < n:
                break
            for j in range(len(words) - n + 1):
                entry = grams.setdefault(tuple(words[j:j + n]), [set(), 0])
                entry[0].add(i)
                entry[1] += 1
    # Pass 2: occurrence positions, candidates only — a phrase below
    # min_docs can never be listed, so it can never cover anything.
    candidates = {p for p, (d, _t) in grams.items() if len(d) >= min_docs}
    positions: dict[tuple[str, ...], list[tuple[int, int]]] = {p: [] for p in candidates}
    for i, words in enumerate(docs):
        for n in range(n_min, n_max + 1):
            if len(words) < n:
                break
            for j in range(len(words) - n + 1):
                key = tuple(words[j:j + n])
                if key in positions:
                    positions[key].append((i, j))
    # Longest phrases settle first: each listed phrase's occurrences
    # cover every span they contain, and shorter phrases count only the
    # occurrences that remain uncovered.
    by_len: dict[int, list[tuple[str, ...]]] = {}
    for phrase in positions:
        by_len.setdefault(len(phrase), []).append(phrase)
    covered: dict[int, set[tuple[int, int]]] = {}   # song -> {(start, len)}
    rows: list[dict] = []
    for n in range(n_max, n_min - 1, -1):
        for phrase in by_len.get(n, ()):
            occ = positions[phrase]
            excl = [(s, j) for s, j in occ
                    if (j, n) not in covered.get(s, ())]
            excl_docs = {s for s, _ in excl}
            if len(excl_docs) < min_docs:
                continue
            rows.append({"phrase": " ".join(phrase), "songs": len(excl_docs),
                         "pct": round(100.0 * len(excl_docs) / len(docs), 1),
                         "total": len(excl)})
            # Mark with ALL occurrences (not just exclusive ones): an
            # occurrence covered by a longer listed phrase had its
            # sub-spans marked by that phrase already.
            for s, j in occ:
                marks = covered.setdefault(s, set())
                for length in range(n_min, n):
                    for i in range(j, j + n - length + 1):
                        marks.add((i, length))
    rows.sort(key=lambda r: (-r["songs"], -len(r["phrase"].split()), r["phrase"]))
    return rows[:limit]
