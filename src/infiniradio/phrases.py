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


def top_phrases(songs: list[str], *, n_min: int = 3,
                min_docs: int = 2, limit: int = 60) -> list[dict]:
    """The most over-represented word phrases (3+ words, no length cap)
    across a lyric corpus.

    Ranked by document frequency — how many songs contain the phrase —
    with occurrence totals breaking ties. Counts are EXCLUSIVE: an
    occurrence that sits inside a longer phrase VISIBLE in the returned
    table is attributed to the longer form. "i was a functioning" (1000
    raw occurrences) keeps its row at 900 when 100 of them continue into
    "i was a functioning alcoholic" (also listed, at 100); a phrase whose
    every occurrence extends longer LISTED phrases has nothing left and
    drops out. Absorption never comes from rows outside the table, so a
    visible count is always explainable from the table itself — and when
    no visible row extends a phrase, its count matches what a library
    phrase search finds.

    Repeated phrases are mined level-wise — every repeated k-gram is a
    right-extension of its repeated (k-1)-word prefix, so candidates grow
    upward from n_min until repetition dies. A memorized line therefore
    surfaces once at its full shared length, not as sliding windows.
    Conceptually this is a counted successor tree of words: each repeated
    node carries its occurrence positions, displayed nodes "emit" their
    occurrences, and every other occurrence rolls up to the deepest
    displayed phrase CONTAINING it (containment, not just trie ancestry —
    "supposed to leave" sits inside "i was supposed to leave", which is
    not its descendant). Phrases may span line breaks; a repeat shared
    across three lines is found whole.
    """
    docs = [w for w in (normalize(s).split() for s in songs) if w]
    if not docs:
        return []

    # Level n_min: one full counting scan, then a positions pass that
    # keeps only repeated phrases (anything rarer can never be listed).
    seen: dict[tuple[str, ...], set[int]] = {}
    for si, words in enumerate(docs):
        for j in range(len(words) - n_min + 1):
            seen.setdefault(tuple(words[j:j + n_min]), set()).add(si)
    repeated: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for si, words in enumerate(docs):
        for j in range(len(words) - n_min + 1):
            key = tuple(words[j:j + n_min])
            if len(seen[key]) >= min_docs:
                repeated.setdefault(key, []).append((si, j))
    del seen

    # Upward: extend each repeated phrase's occurrences by one word. The
    # same (k-gram, position) is produced exactly once (from its prefix),
    # so positions never duplicate.
    by_len: dict[int, dict[tuple[str, ...], list[tuple[int, int]]]] = {}
    k = n_min
    while repeated:
        by_len[k] = repeated
        k += 1
        nxt: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for phrase, occ in repeated.items():
            for si, j in occ:
                words = docs[si]
                if j + k - 1 < len(words):
                    nxt.setdefault(phrase + (words[j + k - 1],), []).append((si, j))
        repeated = {p: occ for p, occ in nxt.items()
                    if len({s for s, _ in occ}) >= min_docs}

    # Longest phrases settle first within each pass: a displayed phrase's
    # occurrences cover every span they contain, and shorter phrases count
    # only the occurrences that remain uncovered. Only DISPLAYED phrases
    # absorb — a row the table doesn't show must never silently eat a
    # visible row's count — so passes repeat until the displayed set is
    # stable (typically two: full absorption, then restricted).
    def _pass(absorbers: frozenset[tuple[str, ...]]) -> list[tuple[dict, tuple]]:
        covered: dict[int, set[tuple[int, int]]] = {}   # song -> {(start, len)}
        rows: list[tuple[dict, tuple]] = []
        for n in sorted(by_len, reverse=True):
            for phrase, occ in by_len[n].items():
                excl = [(s, j) for s, j in occ
                        if (j, n) not in covered.get(s, ())]
                excl_docs = {s for s, _ in excl}
                if len(excl_docs) < min_docs:
                    continue
                rows.append((
                    {"phrase": " ".join(phrase), "songs": len(excl_docs),
                     "pct": round(100.0 * len(excl_docs) / len(docs), 1),
                     "total": len(excl)},
                    phrase))
                if phrase not in absorbers:
                    continue
                # Mark with ALL occurrences (not just exclusive ones): an
                # occurrence covered by a longer displayed phrase had its
                # sub-spans marked by that phrase already.
                for s, j in occ:
                    marks = covered.setdefault(s, set())
                    for length in range(n_min, n):
                        for i in range(j, j + n - length + 1):
                            marks.add((i, length))
        rows.sort(key=lambda r: (-r[0]["songs"],
                                 -len(r[1]), r[0]["phrase"]))
        return rows

    absorbers = frozenset(p for level in by_len.values() for p in level)
    rows: list[tuple[dict, tuple]] = []
    for _ in range(10):
        rows = _pass(absorbers)
        top = {phrase for _row, phrase in rows[:limit]}
        if top == absorbers:
            break
        absorbers = top
    else:
        # No fixpoint within the cap (theoretical): emit one last pass
        # consistent with the final displayed set.
        rows = _pass(absorbers)
    return [row for row, _phrase in rows[:limit]]
