"""Track title generation: word-list files + template patterns.

Completely random (and seedable) by design. An LLM namer was considered and
deliberately deferred: models randomize poorly, while word-slot templates
give predictable grammar with unpredictable imagery — and zero inference
cost. Titles keep template literals lowercase and Title-Case only the
slotted words ("Cathedral of Glass", never "Cathedral Of Glass").

Word lists live as plain text files (default: ``names/`` in the repo, see
``naming.words_dir``): one file per slot, one word per line. ``#`` starts a
comment (stripped, then whitespace trimmed) and blank lines are skipped.
Lists are loaded once at startup — no hot reload — and validated eagerly:
missing/empty files and non-alphabetic words fail fast with a clear
message. A word may live in more than one list (orange the noun and
adjective is fine); only duplicates *within* a file are collapsed.

The lists are curated for common, recognizable, neutral English — no person
names, trademarks, profanity or obscure vocabulary, but generous enough
that the names keep surprising.

Audition with (from the repo root):

    python -m infiniradio.naming --n 60 --seed 3
    python -m infiniradio.naming --template "{noun} of {noun}" -n 20
"""

from __future__ import annotations

import random
import re
from pathlib import Path

DEFAULT_WORDS_DIR = "names"

# slot name -> file inside the words dir
LIST_FILES = {
    "noun": "nouns.txt",
    "adjective": "adjectives.txt",
    "verb": "verbs.txt",
    "verb3s": "verbs_third.txt",
    "adverb": "adverbs.txt",
}

# (weight, template) — literals stay exactly as written (lowercase), only
# slotted words are capitalized.
TEMPLATES: list[tuple[int, str]] = [
    (30, "{adjective} {noun}"),
    ( 5, "{noun} of {noun}"),
    ( 1, "{noun} or {noun}"),
    ( 3, "{noun} and {noun}"),
    ( 1, "{noun} & {noun}"),
    ( 1, "{noun} / {noun}"),
    ( 1, "{noun}+{noun}"),
    (10, "{adjective} and {adjective}"),
    ( 3, "{adjective} & {adjective}"),
    ( 1, "{adjective}+{adjective}"),
    (20, "{verb} {adverb}"),
    ( 5, "{verb} the {noun}"),
    (10, "{noun} {verb3s}"),
    ( 5, "{noun} {verb3s} {adverb}"),
    ( 2, "{adverb} or {adverb}"),
    ( 2, "{verb} or {verb}"),
    (20, "{adjective} {noun} {verb3s}"),
    ( 1, "{adjective} {noun} {verb3s} {adverb}"),
]

_SLOT_RE = re.compile(r"\{(\w+)\}")

_lists: dict[str, list[str]] | None = None


class NamingError(RuntimeError):
    """Bad word-list configuration; the server must not start."""


# ------------------------------------------------------------------ loading

def parse_words(text: str) -> list[str]:
    """Words from list-file text: '#' comments stripped, then trimmed,
    blank lines skipped, lowercased, in-file duplicates collapsed."""
    words: list[str] = []
    for line in text.splitlines():
        word = line.split("#", 1)[0].strip().lower()
        if word and word not in words:
            words.append(word)
    return words


# How many words a template may draw from one slot ("{noun} and {noun}" -> 2);
# lists must be at least this large or sampling would fail.
_MAX_SLOT_USE = {
    slot: max((tpl.count("{" + slot + "}") for _, tpl in TEMPLATES), default=1)
    for slot in LIST_FILES
}


def load_lists(words_dir: str | Path | None = None) -> dict[str, list[str]]:
    """Load the word lists once and cache them (no hot reload).

    Raises NamingError with a specific, actionable message for missing or
    empty files, non-alphabetic words, and lists too small for the
    templates (a template like "{noun} and {noun}" needs two nouns).
    Words may overlap between lists.
    """
    global _lists
    if _lists is not None:
        return _lists
    directory = Path(words_dir) if words_dir is not None else Path(DEFAULT_WORDS_DIR)
    loaded: dict[str, list[str]] = {}
    for slot, filename in LIST_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise NamingError(f"word list file missing: {path}")
        words = parse_words(path.read_text(encoding="utf-8"))
        if not words:
            raise NamingError(f"word list file is empty: {path}")
        if len(words) < _MAX_SLOT_USE[slot]:
            raise NamingError(
                f"{path}: only {len(words)} word(s); template patterns draw "
                f"up to {_MAX_SLOT_USE[slot]} distinct words from one list")
        for word in words:
            if not word.isalpha():
                raise NamingError(f"{path}: {word!r} is not a single alphabetic word")
        loaded[slot] = words
    _lists = loaded
    return _lists


def slots() -> dict[str, list[str]]:
    """The loaded word lists (lazily loads from the default dir)."""
    return load_lists()


# ---------------------------------------------------------------- generation

def generate(rng: random.Random | None = None,
             template: str | None = None) -> tuple[str, str]:
    """(template, title): one random title.

    Repeated slots of the same type sample without replacement, so
    "{noun} and {noun}" can never render "Ember and Ember".
    """
    rng = rng or random
    if template is None:
        weights, templates = zip(*TEMPLATES)
        template = rng.choices(list(templates), weights=list(weights), k=1)[0]
    word_lists = slots()
    picked: dict[str, list[str]] = {}
    for slot in set(_SLOT_RE.findall(template)):
        pool = word_lists.get(slot)
        if pool is None:
            raise ValueError(f"unknown template slot {{{slot}}}")
        n = template.count("{" + slot + "}")
        picked[slot] = [w.capitalize() for w in rng.sample(pool, n)]
    out = template
    for slot, words in picked.items():
        for w in words:
            out = out.replace("{" + slot + "}", w, 1)
    return template, out


def title(rng: random.Random | None = None, template: str | None = None) -> str:
    return generate(rng, template)[1]


def unique_title(db, rng: random.Random | None = None, tries: int = 8) -> str:
    """title() retried until no existing track carries it (best effort —
    after `tries` attempts the collision is accepted; with hundreds of
    words per slot that is astronomically unlikely)."""
    rng = rng or random
    t = title(rng)
    for _ in range(tries):
        if db.one("SELECT 1 FROM tracks WHERE title = ? LIMIT 1", (t,)) is None:
            return t
        t = title(rng)
    return t


# ----------------------------------------------------------------- audition

if __name__ == "__main__":
    import argparse
    import sys
    from collections import Counter

    ap = argparse.ArgumentParser(
        description="audition random track titles (no server involved)")
    ap.add_argument("-n", "--n", type=int, default=40,
                    help="how many titles to print (default 40)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for reproducible output")
    ap.add_argument("--template", default=None,
                    help='force one pattern, e.g. "{noun} of {noun}"')
    ap.add_argument("--words", default=DEFAULT_WORDS_DIR,
                    help="word-list directory (default: ./names)")
    args = ap.parse_args()

    try:
        load_lists(args.words)
    except NamingError as exc:
        sys.exit(f"error: {exc}")

    rng = random.Random(args.seed)
    titles, counts = [], Counter()
    for _ in range(max(1, args.n)):
        tpl, t = generate(rng, args.template)
        titles.append(t)
        counts[tpl] += 1
    for i, t in enumerate(titles, 1):
        print(f"{i:3d}  {t}")
    print(f"\n{len(titles)} titles, {len(set(titles))} unique")
    if args.template is None:
        print("template distribution:")
        for tpl, n in counts.most_common():
            print(f"  {n:4d}  {tpl}")
