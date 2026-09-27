#!/usr/bin/env python3
"""Filter a word list by word frequency.

Loads a word-frequency CSV (``word,count``) and a word list (one entry per
line; ``#`` comments and blank lines allowed), then writes the word list
with all words whose log-probability of occurrence is below a cutoff
removed.  Comments and blank lines are preserved verbatim; a comment
attached to a removed word is removed with it.

A tuning mode instead reports the borderline words around the cutoff so
the cutoff can be chosen sensibly, and a sorting mode reorders the kept
words from most to least common.
"""

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

LOG = logging.getLogger("freqfilter")


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "freq_csv",
        type=Path,
        help="CSV file with header 'word,count' of corpus frequencies.",
    )
    parser.add_argument(
        "word_list",
        type=Path,
        help="Word list file; '#' comments and blank lines are allowed.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Where to write the filtered list (default: stdout).",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        required=True,
        help="Log10 probability cutoff: words with log10(count/total) "
             "below this value are removed (e.g. -8.5).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Instead of filtering, report the rarest N words that would "
             "be kept and the most common N words that would be dropped.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the emitted words most common to least common; "
             "comments and blank lines are moved to the top (keeping "
             "their order) and comments attached to words follow their "
             "words.  Ties keep file order.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of borderline words to report in --tune mode "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def load_frequencies(path):
    """Load a 'word,count' CSV and return {word: log10 probability}."""
    counts = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["word", "count"]:
            LOG.warning("Unexpected CSV header %r; expected ['word', "
                        "'count']", header)
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                word, count = row[0], int(row[1])
            except (IndexError, ValueError):
                LOG.warning("Skipping malformed CSV line %d: %r",
                            line_no, row)
                continue
            counts[word] = counts.get(word, 0) + count

    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"No valid counts found in {path}")
    return {word: math.log10(count / total) for word, count in counts.items()}


def load_word_list(path):
    """Parse the word list into entries, preserving original text.

    Each entry is a dict with a 'kind' key ('word', 'comment', or
    'blank'), the original line text under 'line', and, for words, the
    word itself under 'word'.  A comment on the same line as a word is
    carried by that word's entry, so it is removed if the word is
    removed.
    """
    entries = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                entries.append({"kind": "blank", "line": line})
            elif stripped.startswith("#"):
                entries.append({"kind": "comment", "line": line})
            else:
                word, _, _comment = line.partition("#")
                word = word.strip()
                if not word:
                    entries.append({"kind": "comment", "line": line})
                else:
                    entries.append({"kind": "word", "line": line,
                                    "word": word})
    return entries


def tune(log_probs, entries, cutoff, n):
    """Report the rarest kept and most common dropped words near cutoff."""
    words = [e["word"] for e in entries if e["kind"] == "word"]
    kept = [(w, log_probs[w]) for w in words if log_probs.get(w, -math.inf)
            >= cutoff]
    dropped = [(w, log_probs.get(w, -math.inf)) for w in words
               if log_probs.get(w, -math.inf) < cutoff]

    kept.sort(key=lambda item: item[1])
    dropped.sort(key=lambda item: item[1], reverse=True)

    print(f"Cutoff: log10(p) >= {cutoff}")
    print(f"Kept: {len(kept)} words; dropped: {len(dropped)} words")
    print(f"\nRarest {min(n, len(kept))} words that would be KEPT:")
    for word, lp in kept[:n]:
        print(f"  {word:<20} log10(p) = {lp:.4f}")
    print(f"\nMost common {min(n, len(dropped))} words that would be "
          f"DROPPED:")
    for word, lp in dropped[:n]:
        print(f"  {word:<20} log10(p) = {lp:.4f}")


def filter_entries(log_probs, entries, cutoff):
    """Drop word entries below the cutoff; keep everything else."""
    kept = []
    for entry in entries:
        if entry["kind"] != "word":
            kept.append(entry)
        elif log_probs.get(entry["word"], -math.inf) >= cutoff:
            kept.append(entry)
    return kept


def sort_entries(log_probs, entries):
    """Move comments/blanks to the top; sort words by frequency.

    Words are sorted most common to least common (descending log
    probability); ties keep their file order because the sort is
    stable.  Words with no frequency entry sort last.  A word's
    original line (including any attached comment) travels with it.
    """
    preamble = [e for e in entries if e["kind"] != "word"]
    words = [e for e in entries if e["kind"] == "word"]
    words.sort(key=lambda e: -log_probs.get(e["word"], -math.inf))
    return preamble + words


def write_entries(entries, output):
    """Write entries to output (a path or None for stdout)."""
    out = open(output, "w", encoding="utf-8") if output else sys.stdout
    try:
        for entry in entries:
            out.write(entry["line"] + "\n")
    finally:
        if output:
            out.close()


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    log_probs = load_frequencies(args.freq_csv)
    entries = load_word_list(args.word_list)
    LOG.debug("Loaded %d frequency entries, %d list lines",
              len(log_probs), len(entries))

    if args.tune:
        tune(log_probs, entries, args.cutoff, args.n)
        return 0

    kept = filter_entries(log_probs, entries, args.cutoff)
    if args.sort:
        kept = sort_entries(log_probs, kept)
    write_entries(kept, args.output)
    n_words = sum(1 for e in kept if e["kind"] == "word")
    LOG.info("Kept %d words (cutoff log10(p) >= %g)", n_words, args.cutoff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
