"""ABC score sanity gate.

Fresh songs generate in two passes: ABC planning is cheap (text-only), the
audio render is expensive — so the score is screened between them.

Deliberately narrow: the gate only catches pathological scores — token-cap
exhaustion (a planner that pads with bars until its budget runs out) and
malformed output. It does NOT judge musical quality: repetition statistics
were tried as gates and rejected, because they don't predict listener
ratings — the station's top-voted and hand-protected songs are often highly
repetitive (repetition is genre-flavored, not a defect). See
``--vs-votes`` below before adding any new gate.

Calibrate against the library any time with:

    python -m infiniradio.abc_audit --library          # audit every score
    python -m infiniradio.abc_audit --library --vs-votes   # stats vs ratings
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# A note-ish character in ABC body lines: pitches a-g/A-G and the rest z.
_NOTE = re.compile(r"[a-gA-Gz]")
# Information line ("X:1", "T:title", "%%directive", ...) vs body.
_HEADER = re.compile(r"^(?:[A-Za-z]:|%%)")

# Rough note density of the YuE2 render, for the estimated-duration report
# only (measured ~5 notes/s across the station library; it is NOT a gate —
# note density varies too much between songs to discriminate runaways).
NOTES_PER_SECOND = 5.0


@dataclass(frozen=True)
class Thresholds:
    # Pairs with the workflow's max_abc_tokens (8192): a generous token
    # budget lets a looping planner run long enough to reveal itself, and
    # this bound — set BELOW the cap's char equivalent, ABOVE the good-song
    # mass (library median ~2400 chars) — stops it before the render.
    # Historical trade-off at 5000: rejects three unrated 300s fillers plus
    # two protected songs (#23, #48) the station accepted losing.
    max_chars: int = 5000
    min_chars: int = 300           # below this the score is malformed/trivial


def body_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not _HEADER.match(ln.strip())]


def abc_stats(text: str) -> dict:
    lines = body_lines(text)
    counts = Counter(lines)
    notes = sum(len(_NOTE.findall(ln)) for ln in lines)
    dup = sum(v - 1 for v in counts.values())
    return {
        "chars": len(text),
        "body_lines": len(lines),
        "dup_ratio": (dup / len(lines)) if lines else 1.0,
        "max_line_repeats": max(counts.values(), default=0),
        "est_duration_s": round(notes / NOTES_PER_SECOND),
    }


def audit_abc(text: str, th: Thresholds | None = None) -> tuple[bool, str]:
    """(ok, reason). reason is empty when ok, human-readable when not."""
    th = th or Thresholds()
    if len(text) < th.min_chars:
        return False, f"score too short ({len(text)} chars < {th.min_chars})"
    if not re.search(r"(?m)^X:", text):
        return False, "no X: header — not a complete ABC score"
    s = abc_stats(text)
    if s["chars"] > th.max_chars:
        return False, (f"{s['chars']} chars > {th.max_chars} "
                       "(token-budget exhaustion)")
    return True, ""


# ------------------------------------------------------------------ CLI

def _wilson(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 0.0
    z = 1.96
    p = up / n
    return (p + z * z / (2 * n)
            - z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / (1 + z * z / n)


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def main() -> None:
    import argparse
    import sqlite3

    from .config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m infiniradio.abc_audit",
        description="Screen ABC scores against the sanity gate, and analyze "
                    "score statistics against listener ratings.")
    ap.add_argument("paths", nargs="*", help="score files to audit")
    ap.add_argument("--library", nargs="?", const="data", default=None,
                    help="audit every track in a station data dir (default: data)")
    ap.add_argument("--vs-votes", action="store_true",
                    help="with --library: report how score stats correlate with"
                         " listener votes and which protected songs each"
                         " candidate statistic would reject")
    ap.add_argument("--max-chars", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    th = Thresholds(
        max_chars=args.max_chars if args.max_chars is not None
        else int(cfg.get("abc_audit.max_chars", 10000)),
    )

    files: list[tuple[str, Path]] = []
    votes: dict[str, tuple[int, int, int]] = {}  # label -> (up, down, protected)
    if args.library:
        # Read-only: safe to point at a running station's data dir.
        con = sqlite3.connect(f"file:{Path(args.library) / 'infiniradio.db'}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for r in con.execute(
                "SELECT id, duration_s, vote_up, vote_down, protected FROM tracks"
                " WHERE status != 'deleted' AND abc_text != ''"):
            files.append((f"#{r['id']} ({r['duration_s'] or 0:.0f}s)",
                          Path(args.library) / "music" / str(r["id"]) / "score.abc"))
            votes[f"#{r['id']}"] = (r["vote_up"] or 0, r["vote_down"] or 0,
                                    int(r["protected"] or 0))
        con.close()
    else:
        files = [(str(p), p) for p in map(Path, args.paths)]

    if not files:
        ap.error("nothing to audit: pass score files or --library")

    rows = []
    for label, path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"{label:>16}  unreadable: {exc}")
            continue
        rows.append((label, text, abc_stats(text), audit_abc(text, th)))

    if args.vs_votes:
        print(f"{'score':>16} {'wil':>5} {'prot':>4} {'chars':>6} {'rep':>4} "
              f"{'dup%':>4} {'est_s':>5}  verdict")
        scored = []
        for label, text, s, (ok, why) in rows:
            u, d, prot = votes.get(label.split(" ")[0], (0, 0, 0))
            w = _wilson(u, d)
            scored.append((w, prot, s))
            print(f"{label:>16} {w:>5.2f} {'yes' if prot else '-':>4} "
                  f"{s['chars']:>6} {s['max_line_repeats']:>4} "
                  f"{s['dup_ratio']*100:>4.0f}% {s['est_duration_s']:>5}  "
                  f"{'ok' if ok else 'REJECT: ' + why}")
        ws = [r[0] for r in scored]
        print(f"\ncorrelation with Wilson score (n={len(scored)}):")
        for key, vals in (("chars", [r[2]["chars"] for r in scored]),
                          ("max_line_repeats", [r[2]["max_line_repeats"] for r in scored]),
                          ("dup_ratio", [r[2]["dup_ratio"] for r in scored]),
                          ("est_duration_s", [r[2]["est_duration_s"] for r in scored])):
            print(f"  {key:16} {_corr(vals, ws):+.2f}")
        prot_rows = [r for r in scored if r[1]]

        def _is_prot(label: str) -> bool:
            return bool(votes.get(label.split(" ")[0], (0, 0, 0))[2])

        n_rej_prot = [label for label, _, _s, (ok, _w) in rows
                      if not ok and _is_prot(label)]
        print(f"\nprotected songs: {len(prot_rows)}; rejected by current gate: "
              f"{', '.join(n_rej_prot) or 'none'}")
        return

    rejected = 0
    print(f"{'score':>16} {'chars':>6} {'lines':>5} {'est_s':>5}  verdict")
    for label, text, s, (ok, why) in rows:
        rejected += not ok
        print(f"{label:>16} {s['chars']:>6} {s['body_lines']:>5} "
              f"{s['est_duration_s']:>5}  {'ok' if ok else 'REJECT: ' + why}")
    print(f"\n{len(rows) - rejected}/{len(rows)} accepted, {rejected} rejected"
          f"  (max_chars={th.max_chars})")


if __name__ == "__main__":
    main()
