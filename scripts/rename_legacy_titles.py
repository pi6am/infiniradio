#!/usr/bin/env python3
"""One-shot: give tracks still carrying old-scheme titles fresh names.

The original naming scheme was "{prompt} #0068" (or "track #0068"); this
script finds every track whose title ends in its own " #NNNN" id marker and
renames it with the random word-slot generator (see infiniradio.naming).
Variation titles ("Parent · Style") are left alone — they are not reliably
distinguishable from legitimate titles; they age out or can be renamed by
hand from the track page.

Titles are display-only (everything keys on track ids), so it is safe to
run while the server is up. Run from the repo root:

    .venv/bin/python scripts/rename_legacy_titles.py [--config CONFIG] [--dry-run]

--dry-run previews renames; the preview names are freshly generated and may
differ from what the real run picks.
"""

from __future__ import annotations

import argparse
import random
import sys

from infiniradio import naming
from infiniradio.config import load_config
from infiniradio.db import DB


def main() -> int:
    ap = argparse.ArgumentParser(
        description="rename tracks with legacy '{prompt} #NNNN' titles")
    ap.add_argument("--config", default=None,
                    help="path to config.toml (auto-discovered by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview renames without writing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    naming.load_lists(cfg.path("naming.words_dir"))
    db = DB(cfg.path("library.data_dir") / "infiniradio.db")
    rows = db.query("SELECT id, title FROM tracks ORDER BY id")
    legacy = [r for r in rows
              if r["title"] and r["title"].endswith(f" #{r['id']:04d}")]
    if not legacy:
        print('no legacy titles found (nothing ends in its own "#NNNN").')
        return 0

    rng = random.Random()
    for r in legacy:
        if args.dry_run:
            preview = naming.title(rng)
            print(f"  #{r['id']:4d}  {r['title']}  ->  {preview}  (preview)")
        else:
            new = naming.unique_title(db)
            db.execute("UPDATE tracks SET title = ? WHERE id = ?",
                       (new, r["id"]))
            print(f"  #{r['id']:4d}  {r['title']}  ->  {new}")

    verb = "would rename" if args.dry_run else "renamed"
    print(f"\n{verb} {len(legacy)} track(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
