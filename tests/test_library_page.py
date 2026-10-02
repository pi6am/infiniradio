"""Library tab sorting + paging (against _library_view, the route's whole
body): every column sorts both directions with newest-id tiebreaks, 'votes'
sorts by TOTAL votes (not net — the score column already covers net),
'score' orders exactly as the displayed Wilson values, pages hold 100
tracks, out-of-range pages clamp, and the pager/column URLs carry state.
"""

from __future__ import annotations

import shutil

import _helpers

_helpers.ensure_repo()

from infiniradio.db import DB
from infiniradio.util import wilson_score
from infiniradio.web.routes_admin import LIBRARY_PAGE_SIZE, _library_view

from _helpers import isolated_config


def _add(db, title: str, **kw) -> None:
    cols = {"title": title, "status": "ready",
            "created_at": "2026-01-01T00:00:00", **kw}
    db.execute(
        f"INSERT INTO tracks ({','.join(cols)})"
        f" VALUES ({','.join('?' * len(cols))})",
        tuple(cols.values()))


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        db = DB(tmp / "infiniradio.db")

        # 130 plain fillers (more than one page), then some deliberate rows
        for i in range(LIBRARY_PAGE_SIZE + 30):
            _add(db, f"filler {i:03d}")
        _add(db, "apple", vote_up=5, vote_down=0, play_count=4,
             last_played_at="2026-09-01T00:00:00")
        _add(db, "Banana", vote_up=3, vote_down=3, play_count=4,
             last_played_at="2026-09-02T00:00:00")
        _add(db, "cherry", vote_up=100, vote_down=10, play_count=9,
             last_played_at="2026-09-03T00:00:00")
        _add(db, "drub", vote_up=0, vote_down=10)    # wilson 0.0, net -10
        _add(db, "worse", vote_up=0, vote_down=100)  # wilson 0.0, net -100
        _add(db, "ghost", status="trashed")     # other tabs are separate

        total_ready = LIBRARY_PAGE_SIZE + 35

        # 1. default: newest first, first page full, pager arithmetic
        v = _library_view(db, "ready", "id", "desc", 1)
        assert v["sort"] == "id" and v["dir"] == "desc"
        assert len(v["tracks"]) == LIBRARY_PAGE_SIZE
        assert v["tracks"][0]["title"] == "worse"       # highest ready id
        p = v["pager"]
        assert p["page"] == 1 and p["total"] == total_ready
        assert p["max_page"] == (total_ready + LIBRARY_PAGE_SIZE - 1) // LIBRARY_PAGE_SIZE
        assert p["first"] is None and p["prev"] is None
        assert p["next"] and p["last"]
        print(f"1. default sort newest-first; page 1/{p['max_page']}, "
              f"{p['total']} tracks")

        # 2. last page holds the remainder; beyond-range pages clamp
        v = _library_view(db, "ready", "id", "desc", p["max_page"])
        assert len(v["tracks"]) == total_ready - LIBRARY_PAGE_SIZE * (p["max_page"] - 1)
        assert v["pager"]["next"] is None and v["pager"]["last"] is None
        assert v["pager"]["first"] and v["pager"]["prev"]
        assert _library_view(db, "ready", "id", "desc", 999)["pager"]["page"] == p["max_page"]
        assert _library_view(db, "ready", "id", "desc", 0)["pager"]["page"] == 1
        print("2. last page remainder; page 0/999 clamp to 1/max")

        # 3. votes sort by TOTAL votes: Banana (3+3=6) outranks apple (5+0)
        #    even though its NET is lower — and worse (0+100) ranks high
        #    despite net -100; net is the score column's job
        names = [t["title"] for t in
                 _library_view(db, "ready", "votes", "desc", 1)["tracks"][:5]]
        assert names == ["cherry", "worse", "drub", "Banana", "apple"], names
        asc = [t["title"] for t in
               _library_view(db, "ready", "votes", "asc", 1)["tracks"][:2]]
        # zero-vote fillers all tie -> newest id first (filler 129 is last)
        assert asc == ["filler 129", "filler 128"]
        print("3. votes column = total votes (cherry 110 > Banana 6 > apple 5)")

        # 4. score orders by the displayed Wilson lower bound
        assert wilson_score(100, 10) > wilson_score(5, 0) > wilson_score(3, 3)
        assert wilson_score(0, 10) == wilson_score(0, 100) == 0.0
        names = [t["title"] for t in
                 _library_view(db, "ready", "score", "desc", 1)["tracks"][:3]]
        assert names == ["cherry", "apple", "Banana"], names
        # ...with NET votes breaking Wilson ties: drub (0,-10) outranks
        # worse (0,-100), and zero-net fillers outrank both
        tail = _library_view(db, "ready", "score", "desc",
                             _library_view(db, "ready", "score", "desc", 1)
                             ["pager"]["max_page"])["tracks"][-2:]
        assert [t["title"] for t in tail] == ["drub", "worse"], tail
        print("4. score matches displayed Wilson; net breaks 0.0 ties")

        # 5. title sorts case-insensitively; ties break to newest id
        v = _library_view(db, "ready", "title", "asc", 1)
        assert [t["title"] for t in v["tracks"][:3]] == ["apple", "Banana", "cherry"]
        # plays: cherry 9, then apple & Banana tied at 4 — the tie goes to
        # the newer id (Banana was inserted after apple) — then the fillers
        v = _library_view(db, "ready", "plays", "desc", 1)
        plays = [t["play_count"] for t in v["tracks"]]
        assert plays[:3] == [9, 4, 4]
        assert v["tracks"][1]["title"] == "Banana" and v["tracks"][2]["title"] == "apple"
        assert v["tracks"][1]["id"] > v["tracks"][2]["id"]
        assert plays == sorted(plays, reverse=True)
        print("5. case-insensitive titles; ties break to newest id")

        # 6. column URLs: active column flips direction, others use their
        #    default, both reset to page 1; status rides along
        v = _library_view(db, "ready", "votes", "desc", 2)
        cols = {c["label"]: c for c in v["columns"]}
        assert cols["votes"]["arrow"] == "▼"
        assert "dir=asc" in cols["votes"]["url"] and "page=1" in cols["votes"]["url"]
        assert "status=ready" in cols["votes"]["url"]
        assert cols["title"]["arrow"] == "" and "dir=asc" in cols["title"]["url"]
        assert v["keep"] == "sort=votes&dir=desc"
        flipped = _library_view(db, "ready", "votes", "asc", 2)
        fcols = {c["label"]: c for c in flipped["columns"]}
        assert fcols["votes"]["arrow"] == "▲" and "dir=desc" in fcols["votes"]["url"]
        print("6. header links flip/reset correctly; state rides along")

        # 7. unknown sort/dir fall back to the defaults; tab filter applies
        v = _library_view(db, "ready", "bogus", "sideways", 1)
        assert v["sort"] == "id" and v["dir"] == "desc"
        assert all(t["status"] == "ready" for t in v["tracks"])
        assert _library_view(db, "trashed", "id", "desc", 1)["pager"]["total"] == 1
        print("7. unknown params fall back; trashed tab sees only 'ghost'")

        print("test_library_page: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
