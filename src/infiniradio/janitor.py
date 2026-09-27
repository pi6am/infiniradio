"""Janitor: disk quota enforcement and eviction.

Policy (per plan):
  * Generation *pauses* when over quota — that is the primary control.
  * Unplayed audio is NEVER auto-deleted: generating then deleting without
    airing would waste time and energy.
  * Eviction (only when over quota, or via downvote rule) touches played
    tracks only, never protected tracks, never whatever is currently being
    fed to the encoder.
  * ABC text always survives (kept in the tracks row).
"""

from __future__ import annotations

import logging

from .config import Config, runtime_get
from .db import DB
from .library import delete_track, disk_usage
from .util import human_bytes, wilson_score

log = logging.getLogger(__name__)

RESUME_FRACTION = 0.95  # resume generation below 95% of quota (hysteresis)


def quota_bytes(db: DB, cfg: Config) -> int:
    return int(runtime_get(cfg, db, "library.quota_gb") * (1 << 30))


def quota_state(db: DB, cfg: Config) -> dict:
    usage = disk_usage(db)
    quota = quota_bytes(db, cfg)
    paused = usage > quota
    return {
        "usage": usage,
        "quota": quota,
        "paused": paused,
        "reason": "disk quota exceeded" if paused else "",
    }


def generation_paused(db: DB, cfg: Config) -> tuple[bool, str]:
    """Why (not whether) commissioning is currently paused.

    Two independent causes: the admin-panel pause switch (manual, sticky
    across restarts) and the disk quota (automatic, with hysteresis).
    """
    if db.get_setting("generation_paused"):
        return True, "paused from the admin panel"
    st = quota_state(db, cfg)
    return st["paused"], st["reason"]


def _eviction_candidates(db: DB, active_ids: set[int]) -> list[dict]:
    """Played, ready, unprotected tracks; sorted worst-first."""
    rows = db.query(
        "SELECT * FROM tracks WHERE status = 'ready' AND protected = 0 AND play_count > 0"
    )
    rows = [r for r in rows if r["id"] not in active_ids]
    rows.sort(key=lambda r: (
        r["vote_up"] - r["vote_down"],                 # most net-downvoted first
        wilson_score(r["vote_up"], r["vote_down"]),    # then lowest score
        r["last_played_at"] or "",                     # then least recently played
    ))
    return rows


def run_janitor(db: DB, cfg: Config, active_ids: set[int]) -> dict:
    """Downvote auto-evict + over-quota eviction. Returns a summary."""
    summary = {"downvote_evicted": 0, "quota_evicted": 0}

    # 1. Tracks the audience has rejected: net votes at/below threshold after
    #    enough plays. They had their chances.
    net = runtime_get(cfg, db, "dj.downvote_evict_net")
    plays = runtime_get(cfg, db, "dj.downvote_evict_plays")
    doomed = db.query(
        "SELECT id FROM tracks WHERE status = 'ready' AND protected = 0"
        " AND (vote_up - vote_down) <= ? AND play_count >= ?",
        (net, plays),
    )
    for r in doomed:
        if r["id"] in active_ids:
            continue
        if delete_track(db, cfg, r["id"]):
            summary["downvote_evicted"] += 1

    # 2. If still over quota, evict played tracks (worst first) until we are
    #    back under the resume threshold or nothing is evictable.
    quota = quota_bytes(db, cfg)
    target = int(quota * RESUME_FRACTION)
    if disk_usage(db) > quota:
        for row in _eviction_candidates(db, active_ids):
            if disk_usage(db) <= target:
                break
            if delete_track(db, cfg, row["id"]):
                summary["quota_evicted"] += 1
                log.info("quota eviction: removed track %s (%s), usage now %s",
                         row["id"], row["title"], human_bytes(disk_usage(db)))

    if summary["downvote_evicted"] or summary["quota_evicted"]:
        log.info("janitor: %s", summary)
    return summary
