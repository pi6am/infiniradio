"""Listener telemetry: unique listeners per minute and per aired track.

Heartbeats come from HLS requests (/live/...): every active player polls
the playlist every few seconds, so presence == "fetched an HLS file just
now". Listener identity is the ``ir_listener`` cookie when present (web
player) and otherwise a hash of client IP + user-agent (best-effort for
cookie-less clients such as VLC; NAT undercounts, dynamic IPs overcount).

All hot-path state lives in memory; the database is written once per
finished minute (listener_minutes, only when the count is non-zero) and
once per finished airing (tracks.listener_count += unique listeners that
heard it). No listener ids are ever persisted — only counts. In-flight
state is lost on restart, which is acceptable for best-effort telemetry.

The "currently audible" airing used for attribution is refreshed by the
background tick (every few seconds), not per heartbeat — a heartbeat
landing within a few seconds of a song boundary may attribute to the
neighbouring song. With multi-minute tracks that noise is negligible.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta

from .db import DB
from .util import iso, parse_iso, utcnow

log = logging.getLogger(__name__)

TICK_S = 5.0


class Telemetry:
    def __init__(self, db: DB):
        self.db = db
        self._minute_key: str | None = None       # UTC "YYYY-MM-DDTHH:MM"
        self._minute_ids: set[str] = set()
        self.last_minute_count = 0                # last *completed* minute
        # (track_id, started_at) of the airing that is audible right now
        self._current: tuple[int, str] | None = None
        self._airing_ids: dict[tuple[int, str], set[str]] = {}

    # -- identity -----------------------------------------------------------

    def listener_id(self, request) -> str:
        """Web player cookie, or a best-effort hash for cookie-less clients."""
        lid = request.cookies.get("ir_listener")  # routes_public.LISTENER_COOKIE
        if lid:
            return lid
        ip = request.client.host if request.client else "?"
        ua = request.headers.get("user-agent", "")
        return "ip:" + hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:16]

    # -- hot path (sync; atomic on the event loop) ---------------------------

    def heartbeat(self, request, now: datetime | None = None) -> None:
        """One HLS fetch by one listener: counts for the current minute and,
        if something is audible, for that airing."""
        now = now or utcnow()
        lid = self.listener_id(request)
        self._roll_minute(now)
        self._minute_ids.add(lid)
        if self._current is not None:
            self._airing_ids.setdefault(self._current, set()).add(lid)

    def _roll_minute(self, now: datetime) -> None:
        key = now.strftime("%Y-%m-%dT%H:%M")
        if key == self._minute_key:
            return
        if self._minute_ids:
            # Upsert-add: two bursts separated by a lull inside the same
            # minute must accumulate, not overwrite.
            self.db.execute(
                "INSERT INTO listener_minutes (minute, listeners) VALUES (?, ?)"
                " ON CONFLICT(minute) DO UPDATE SET listeners = listeners + excluded.listeners",
                (self._minute_key, len(self._minute_ids)))
            self.last_minute_count = len(self._minute_ids)
        self._minute_key = key
        self._minute_ids = set()

    # -- background ------------------------------------------------------------

    def tick(self, now: datetime | None = None) -> None:
        """Roll the minute, refresh the audible airing, commit finished
        airings to their track's listener_count."""
        now = now or utcnow()
        self._roll_minute(now)
        self._refresh_current(now)
        for key in [k for k in self._airing_ids if k != self._current]:
            ids = self._airing_ids.pop(key)
            if ids:
                self.db.execute(
                    "UPDATE tracks SET listener_count = listener_count + ?"
                    " WHERE id = ?", (len(ids), key[0]))
                log.debug("track %s heard by %d listener(s)", key[0], len(ids))

    def _refresh_current(self, now: datetime) -> None:
        """The airing whose audible window [started_at, started_at+duration]
        covers `now` (started_at is already lag-adjusted at write time)."""
        row = self.db.one(
            "SELECT p.track_id AS track_id, p.started_at AS started_at,"
            " COALESCE(p.played_s, t.duration_s) AS duration_s"
            " FROM play_log p JOIN tracks t ON t.id = p.track_id"
            " WHERE p.started_at <= ? AND t.status != 'deleted'"
            " ORDER BY p.started_at DESC, p.id DESC LIMIT 1", (iso(now),))
        if row is None:
            self._current = None
            return
        end = parse_iso(row["started_at"]) + timedelta(seconds=row["duration_s"] or 0)
        self._current = (row["track_id"], row["started_at"]) if end >= now else None


async def telemetry_loop(rt) -> None:
    while True:
        try:
            if rt.telemetry is not None:
                rt.telemetry.tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("telemetry tick failed")
        await asyncio.sleep(TICK_S)
