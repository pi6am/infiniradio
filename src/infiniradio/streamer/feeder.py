"""Track feeder: keeps the encoder fed with gapless PCM.

For each selected track, a short-lived ffmpeg decoder turns the playout file
into raw PCM which is piped straight into the encoder's stdin. The feeder runs
about one track ahead of the live edge; each track gets a play_log entry with
its computed *airtime* (now + queued audio), which is what the now-playing API
reports against the latency hint.

If the DJ has nothing to offer, real silence is written so the stream never
dies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta

from .. import dj, library
from ..config import Config
from ..streamer.encoder import Encoder, EncoderDied
from ..util import iso, utcnow

log = logging.getLogger(__name__)

CHUNK = 1 << 16
SILENCE_WARN_EVERY_S = 60.0
# The encoder transcodes as fast as PCM arrives, so *we* must pace it: keep at
# most MAX_LEAD_S of audio buffered ahead of realtime. Small enough that the
# schedule/now-playing model stays meaningful, large enough for gapless
# track swaps (the lead only drains at 1x while we open the next decoder).
MAX_LEAD_S = 90.0


class Feeder:
    def __init__(self, rt):
        self.rt = rt
        self.cfg: Config = rt.cfg
        self.encoder = Encoder(rt.cfg, rt.cfg.path("library.data_dir") / "hls")
        self.current_track_id: int | None = None
        self.skip_requested = False
        self._silence_since: float | None = None

    @property
    def active_ids(self) -> set[int]:
        ids = set()
        if self.current_track_id:
            ids.add(self.current_track_id)
        return ids

    # -- manual queue & skip -------------------------------------------------

    def _pop_queue(self) -> dict | None:
        """Oldest manually queued playable track (drops stale entries)."""
        while True:
            row = self.rt.db.one(
                "SELECT q.id AS qid, t.* FROM play_queue q"
                " JOIN tracks t ON t.id = q.track_id ORDER BY q.id LIMIT 1"
            )
            if row is None:
                return None
            self.rt.db.execute("DELETE FROM play_queue WHERE id = ?", (row["qid"],))
            if row["status"] == "ready" and row["playout_path"]:
                return row
            log.info("dropping stale queue entry for track %s (status %s)",
                     row["id"], row["status"])

    def queued_ids(self) -> set[int]:
        return {r["track_id"] for r in self.rt.db.query("SELECT track_id FROM play_queue")}

    def request_skip(self) -> bool:
        """Skip the track currently being fed (flushes buffered audio)."""
        if self.current_track_id is None:
            return False
        self.skip_requested = True
        return True

    # ------------------------------------------------------------------------

    async def run(self) -> None:
        await self.encoder.start()
        log.info("feeder running; live playlist at %s", self.encoder.hls_dir / "live.m3u8")
        while True:
            try:
                track = self._pop_queue() or dj.pick_next(
                    self.rt.db, self.cfg,
                    exclude_ids=self.active_ids | self.queued_ids(),
                )
                if track is None:
                    await self._feed_silence()
                    continue
                self._silence_since = None
                skipped = await self._feed_track(track)
                if skipped:
                    # Flush everything already buffered of the skipped track
                    # by restarting the encoder (players resync in seconds).
                    log.info("flushing buffer after skip (encoder restart)")
                    await self.encoder.restart()
            except EncoderDied:
                await self.encoder.restart()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("feeder iteration failed; cooling down")
                await asyncio.sleep(2)

    async def _pace(self) -> None:
        """Throttle writes so we stay at most MAX_LEAD_S ahead of realtime."""
        while True:
            lead = self.encoder.queued_seconds()
            if lead <= MAX_LEAD_S:
                return
            await asyncio.sleep(min(lead - MAX_LEAD_S, 5.0))

    async def _feed_silence(self) -> None:
        now = time.monotonic()
        if self._silence_since is None:
            self._silence_since = now
            log.warning("feeder: nothing eligible to play — airing silence")
        elif now - self._silence_since > SILENCE_WARN_EVERY_S:
            self._silence_since = now
            log.warning("feeder: still silent (empty library?)")
        bps = self.encoder.bytes_per_second
        chunk = b"\x00" * (bps // 4)  # 0.25 s
        await self.encoder.write(chunk)
        await asyncio.sleep(0.2)

    async def _feed_track(self, track: dict) -> None:
        playout = self.cfg.path("library.data_dir") / "music" / track["playout_path"]
        if not playout.is_file():
            log.error("playout file missing for track %s (%s) — skipping",
                      track["id"], playout)
            return

        airtime = iso(utcnow())  # replaced below with queue-aware estimate
        self.current_track_id = track["id"]
        proc = await asyncio.create_subprocess_exec(
            self.cfg.get("stream.ffmpeg_path", "ffmpeg"),
            "-hide_banner", "-loglevel", "error",
            "-i", str(playout),
            "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.encoder.cfg.get("stream.sample_rate", 48000)),
            "-ac", "2", "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        first = True
        skipped = False
        while True:
            if self.skip_requested:
                self.skip_requested = False
                skipped = True
                break
            chunk = await proc.stdout.read(CHUNK)
            if not chunk:
                break
            if first:
                # This track starts airing once everything already queued
                # has been consumed.
                airtime = iso(utcnow() + timedelta(seconds=self.encoder.queued_seconds()))
                library.mark_aired(self.rt.db, track["id"], airtime)
                log.info("airing track %s (%s) at %s [queued %.1fs]",
                         track["id"], track["title"], airtime,
                         self.encoder.queued_seconds())
                first = False
            await self._pace()
            await self.encoder.write(chunk)
        if skipped:
            proc.terminate()
        _, err = await proc.communicate()
        if not skipped and proc.returncode != 0:
            log.error("decode failed for track %s: %s", track["id"],
                      err.decode(errors="replace").strip())
        self.current_track_id = None
        if skipped:
            log.info("skipped track %s (%s)", track["id"], track["title"])
        return skipped

    async def stop(self) -> None:
        await self.encoder.stop()
