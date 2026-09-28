"""Track feeder: keeps the encoder fed with gapless PCM.

For each selected track, a short-lived ffmpeg decoder turns the playout file
into raw PCM which is piped straight into the encoder's stdin. The feeder runs
about one track ahead of the live edge; each track gets a play_log entry with
its computed *airtime* (now + queued audio), which is what the now-playing API
reports against the latency hint.

If the DJ has nothing to offer, real silence is written so the stream never
dies. The admin-panel playback pause uses the same mechanism: the current
track is skipped out of the buffer (encoder restart, players resync within
seconds) and deliberate silence follows until playback is resumed — the
encoder keeps running the whole time, so listeners stay connected.

A restart resumes the airing it interrupted: the position is predicted from
the airing's start time and the track duration (downtime-worth of song is
skipped — imperceptible for quick restarts, and long downtimes exhaust the
song so playback starts fresh), the remainder is seeked into the playout
file, and the existing play_log row is updated in place with the remainder's
length so the now-playing timeline stays frame-accurate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta

from .. import dj, library
from ..config import Config
from ..streamer.encoder import Encoder, EncoderDied
from ..util import iso, parse_iso, utcnow

log = logging.getLogger(__name__)

CHUNK = 1 << 16
SILENCE_WARN_EVERY_S = 60.0
# The encoder transcodes as fast as PCM arrives, so *we* must pace it: keep at
# most MAX_LEAD_S of audio buffered ahead of realtime. Small enough that the
# schedule/now-playing model stays meaningful, large enough for gapless
# track swaps (the lead only drains at 1x while we open the next decoder).
MAX_LEAD_S = 90.0
# A restart resumes the interrupted song only if at least this much of it
# is left; shorter remainders just move on to a fresh pick.
RESUME_MIN_REMAINDER_S = 5.0


class Feeder:
    def __init__(self, rt):
        self.rt = rt
        self.cfg: Config = rt.cfg
        self.encoder = Encoder(rt.cfg, rt.cfg.path("library.data_dir") / "hls",
                               db=rt.db)
        self.current_track_id: int | None = None
        self.skip_requested = False
        self._silence_since: float | None = None
        self._pending_resume: tuple[dict, float, int] | None = None

    @property
    def active_ids(self) -> set[int]:
        ids = set()
        if self.current_track_id:
            ids.add(self.current_track_id)
        return ids

    # -- manual queue & skip -------------------------------------------------

    @property
    def playback_paused(self) -> bool:
        """Admin-panel pause switch (persisted; sticky across restarts)."""
        return bool(self.rt.db.get_setting("playback_paused"))

    def _listener_lag(self) -> float:
        """How far listeners trail what we write into the encoder.

        ffmpeg muxes PCM into segments immediately (at full speed); players
        ride the HLS live edge roughly liveSyncDurationCount (3) segments
        behind the newest segment. The feeder's pacing lead does NOT delay
        listeners, so the only offset between "written now" and "heard now"
        is this player-side lag.
        """
        return float(self.cfg.get("stream.latency_hint", 20.0))

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

    def _resume_plan(self) -> tuple[dict, float, int] | None:
        """(track, offset_s, play_log row id): continue the airing that a
        restart cut off, if resuming is worthwhile.

        The position is predicted from the airing's start time and the
        track's duration: the downtime-worth of song is simply skipped,
        which for a quick restart is imperceptible, and a long downtime
        exhausts the song on its own (remainder under the threshold) so
        playback starts fresh instead.
        """
        row = self.rt.db.one(
            "SELECT p.id AS row_id, p.started_at AS started_at, t.*"
            " FROM play_log p JOIN tracks t ON t.id = p.track_id"
            " ORDER BY p.id DESC LIMIT 1")
        if row is None or row["status"] != "ready" or not row["playout_path"]:
            return None
        duration = row["duration_s"] or 0.0
        heard = (utcnow() - parse_iso(row["started_at"])).total_seconds()
        offset = max(0.0, min(heard, duration))
        if duration - offset < RESUME_MIN_REMAINDER_S:
            return None
        return row, offset, row["row_id"]

    # ------------------------------------------------------------------------

    async def run(self) -> None:
        await self.encoder.start()
        log.info("feeder running; live playlist at %s", self.encoder.hls_dir / "live.m3u8")
        self._pending_resume = self._resume_plan()
        while True:
            try:
                if self.playback_paused:
                    await self._feed_silence(paused=True)
                    continue
                if self._pending_resume is not None:
                    track, offset, row_id = self._pending_resume
                    self._pending_resume = None
                    log.info("resuming track %s (%s) at %.1fs after restart",
                             track["id"], track["title"], offset)
                    if await self._feed_track(track, resume_offset_s=offset,
                                              resume_row_id=row_id):
                        await self._flush_after_skip(track)
                    continue
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
                    await self._flush_after_skip(track)
            except EncoderDied:
                await self.encoder.restart()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("feeder iteration failed; cooling down")
                await asyncio.sleep(2)

    async def _flush_after_skip(self, track: dict) -> None:
        # The flushed audio never reached listeners as scheduled; drop its
        # play_log entry so now-playing stays truthful, and flush everything
        # already buffered by restarting the encoder (players resync quickly).
        self.rt.db.execute(
            "DELETE FROM play_log WHERE id ="
            " (SELECT MAX(id) FROM play_log WHERE track_id = ?)",
            (track["id"],),
        )
        log.info("flushing buffer after skip (encoder restart)")
        await self.encoder.restart()


    async def _pace(self) -> None:
        """Throttle writes so we stay at most MAX_LEAD_S ahead of realtime."""
        while True:
            lead = self.encoder.queued_seconds()
            if lead <= MAX_LEAD_S:
                return
            await asyncio.sleep(min(lead - MAX_LEAD_S, 5.0))

    async def _feed_silence(self, paused: bool = False) -> None:
        """Air real silence so the stream never dies.

        Deliberate pause (admin panel) is a quiet info log — the empty-
        library case stays a warning with periodic reminders, since it
        means the DJ has run out of eligible material.
        """
        now = time.monotonic()
        if self._silence_since is None:
            self._silence_since = now
            if paused:
                log.info("playback paused — airing silence")
            else:
                log.warning("feeder: nothing eligible to play — airing silence")
        elif not paused and now - self._silence_since > SILENCE_WARN_EVERY_S:
            self._silence_since = now
            log.warning("feeder: still silent (empty library?)")
        bps = self.encoder.bytes_per_second
        chunk = b"\x00" * (bps // 4)  # 0.25 s
        # Respect the same lead budget as tracks: a multi-hour pause must not
        # stuff the encoder with hours of buffered silence.
        await self._pace()
        await self.encoder.write(chunk)
        await asyncio.sleep(0.2)

    async def _feed_track(self, track: dict, resume_offset_s: float = 0.0,
                          resume_row_id: int | None = None) -> bool:
        playout = self.cfg.path("library.data_dir") / "music" / track["playout_path"]
        if not playout.is_file():
            log.error("playout file missing for track %s (%s) — skipping",
                      track["id"], playout)
            return False

        airtime = iso(utcnow())  # replaced below with queue-aware estimate
        self.current_track_id = track["id"]
        seek = ["-ss", f"{resume_offset_s:.3f}"] if resume_offset_s > 0 else []
        proc = await asyncio.create_subprocess_exec(
            self.cfg.get("stream.ffmpeg_path", "ffmpeg"),
            "-hide_banner", "-loglevel", "error", *seek,
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
                # Listeners start hearing this track one listener-lag after
                # we write its first samples (players trail the live edge).
                airtime = iso(utcnow() + timedelta(seconds=self._listener_lag()))
                played_s = None
                if resume_row_id is not None:
                    duration = track["duration_s"] or 0.0
                    played_s = max(0.0, duration - resume_offset_s)
                library.mark_aired(
                    self.rt.db, track["id"], airtime,
                    prog_offset=self.encoder.program_seconds(),
                    epoch=self.encoder.epoch,
                    played_s=played_s, resume_row_id=resume_row_id,
                )
                log.info("%s track %s (%s) — audible from %s",
                         "resuming" if resume_row_id is not None else "airing",
                         track["id"], track["title"], airtime)
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
