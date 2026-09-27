"""HLS encoder: one long-lived ffmpeg child process.

Receives raw PCM (s16le / 48 kHz / stereo) on stdin and writes an fMP4 HLS
live window into data/hls. The playlist is served with no-store; segments are
immutable within the window and publicly cacheable, which is what makes the
"put a CDN in front later" scalability story trivial.

Every encoder life is a self-contained "generation": the fMP4 init segment
and segment filenames carry the epoch (a URL must never serve bytes from
two generations — segments are publicly cached), the previous life's files
are purged on start, and each life's playlist starts its media sequence at
0 like any fresh stream (huge sequence bases poison players' duration
math). Sequence regression across lives is safe because clients rebuild
their playback session when the URL epoch changes; stitching lives
together with append_list is what stalled surviving players across server
restarts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..config import Config

log = logging.getLogger(__name__)


class EncoderDied(Exception):
    pass


class Encoder:
    def __init__(self, cfg: Config, hls_dir: Path, db=None):
        self.cfg = cfg
        self.hls_dir = hls_dir
        self.db = db  # optional: persists the epoch counter across restarts
        self.proc: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None   # wall clock of process start
        self.total_bytes = 0                    # PCM bytes written this life
        self.epoch = 0
        self.started_wall = None                # utcnow() at process start
        self._stderr_task: asyncio.Task | None = None

    @property
    def bytes_per_second(self) -> int:
        return int(cfg_sr(self.cfg)) * 2 * 2  # s16le stereo

    def _args(self) -> list[str]:
        c = self.cfg
        return [
            c.get("stream.ffmpeg_path", "ffmpeg"),
            "-hide_banner", "-loglevel", "warning",
            "-f", "s16le", "-ar", str(cfg_sr(c)), "-ac", "2", "-i", "pipe:0",
            "-c:a", "aac", "-b:a", str(c.get("stream.bitrate", "128k")),
            "-f", "hls",
            "-hls_time", str(int(c.get("stream.segment_seconds", 6))),
            "-hls_list_size", str(int(c.get("stream.playlist_segments", 5))),
            # Media sequence starts at 0 (a 10M base reads as ~694 days of
            # media in players' duration math); generation isolation lives
            # in the URLs instead: the epoch prefixes every segment and
            # init filename, so cached bytes can never cross generations.
            "-hls_flags", "+delete_segments+omit_endlist+independent_segments+program_date_time",
            "-hls_segment_type", "fmp4",
            "-hls_fmp4_init_filename", f"init{self.epoch}.mp4",
            "-hls_segment_filename", str(self.hls_dir / f"seg{self.epoch}-%06d.m4s"),
            str(self.hls_dir / "live.m3u8"),
        ]

    async def start(self) -> None:
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        if self.proc and self.proc.returncode is None:
            return
        # The epoch must exist before the args are built (it anchors the
        # segment numbering and the init filename).
        self.epoch = self._next_epoch()
        self._purge_stale()
        from datetime import datetime, timezone
        self.proc = await asyncio.create_subprocess_exec(
            *self._args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self.started_at = time.monotonic()
        self.total_bytes = 0
        self.started_wall = datetime.now(timezone.utc).replace(tzinfo=None)
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        log.info("HLS encoder started (pid %s, epoch %d)", self.proc.pid, self.epoch)

    def _purge_stale(self) -> None:
        """Remove the previous encoder life's output.

        Each life writes a fresh playlist; files from an older life would
        linger forever (delete_segments only manages the current window).
        Clients still holding the old playlist see brief 404s and resync
        via the new one.
        """
        for pattern in ("live*.m3u8", "seg*.m4s", "init*.mp4", "*.tmp"):
            for p in self.hls_dir.glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass  # e.g. a Windows handle still open; next start retries

    def _next_epoch(self) -> int:
        """Globally unique epoch id for this encoder life.

        play_log rows are matched by (epoch, prog_offset); an in-memory
        counter would mint a colliding "epoch 1" on every server restart,
        so the counter is persisted. First run bootstraps above any epoch
        already present in play_log (stale rows from the in-memory-counter
        era must never be matched again).
        """
        if self.db is None:
            self.epoch += 1
            return self.epoch
        prev = self.db.get_setting("encoder.epoch", None)
        if prev is None:
            row = self.db.one("SELECT COALESCE(MAX(epoch), 0) AS m FROM play_log")
            prev = row["m"] if row else 0
        epoch = int(prev) + 1
        self.db.set_setting("encoder.epoch", epoch)
        return epoch

    def program_seconds(self) -> float:
        """Seconds of program written this encoder life (== the HLS timeline,
        since EXT-X-PROGRAM-DATE-TIME = started_wall + program position)."""
        return self.total_bytes / self.bytes_per_second

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            log.warning("encoder: %s", line.decode(errors="replace").rstrip())

    async def write(self, pcm: bytes) -> None:
        if not self.proc or self.proc.returncode is not None or not self.proc.stdin:
            raise EncoderDied("encoder process is not running")
        try:
            self.proc.stdin.write(pcm)
            await self.proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise EncoderDied(str(exc)) from exc
        self.total_bytes += len(pcm)

    def queued_seconds(self) -> float:
        """Audio written but not yet played (encoder consumes ~1x realtime)."""
        if self.started_at is None:
            return 0.0
        played = time.monotonic() - self.started_at - 0.5  # small priming margin
        return max(0.0, self.total_bytes / self.bytes_per_second - played)

    async def restart(self) -> None:
        log.warning("restarting HLS encoder")
        await self.stop()
        await self.start()

    async def stop(self) -> None:
        proc, self.proc = self.proc, None
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_task = None
        if proc and proc.returncode is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


def cfg_sr(cfg: Config) -> int:
    return int(cfg.get("stream.sample_rate", 48000))
