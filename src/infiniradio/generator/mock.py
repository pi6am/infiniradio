"""Mock generator: local ffmpeg sine tones + synthetic ABC.

Testing only — reachable exclusively via the --mockaudio command line flag,
never through config, so listeners cannot accidentally be served test tones.
Exercises the whole pipeline (DJ, stream, votes, eviction, admin) on a
machine without a GPU.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..config import Config
from .base import GenRequest, GenResult, Generator

log = logging.getLogger(__name__)

KEYS = ["C", "G", "D", "Am", "Em", "Bm", "F#m", "G#m", "D#m", "Cm"]


def fake_abc(req: GenRequest) -> str:
    tempo = 110 + req.seed % 90
    key = KEYS[req.seed % len(KEYS)]
    return (
        "X:1\n"
        "T:\n"
        "M:4/4\n"
        f"L:1/16\n"
        f"Q:1/4={tempo}\n"
        f"K:{key}\n"
        "% mock score (generator backend = mock)\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        + "c2c2c2c2c2c2c2c2|\n" * 8
    )


class MockGenerator(Generator):
    name = "mock"

    def __init__(self, cfg: Config):
        self.ffmpeg = cfg.get("stream.ffmpeg_path", "ffmpeg")

    async def generate(self, req: GenRequest, workdir: Path) -> GenResult:
        duration = 30 + req.seed % 45
        freq = 196.0 + (req.seed % 60) * 7.5
        out = workdir / "original.flac"
        fade_out = max(0.0, duration - 2)
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq:.1f}:sample_rate=48000",
            "-t", str(duration),
            "-af", f"afade=t=in:d=1,afade=t=out:st={fade_out:.2f}:d=2",
            "-ac", "2", "-c:a", "flac", str(out),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"mock ffmpeg failed: {err.decode(errors='replace').strip()}")
        log.info("mock: generated %.0fs tone at %.0f Hz (seed %s)", duration, freq, req.seed)
        return GenResult(original_path=out, abc_text=fake_abc(req),
                         info={"duration_s": duration, "frequency": freq,
                               "seed": int(req.seed)})
