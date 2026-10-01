"""Encoder generation mechanics, without spawning ffmpeg:

- every generation's outputs are epoch-named (a URL must never serve two
  generations' bytes — segments are publicly cached), and the playlist is
  the one stable URL each new life overwrites;
- a flush keeps the previous life's live.m3u8 in place (the manifest
  bridge). Deleting it first opened a window where every client's manifest
  poll 404'd and burned hls.js's level-retry budget into fatal errors;
  skips wedged listening players until they refreshed.
"""

from __future__ import annotations

import shutil

import _helpers

_helpers.ensure_repo()

from infiniradio.streamer.encoder import Encoder

from _helpers import isolated_config


def run() -> None:
    cfg, tmp = isolated_config()
    try:
        hls_dir = tmp / "hls"
        hls_dir.mkdir()
        enc = Encoder(cfg, hls_dir, db=None)

        # -- a previous life's output sits in the window -------------------
        prev_epoch = enc._next_epoch()          # persists via settings when
        files = {                              # a db is attached; 1 in-memory
            f"seg{prev_epoch}-000000.m4s": b"seg0",
            f"seg{prev_epoch}-000001.m4s": b"seg1",
            f"init{prev_epoch}.mp4": b"init",
            "live.m3u8": b"#EXTM3U old generation",
            "live.m3u8.tmp": b"partial write",
        }
        for name, blob in files.items():
            (hls_dir / name).write_bytes(blob)

        enc._purge_stale()

        # segments/init/tmp from the old life are gone...
        for name in files:
            if name != "live.m3u8":
                assert not (hls_dir / name).exists(), f"{name} survived purge"
        # ...but the manifest bridges the flush: the new life overwrites the
        # same URL once its first segment is cut, so client manifest polls
        # never 404 across a skip/restart/flush
        assert (hls_dir / "live.m3u8").read_bytes() == b"#EXTM3U old generation"
        print("purge: old segments/init purged, live.m3u8 bridges the flush")

        # -- the new life's outputs are generation-unique -------------------
        enc.epoch = 42
        args = enc._args()
        seg_arg = next(a for a in args if a.endswith("%06d.m4s"))
        assert f"seg42-" in seg_arg and str(hls_dir) in seg_arg
        init_arg = args[args.index("-hls_fmp4_init_filename") + 1]
        assert init_arg == "init42.mp4"
        # the playlist URL is stable across lives (that's what makes the
        # bridge overwrite safe); it is served no-store, never cached
        assert args[-1] == str(hls_dir / "live.m3u8")
        print("args: epoch-named segments/init, stable live.m3u8 URL")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
    print("OK")
