#!/usr/bin/env python3
"""ComfyUI output cleanup sidecar for infiniradio.

Runs on the same machine as ComfyUI and deletes already-ingested generation
artifacts (abc_*.txt / audio_* files) from ComfyUI's output folder. ComfyUI
never deletes these itself, so without this service the drive eventually
overflows.

The infiniradio server calls POST /cleanup before submitting every Nth
generation job (files of finished jobs have already been downloaded by then).
It is not run on a schedule on purpose: ABC files are written while the audio
for the same job is still generating, so a naive timer could delete files that
belong to a running job.

Zero dependencies (Python standard library only), works on Linux, macOS and
Windows.

Usage:
    python comfyui_cleanup.py --dir /path/to/ComfyUI/output/radio \
        --token <shared-secret> [--host 0.0.0.0] [--port 8765] \
        [--pattern abc_*.txt --pattern audio_*.mp3 --pattern audio_*.flac] \
        [--min-age 0]

Endpoints:
    GET  /status   -> {"files": n, "bytes": n, "disk_free": n, "disk_total": n}
    POST /cleanup  -> deletes matching files; optional JSON body
                      {"keep": ["abc_00001.txt"], "min_age": 60}
                      returns {"deleted": n, "bytes_freed": n, "errors": [...]}
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PATTERNS = ["abc_*.txt", "audio_*.flac", "audio_*.mp3", "audio_*.wav"]


def matching_files(directory: Path, patterns: list[str], min_age: float, keep: set[str]) -> list[Path]:
    now = time.time()
    out: list[Path] = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        return out
    for p in entries:
        if not p.is_file() or p.name in keep:
            continue
        if not any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
            continue
        if min_age > 0:
            try:
                if now - p.stat().st_mtime < min_age:
                    continue
            except OSError:
                continue
        out.append(p)
    return out


def make_handler(directory: Path, patterns: list[str], token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "infiniradio-sidecar/1.0"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return bool(token) and self.headers.get("X-Token") == token

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/status":
                self._send(404, {"error": "not found"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            files = matching_files(directory, patterns, 0, set())
            total = sum(f.stat().st_size for f in files)
            usage = shutil.disk_usage(directory if directory.exists() else Path.cwd())
            self._send(200, {
                "files": len(files),
                "bytes": total,
                "disk_free": usage.free,
                "disk_total": usage.total,
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/cleanup":
                self._send(404, {"error": "not found"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            keep: set[str] = set()
            min_age = 0.0
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    keep = {str(k) for k in body.get("keep", [])}
                    min_age = float(body.get("min_age", 0))
                except (ValueError, TypeError):
                    self._send(400, {"error": "invalid JSON body"})
                    return
            deleted, freed, errors = 0, 0, []
            for f in matching_files(directory, patterns, min_age, keep):
                try:
                    size = f.stat().st_size
                    f.unlink()
                    deleted += 1
                    freed += size
                except OSError as exc:
                    errors.append(f"{f.name}: {exc}")
            self._send(200, {"deleted": deleted, "bytes_freed": freed, "errors": errors})

        def log_message(self, fmt: str, *args) -> None:  # quiet default access log
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory to watch (ComfyUI output subfolder)")
    ap.add_argument("--token", required=True, help="shared secret; sent by infiniradio as X-Token header")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pattern", action="append", default=None,
                    help="glob pattern of files to delete (repeatable; default: %s)" % " ".join(DEFAULT_PATTERNS))
    ap.add_argument("--min-age", type=float, default=0.0, dest="min_age",
                    help="skip files younger than this many seconds")
    args = ap.parse_args()

    directory = Path(args.dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    patterns = args.pattern or DEFAULT_PATTERNS
    print(f"infiniradio sidecar: watching {directory}")
    print(f"  patterns: {', '.join(patterns)}   auth: {'on' if args.token else 'OFF (insecure!)'}")
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(directory, patterns, args.token))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
