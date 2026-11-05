"""App factory (no lifespan, no ports, no server): startup validation for
mock flags and [llm] misconfiguration, route table wiring.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import _helpers

_helpers.ensure_repo()

from infiniradio.app import create_app
from infiniradio.web.routes_admin import router as admin_router


def _write_cfg(tmp: Path, name: str, extra: str = "", subdir: str = "data") -> Path:
    d = tmp / subdir
    d.mkdir(parents=True, exist_ok=True)
    cfg = tmp / name
    cfg.write_text(f"""
[generator]
backend = "comfyui"
[comfyui]
url = "http://127.0.0.1:8188"
[library]
data_dir = "{d}"
{extra}""")
    return cfg


def run() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="infiniradio-test-", dir=tempfile.gettempdir()))
    try:
        # 1. mockaudio + mockllm: full feature set wires up
        cfg = _write_cfg(tmp, "config.toml")
        app = create_app(cfg, mockaudio=True, mockllm=True)
        paths = {r.path for r in admin_router.routes}
        for need in ("/admin/jobs/{job_id}", "/admin/styles",
                     "/admin/styles/edit/{style_id}", "/admin/workflows/{workflow_id}/toggle",
                     "/admin/track/{track_id}/undelete", "/admin/track/{track_id}/hold",
                     "/admin/track/{track_id}/vary", "/admin/track/{track_id}/download",
                     "/admin/tags", "/admin/tags/create", "/admin/tags/{tag_id}/rename", "/admin/tags/{tag_id}/delete",
                     "/admin/tags/{tag_id}/pickable"):
            assert need in paths, f"missing route {need}"
        print(f"1. app with --mockaudio --mockllm created; {len(paths)} admin routes")

        # 1b. track download: FLAC master preferred (attachment disposition,
        #     title sanitized into the filename), playout fallback, 404s
        from infiniradio.config import load_config
        from infiniradio.db import DB
        from infiniradio.web import routes_admin
        from _helpers import RT
        cfg_dl = _write_cfg(tmp, "dl.toml", subdir="data-dl")
        dl = RT(load_config(cfg_dl), DB(tmp / "data-dl" / "infiniradio.db"), None, None, {})
        music = tmp / "data-dl" / "music"
        (music / "1").mkdir(parents=True)
        (music / "1" / "original.flac").write_bytes(b"flac")
        (music / "1" / "playout.m4a").write_bytes(b"m4a")
        dl.db.execute(
            "INSERT INTO tracks (title, original_path, playout_path, status, created_at)"
            " VALUES ('A / \"Quoted\": Song', '1/original.flac', '1/playout.m4a',"
            " 'ready', '2026-01-01T00:00:00')")
        fake_app = type("A", (), {})()
        fake_app.state = type("S", (), {})()
        fake_app.state.rt = dl
        req = type("R", (), {"app": fake_app, "client": None})()
        real_admin = routes_admin._is_admin
        routes_admin._is_admin = lambda r: True
        try:
            resp = asyncio.run(routes_admin.track_download(req, 1))
            missing = asyncio.run(routes_admin.track_download(req, 999))
            (music / "1" / "original.flac").unlink()  # exercise the fallback
            fallback = asyncio.run(routes_admin.track_download(req, 1))
        finally:
            routes_admin._is_admin = real_admin
        assert resp.status_code == 200 and resp.media_type == "audio/flac"
        cd = resp.headers["content-disposition"]
        # spaces make Starlette use the RFC 5987 filename* form (decoded by
        # every browser); the sanitized title must survive either way
        from urllib.parse import quote
        assert cd.startswith("attachment") and quote("A Quoted Song.flac") in cd, cd
        assert "/" not in cd.split("filename", 1)[1]  # no path separators survive
        assert fallback.media_type == "audio/mp4" and quote("A Quoted Song.m4a") in \
            fallback.headers["content-disposition"]
        assert missing.status_code == 404
        print(f"1b. track download ok: {cd}")

        # 2. no llm: llm resolves off, behavior unchanged
        cfg_b = _write_cfg(tmp, "nollm.toml", subdir="data-b")
        create_app(cfg_b, mockaudio=True)
        print("2. app without llm created (llm 'off' by config default)")

        # 3. unknown [llm] backend fails fast at create_app
        cfg_bad = _write_cfg(tmp, "bad.toml",
                             extra='[llm]\nenabled = true\nbackend = "nope"\n',
                             subdir="data2")
        try:
            create_app(cfg_bad, mockaudio=True)
            raise AssertionError("expected clean startup failure")
        except SystemExit as exc:
            assert str(exc) == "2"
        print("3. unknown [llm] backend -> clean startup failure (exit 2)")

        # 4. mock backend in config is rejected (flag-only, like --mockaudio)
        cfg_mock = _write_cfg(tmp, "mockconf.toml",
                              extra='[llm]\nenabled = true\nbackend = "mock"\n',
                              subdir="data3")
        try:
            create_app(cfg_mock, mockaudio=True)
            raise AssertionError("expected clean startup failure")
        except SystemExit as exc:
            assert str(exc) == "2"
        print("4. [llm] backend = \"mock\" in config -> rejected (flag only)")
        print("test_app: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
