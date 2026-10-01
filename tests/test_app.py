"""App factory (no lifespan, no ports, no server): startup validation for
mock flags and [llm] misconfiguration, route table wiring.
"""

from __future__ import annotations

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
                     "/admin/tags", "/admin/tags/create", "/admin/tags/{tag_id}/rename",
                     "/admin/tags/{tag_id}/delete", "/admin/tags/{tag_id}/pickable"):
            assert need in paths, f"missing route {need}"
        print(f"1. app with --mockaudio --mockllm created; {len(paths)} admin routes")

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
