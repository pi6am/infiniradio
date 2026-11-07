"""CLI entry point: `infiniradio [serve]` and `infiniradio passwd`."""

from __future__ import annotations

import argparse
import getpass
import sys

from .util import setup_logging


def cmd_serve(config: str | None, verbose: bool, mockaudio: bool = False,
              mockllm: bool = False) -> int:
    import uvicorn

    from .app import create_app
    from .config import load_config

    setup_logging(verbose)
    cfg = load_config(config)
    app = create_app(config, mockaudio=mockaudio, mockllm=mockllm)
    uvicorn.run(
        app,
        host=str(cfg.get("server.host", "127.0.0.1")),
        port=int(cfg.get("server.port", 8000)),
        log_config=None,
        access_log=False,
        # Cap the graceful-shutdown wait: reverse proxies (cloudflared) hold
        # persistent connections, and without a bound uvicorn can sit in
        # "waiting for connections to close" long enough that a restart
        # outage outlasts players' retry budgets.
        timeout_graceful_shutdown=5,
    )
    return 0


def cmd_passwd(config: str | None) -> int:
    from .app import WEB_DIR  # noqa: F401  (ensures package import works)

    from .config import load_config
    from .db import DB
    from .web.auth import hash_password

    cfg = load_config(config)
    data_dir = cfg.path("library.data_dir")
    db = DB(data_dir / "infiniradio.db")

    print(f"admin username: {cfg.get('admin.username', 'admin')}")
    pw1 = getpass.getpass("new password: ")
    pw2 = getpass.getpass("repeat: ")
    if not pw1 or pw1 != pw2:
        print("passwords do not match or empty", file=sys.stderr)
        return 1
    db.set_setting("admin.password_hash", hash_password(pw1))
    print(f"stored in {data_dir / 'infiniradio.db'} (settings.admin.password_hash)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="infiniradio",
        description="Infinite AI radio station (ComfyUI/KoboldCpp music + HLS).",
    )
    ap.add_argument("-c", "--config", default=None, help="path to config.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--mockaudio", action="store_true",
                    help="use the mock generator (sine tones) for pipeline "
                         "testing — never for real listening")
    ap.add_argument("--mockllm", action="store_true",
                    help="use the scripted mock LLM client (no endpoint needed) "
                         "so lyrics workflows can be pipeline-tested")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("passwd", help="set the admin password")
    args = ap.parse_args()

    if args.cmd == "passwd":
        raise SystemExit(cmd_passwd(args.config))
    raise SystemExit(cmd_serve(args.config, args.verbose, args.mockaudio, args.mockllm))


if __name__ == "__main__":
    main()
