"""FastAPI application assembly and runtime wiring."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import dj, library, llm as llm_mod
from .config import Config, load_config
from .db import DB
from .generator.base import Generator
from .generator.comfyui import ComfyUIGenerator
from .generator.koboldcpp import KoboldCppGenerator
from .generator.mock import MockGenerator
from .llm import LLMClient, make_images_client, make_llm_client
from .util import utcnow
from .web import auth
from .web.routes_admin import router as admin_router
from .web.routes_public import router as public_router

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"


@dataclass
class Runtime:
    cfg: Config
    db: DB
    generator: Generator | None
    llm: LLMClient | None = None
    workflows: dict = field(default_factory=dict)
    images: object | None = None       # track-art backend (llm.images)
    artworkflows: dict = field(default_factory=dict)
    gen_lock: object | None = None    # asyncio.Lock when lyrics.serialize_with_render
    feeder: object | None = None
    telemetry: object | None = None
    started: datetime = field(default_factory=utcnow)


class StartupError(RuntimeError):
    """Fatal misconfiguration; the server must not start."""


def resolve_backend(cfg: Config, mockaudio: bool) -> str:
    """Effective generator backend: 'mock' | 'comfyui' | 'koboldcpp' | 'off'.

    The mock generator is testing-only and reachable exclusively through the
    --mockaudio command line flag so that no listener can ever be served sine
    tones by accident (e.g. a missing or unloaded config file).
    """
    if mockaudio:
        configured = str(cfg.get("generator.backend", "") or "").strip().lower()
        if configured not in ("", "mock"):
            log.warning("--mockaudio overrides configured [generator] backend %r", configured)
        return "mock"
    configured = str(cfg.get("generator.backend", "") or "").strip().lower()
    if configured == "mock":
        raise StartupError(
            '[generator] backend = "mock" is testing-only and cannot be set in '
            "config; start the server with --mockaudio if you really want "
            "sine-tone test audio."
        )
    if configured in ("", "off"):
        return "off"
    if configured in {"comfyui", "koboldcpp"}:
        return configured
    raise StartupError(
        f"unknown [generator] backend {configured!r} "
        "(expected 'comfyui', 'koboldcpp', or 'off')"
    )


def make_generator(cfg: Config, backend: str) -> Generator | None:
    if backend == "comfyui":
        return ComfyUIGenerator(cfg)
    if backend == "koboldcpp":
        return KoboldCppGenerator(cfg)
    if backend == "mock":
        return MockGenerator(cfg)
    return None  # 'off'


def resolve_llm(cfg: Config, mockllm: bool) -> str:
    """Effective LLM backend name: 'openai' | ... | 'mock' | 'off'.

    Pure config validation (no client constructed) so a typo in [llm] fails
    fast at create_app with one clean error line, not a traceback at the
    first commissioned lyrics job. The client itself is made in lifespan.
    """
    if mockllm:
        return "mock"
    if not bool(cfg.get("llm.enabled", False)):
        return "off"
    backend = str(cfg.get("llm.backend", "openai") or "openai").strip().lower()
    if backend == "mock":
        raise StartupError(
            '[llm] backend = "mock" is testing-only and cannot be set in '
            "config; start the server with --mockllm for scripted LLM responses"
        )
    if backend not in llm_mod.BACKENDS:
        raise StartupError(
            f"unknown [llm] backend {backend!r} (expected "
            + " or ".join(map(repr, llm_mod.BACKENDS)) + ")"
        )
    if not str(cfg.get("llm.base_url", "") or "").strip():
        raise StartupError("[llm] enabled but base_url is empty")
    if not str(cfg.get("llm.model", "") or "").strip():
        log.warning("[llm] no model set — endpoints that require an explicit "
                    "model name will reject requests")
    return backend


def _check_playout_viable(cfg: Config, backend: str) -> None:
    """A station with no generator needs an existing library; otherwise it
    could only ever broadcast silence."""
    if backend != "off":
        return
    db = DB(cfg.path("library.data_dir") / "infiniradio.db")
    try:
        ready = db.one("SELECT COUNT(*) AS n FROM tracks WHERE status = 'ready'")["n"]
    finally:
        db.conn.close()
    if ready == 0:
        raise StartupError(
            "no generator backend configured and the library is empty — set "
            '[generator] backend = "comfyui" or "koboldcpp" (and configure that '
            "backend's URL) in config.toml, or run with --mockaudio for testing"
        )
    log.warning("no generator backend configured — airing the existing library "
                "(%d ready track(s)); no new music will be generated", ready)


class _HardeningMiddleware:
    """Security headers + a pre-authentication request-body size cap.

    Headers: a conservative set on every response (nosniff, clickjacking
    protection, referrer policy); HSTS only when the request itself arrived
    over HTTPS (behind a TLS-terminating proxy such as cloudflared), so
    plain-HTTP LAN access is unaffected.

    Body cap: refuse declared bodies over MAX_BODY_BYTES before reading
    them — /api/vote and the login form are reachable pre-authentication.
    """

    MAX_BODY_BYTES = 1 << 20  # 1 MiB; vote JSON and login forms are tiny

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_hardened(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                have = {k.lower() for k, _ in headers}
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"SAMEORIGIN"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                ]
                if scope.get("scheme") == "https":
                    extra.append((b"strict-transport-security", b"max-age=31536000"))
                headers.extend(h for h in extra if h[0] not in have)
            await send(message)

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.MAX_BODY_BYTES:
                        await _plain_response(send_hardened, 413, b"payload too large")
                        return
                except ValueError:
                    pass

        await self.app(scope, receive, send_hardened)


async def _plain_response(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def create_app(config_path: str | Path | None = None, mockaudio: bool = False,
               mockllm: bool = False, mockart: bool = False) -> FastAPI:
    cfg = load_config(config_path)

    # Validate before uvicorn starts so misconfiguration ends in a single
    # clean error line and a non-zero exit code, not a lifespan traceback.
    try:
        backend = resolve_backend(cfg, mockaudio)
        llm_backend = resolve_llm(cfg, mockllm)
        _check_playout_viable(cfg, backend)
        from . import naming
        naming.load_lists(cfg.path("naming.words_dir"))
        workflows = {}
        if llm_backend != "off":
            # WorkflowError is a RuntimeError: a malformed workflow fails
            # startup with its message via the handler below.
            from .lyrics import load_workflows
            workflows = load_workflows(cfg)
        artworkflows = {}
        images_on = mockart or bool(str(cfg.get("art.api.base_url", "") or "").strip())
        if images_on and llm_backend == "off":
            # Art needs the LLM to write image prompts; the endpoint alone
            # is not enough. Warn (not fatal): music runs fine without art.
            log.warning("[art.api] configured but [llm] is disabled — track art "
                        "stays off (image-prompt writing needs the LLM)")
            images_on = False
        if images_on:
            from .art import load_art_workflows
            artworkflows = load_art_workflows(cfg)
        from .util import lock_data_dir
        lock_data_dir(cfg.path("library.data_dir"))
    except (StartupError, RuntimeError) as exc:
        log.error("startup failed: %s", exc)
        raise SystemExit(2) from None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        data_dir = cfg.path("library.data_dir")
        (data_dir / "music").mkdir(parents=True, exist_ok=True)
        (data_dir / "hls").mkdir(parents=True, exist_ok=True)
        db = DB(data_dir / "infiniradio.db")
        library.import_prompts(db, cfg)
        library.import_styles(db, cfg)
        library.seed_tags(db, cfg)
        # v14 art layout -> per-image rows (no-op once relocated; the SQL
        # side of migration 15 created the rows, this moves their files).
        library.migrate_legacy_art(db, cfg)
        # Nothing runs yet: fail jobs/tracks orphaned by a previous crash
        # (a 'running' job is never re-claimed, and its placeholder track
        # would count as "generating" forever).
        dj.reconcile_stale_jobs(db)

        generator = make_generator(cfg, backend)
        llm_client = make_llm_client(cfg, mockllm=mockllm) if llm_backend != "off" else None
        images_client = make_images_client(cfg, mockart=mockart) if images_on else None
        app.state.sessions = auth.Sessions(auth.get_server_secret(db, cfg))

        from .streamer.feeder import Feeder  # deferred import (avoids cycles)
        from . import telemetry as telemetry_mod
        rt = Runtime(cfg=cfg, db=db, generator=generator, llm=llm_client,
                     workflows=workflows, images=images_client,
                     artworkflows=artworkflows,
                     gen_lock=asyncio.Lock()
                     if bool(cfg.get("lyrics.serialize_with_render", False)) else None)
        rt.feeder = Feeder(rt)
        rt.telemetry = telemetry_mod.Telemetry(db)
        app.state.rt = rt

        tasks = [asyncio.create_task(dj.janitor_loop(rt)),
                 asyncio.create_task(rt.feeder.run()),
                 asyncio.create_task(telemetry_mod.telemetry_loop(rt))]
        if generator is not None or images_client is not None:
            # The GPU-side runner serves both music renders and art renders
            # (alternating); an art-only (playout) station still runs it.
            tasks.append(asyncio.create_task(dj.job_runner(rt)))
        if generator is not None:
            tasks.append(asyncio.create_task(dj.commission_loop(rt)))
        if llm_client is not None:
            tasks.append(asyncio.create_task(dj.lyrics_runner(rt)))

        log.info("infiniradio on air: backend=%s llm=%s workflows=%d art=%s(%d "
                 "workflow(s)) data=%s",
                 backend, llm_backend, len(workflows),
                 "on" if images_client is not None else "off",
                 len(artworkflows), data_dir)
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            await rt.feeder.stop()
            if generator is not None:
                await generator.close()
            if llm_client is not None:
                await llm_client.close()
            if images_client is not None:
                await images_client.close()
            db.conn.close()

    app = FastAPI(title="infiniradio", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_HardeningMiddleware)
    app.state.templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.include_router(public_router)
    app.include_router(admin_router)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    return app
