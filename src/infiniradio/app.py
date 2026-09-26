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

from . import dj, library
from .config import Config, load_config
from .db import DB
from .generator.base import Generator
from .generator.comfyui import ComfyUIGenerator
from .generator.mock import MockGenerator
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
    feeder: object | None = None
    started: datetime = field(default_factory=utcnow)


def make_generator(cfg: Config) -> Generator | None:
    backend = str(cfg.get("generator.backend", "mock"))
    if backend == "comfyui":
        return ComfyUIGenerator(cfg)
    if backend == "mock":
        return MockGenerator(cfg)
    if backend != "off":
        log.warning("unknown generator backend %r — treating as 'off'", backend)
    return None


def create_app(config_path: str | Path | None = None) -> FastAPI:
    cfg = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        data_dir = cfg.path("library.data_dir")
        (data_dir / "music").mkdir(parents=True, exist_ok=True)
        (data_dir / "hls").mkdir(parents=True, exist_ok=True)
        db = DB(data_dir / "infiniradio.db")
        library.import_prompts(db, cfg)

        generator = make_generator(cfg)
        app.state.sessions = auth.Sessions(auth.get_server_secret(db, cfg))

        from .streamer.feeder import Feeder  # deferred import (avoids cycles)
        rt = Runtime(cfg=cfg, db=db, generator=generator)
        rt.feeder = Feeder(rt)
        app.state.rt = rt

        tasks = [asyncio.create_task(dj.janitor_loop(rt)),
                 asyncio.create_task(rt.feeder.run())]
        if generator is not None:
            tasks.append(asyncio.create_task(dj.job_runner(rt)))
            tasks.append(asyncio.create_task(dj.commission_loop(rt)))

        log.info("infiniradio on air: backend=%s data=%s",
                 cfg.get("generator.backend"), data_dir)
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
            if isinstance(generator, ComfyUIGenerator):
                await generator.close()
            db.conn.close()

    app = FastAPI(title="infiniradio", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.include_router(public_router)
    app.include_router(admin_router)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    return app
