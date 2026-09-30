"""LLM health ping: short deadline and TTL cache — a down or black-hole
endpoint must never stall the admin dashboard with the generation timeout.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from types import SimpleNamespace

import _helpers

_helpers.ensure_repo()

from infiniradio.llm.openai_compat import (STATUS_CACHE_TTL_S, STATUS_TIMEOUT_S,
                                           OpenAICompatClient)


def _fake_cfg(base_url: str):
    return SimpleNamespace(get=lambda k, d=None: {"llm.base_url": base_url}.get(k, d))


async def _main() -> None:
    # Black hole: accepts the TCP connection, never answers (the worst case
    # for the old code: the /models ping inherited the 900s read timeout).
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def hold():
        conns = []
        try:
            while True:
                conns.append(srv.accept()[0])  # accept and go silent
        except OSError:
            pass

    threading.Thread(target=hold, daemon=True).start()
    client = OpenAICompatClient(_fake_cfg(f"http://127.0.0.1:{port}/v1"))
    client2 = OpenAICompatClient(_fake_cfg("http://127.0.0.1:9/v1"))  # refused
    try:
        t0 = time.monotonic()
        s1 = await client.status()
        d1 = time.monotonic() - t0
        assert not s1["ok"] and d1 < STATUS_TIMEOUT_S + 2.0, (s1, d1)
        t0 = time.monotonic()
        assert not (await client.status())["ok"] and time.monotonic() - t0 < 0.25
        print(f"1. black-hole endpoint: first ping {d1:.1f}s (bounded), cached ~0ms")

        assert not (await client2.status())["ok"]
        t0 = time.monotonic()
        assert not (await client2.status())["ok"] and time.monotonic() - t0 < 0.25
        print("2. refused connection: instant failure, cached likewise")

        client._status_cache = (time.monotonic() - STATUS_CACHE_TTL_S - 1,
                                {"stale": True})
        t0 = time.monotonic()
        s3 = await client.status()
        assert not s3["ok"] and "stale" not in s3
        assert time.monotonic() - t0 < STATUS_TIMEOUT_S + 2.0
        print(f"3. expired cache re-probes within the short deadline "
              f"(TTL {STATUS_CACHE_TTL_S:.0f}s)")
    finally:
        await client.close()
        await client2.close()
        srv.close()


def run() -> None:
    asyncio.run(_main())
    print("test_llm_status: ALL PASSED")


if __name__ == "__main__":
    run()
