"""`python -m infiniradio.llm` — probe the configured LLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ..config import load_config
from . import make_llm_client
from .base import LLMError, Message


async def _run(args) -> int:
    cfg = load_config(args.config)
    client = make_llm_client(cfg, mockllm=args.mock)
    if client is None:
        print("no LLM configured (llm.enabled = false) — pass --mock to probe "
              "the scripted mock", file=sys.stderr)
        return 2
    try:
        if args.ask:
            reply = await client.chat([Message(role="user", content=args.ask)],
                                      max_tokens=args.max_tokens)
            print(reply.content)
            if args.verbose:
                print(f"[model={reply.model} usage={reply.usage}]", file=sys.stderr)
        else:
            print(json.dumps(await client.status(), indent=2))
        return 0
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m infiniradio.llm",
        description="probe the configured [llm] endpoint without starting the server")
    ap.add_argument("-c", "--config", default=None, help="path to config.toml")
    ap.add_argument("--status", action="store_true",
                    help="ping the endpoint and print status JSON (default)")
    ap.add_argument("--ask", default=None, help="send one user message, print the reply")
    ap.add_argument("--mock", action="store_true", help="probe the mock client")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
