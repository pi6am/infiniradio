"""LLM backend package: client registry, factory, and a small probe CLI.

The lyrics feature is optional: with [llm] enabled = false (the default)
no client is ever constructed and the station behaves exactly as before.
Backends are selected with [llm] backend; "mock" is testing-only and can
NOT be selected there — run the server with --mockllm instead (the same
rule that keeps --mockaudio out of config).

Probe the configured endpoint without starting the server:

    python -m infiniradio.llm --status
    python -m infiniradio.llm --ask "reply with the single word: pong"
"""

from __future__ import annotations

from typing import Type

from ..config import Config
from .base import LLMClient, LLMError, Message, Reply
from .images import ImagesClient, ImagesError, MockImagesClient
from .mock import MockLLMClient
from .openai_compat import OpenAICompatClient

BACKENDS: dict[str, Type[LLMClient]] = {
    "openai": OpenAICompatClient,
}


class LLMConfigError(RuntimeError):
    """Fatal [llm] misconfiguration; the server must not start."""


def make_images_client(cfg: Config, mockart: bool = False):
    """Images backend for track art, or None when unconfigured. The mock
    is reachable only through --mockart (the --mockaudio/--mockllm rule:
    testing clients never leak into a config-selected backend)."""
    if mockart:
        return MockImagesClient()
    if not str(cfg.get("art.api.base_url", "") or "").strip():
        return None
    if not str(cfg.get("art.api.model", "") or "").strip():
        return None
    return ImagesClient(cfg)


def make_llm_client(cfg: Config, mockllm: bool = False) -> LLMClient | None:
    """None when no LLM is configured (lyrics workflows stay disabled)."""
    if mockllm:
        return MockLLMClient()
    if not bool(cfg.get("llm.enabled", False)):
        return None
    backend = str(cfg.get("llm.backend", "openai") or "openai").strip().lower()
    if backend == "mock":
        raise LLMConfigError(
            '[llm] backend = "mock" is testing-only and cannot be set in '
            "config; start the server with --mockllm if you want scripted "
            "LLM responses."
        )
    cls = BACKENDS.get(backend)
    if cls is None:
        raise LLMConfigError(
            f"unknown [llm] backend {backend!r} (expected "
            + " or ".join(map(repr, BACKENDS)) + ")"
        )
    return cls(cfg)
