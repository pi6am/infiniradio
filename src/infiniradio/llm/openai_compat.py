"""OpenAI-compatible chat completions client.

One client covers every OpenAI-shaped endpoint in practical use today —
OpenRouter, DeepSeek, GLM, Kimi, koboldcpp, llama.cpp/vLLM/Ollama servers
all speak ``POST {base_url}/chat/completions`` — so "modular backends" here
means a registry entry, not a new client, until a genuinely different API
shows up.

Transport trouble (429, 5xx, connect/read timeouts) is retried with
exponential backoff, honoring Retry-After when the server sends one. Other
4xx responses and malformed/empty completions raise LLMError immediately:
content-level recovery belongs to the workflow stage retry, which can
re-prompt with feedback — retrying the same doomed request here would just
burn the attempt budget invisibly.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from ..config import Config
from .base import LLMClient, LLMError, Message, Reply

log = logging.getLogger(__name__)

BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0

# Health pings (/models for the admin dashboard) are not generation calls:
# a down endpoint must never stall the page for the full generation
# timeout. Short deadline, and the verdict (including failures) is cached
# for a TTL so repeated page loads read the last result instead of
# re-probing.
STATUS_TIMEOUT_S = 4.0
STATUS_CACHE_TTL_S = 30.0


class OpenAICompatClient(LLMClient):
    name = "openai"

    def __init__(self, cfg: Config):
        self.base_url = str(cfg.get("llm.base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("[llm] base_url is empty — set llm.base_url in config.toml")
        self.model = str(cfg.get("llm.model", "") or "")
        self.temperature = cfg.get("llm.temperature", 1.0)
        self.max_output_tokens = int(cfg.get("llm.max_output_tokens", 8192))
        self.transport_retries = max(0, int(cfg.get("llm.transport_retries", 3)))
        headers = {"User-Agent": "infiniradio"}
        api_key = str(cfg.get("llm.api_key", "") or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout_s = float(cfg.get("llm.timeout_s", 900.0))
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
        )
        self._status_cache: tuple[float, dict[str, Any]] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    # -- one chat completion ---------------------------------------------------

    async def chat(self, messages: list[Message], *, model: str | None = None,
                   temperature: float | None = None,
                   max_tokens: int | None = None) -> Reply:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_output_tokens,
            "stream": False,
        }
        attempts = self.transport_retries + 1
        for attempt in range(1, attempts + 1):
            retry_after: float | None = None
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in (408, 429) or resp.status_code >= 500:
                    if attempt < attempts:
                        if resp.status_code == 429:
                            retry_after = _retry_after(resp)
                        await _backoff_sleep(attempt, retry_after, resp.status_code)
                        continue
                if resp.status_code != 200:
                    raise LLMError(
                        f"{self.base_url}/chat/completions returned "
                        f"{resp.status_code}: {resp.text[:400]}")
                return self._parse(resp.json())
            except httpx.HTTPError as exc:
                if attempt < attempts:
                    _backoff_sleep(attempt, None, str(exc))
                    continue
                raise LLMError(f"cannot reach LLM at {self.base_url}: {exc}") from exc
        raise LLMError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(data: dict[str, Any]) -> Reply:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMError(f"malformed completion reply: {str(data)[:400]}")
        content = (choices[0].get("message") or {}).get("content")
        truncated = str(choices[0].get("finish_reason") or "") == "length"
        # An empty reply is normally fatal; an empty TRUNCATED one may just
        # mean the whole budget went to thinking — the engine's continuation
        # can still recover it.
        if not isinstance(content, str) or (not content.strip() and not truncated):
            raise LLMError(f"empty completion content: {str(data)[:400]}")
        usage = data.get("usage") or {}
        return Reply(content=content, model=str(data.get("model", "")),
                     truncated=truncated,
                     usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))})

    async def status(self) -> dict[str, Any]:
        """Health info for the admin dashboard, cached for STATUS_CACHE_TTL_S
        (failures included): the page reads the last verdict instead of
        waiting on a probe, and the probe itself carries a short deadline —
        a down endpoint costs at most one STATUS_TIMEOUT_S wait per TTL
        window, not the generation timeout."""
        now = time.monotonic()
        if self._status_cache is not None and now - self._status_cache[0] < STATUS_CACHE_TTL_S:
            return dict(self._status_cache[1])
        out: dict[str, Any] = {"backend": self.name, "url": self.base_url,
                               "model": self.model}
        try:
            r = await self._client.get("/models",
                                       timeout=httpx.Timeout(STATUS_TIMEOUT_S))
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)]
            out["ok"] = True
            out["models"] = models[:20]
            if self.model and models and self.model not in models:
                out["warning"] = f"configured model {self.model!r} not in /models list"
        except Exception as exc:
            out["ok"] = False
            out["error"] = str(exc)[:300]
        self._status_cache = (now, out)
        return dict(out)


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After", "")
    try:
        return min(float(raw), BACKOFF_CAP_S) if float(raw) > 0 else None
    except ValueError:
        return None


async def _backoff_sleep(attempt: int, retry_after: float | None, why: Any) -> None:
    delay = retry_after if retry_after is not None else min(
        BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
    delay *= 0.5 + random.random()  # full jitter
    log.warning("llm attempt %d failed (%s) — retrying in %.1fs", attempt, why, delay)
    await asyncio.sleep(delay)
