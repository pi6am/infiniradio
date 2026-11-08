"""OpenAI-compatible image generation client.

``POST {base_url}/images/generations`` — the same shape every provider
exposes (OpenAI, gpt-image, CogView, and SD/ComfyUI bridging proxies).
The station's typical endpoint is a small proxy in front of the same
ComfyUI box that renders music, which is why art renders must interleave
with music renders on the GPU-side runner (see dj.job_runner) instead of
running whenever they like.

Responses come back as b64_json when asked for, with a URL fallback for
providers that only return links. Whatever bytes arrive (png/jpeg/webp)
are passed through; library.store_art normalizes the master to PNG via
ffmpeg. Transport trouble (429, 5xx, timeouts) retries with backoff —
the same policy as the chat client.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from ..config import Config
from .openai_compat import _backoff_sleep, _retry_after

log = logging.getLogger(__name__)


class ImagesError(RuntimeError):
    """A generation attempt failed (surfaced as the art job's error)."""


# A valid 1x1 opaque PNG — what the mock client "generates", and what the
# test suite scales into thumbnails. Small on purpose: the mock must be
# instant and never touch the network.
MOCK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class ImagesClient:
    """One OpenAI-shaped images endpoint."""

    name = "openai-images"

    def __init__(self, cfg: Config):
        self.base_url = str(cfg.get("art.api.base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("[art.api] base_url is empty — set art.api.base_url in config.toml")
        self.model = str(cfg.get("art.api.model", "") or "")
        if not self.model:
            raise ValueError("[art.api] model is empty — set art.api.model in config.toml")
        self.send_seed = bool(cfg.get("art.api.send_seed", False))
        self.transport_retries = max(0, int(cfg.get("art.api.transport_retries", 3)))
        headers = {"User-Agent": "infiniradio"}
        api_key = str(cfg.get("art.api.api_key", "") or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout_s = float(cfg.get("art.api.timeout_s", 300.0))
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, prompt: str, *, size: str = "1024x1024",
                       seed: int | None = None) -> bytes:
        """Raw image bytes (png/jpeg/webp — whatever the provider sent)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
        }
        if self.send_seed and seed is not None:
            # Not part of the OpenAI schema; useful for proxies that map it
            # onto a ComfyUI seed. Off by default so strict providers never
            # see an unknown field.
            payload["seed"] = int(seed)
        attempts = self.transport_retries + 1
        for attempt in range(1, attempts + 1):
            retry_after: float | None = None
            try:
                resp = await self._client.post("/images/generations", json=payload)
                if resp.status_code in (408, 429) or resp.status_code >= 500:
                    if attempt < attempts:
                        if resp.status_code == 429:
                            retry_after = _retry_after(resp)
                        await _backoff_sleep(attempt, retry_after, resp.status_code)
                        continue
                if resp.status_code != 200:
                    raise ImagesError(
                        f"{self.base_url}/images/generations returned "
                        f"{resp.status_code}: {resp.text[:400]}")
                return await self._extract(resp.json())
            except httpx.HTTPError as exc:
                if attempt < attempts:
                    await _backoff_sleep(attempt, None, str(exc))
                    continue
                raise ImagesError(f"cannot reach images API at {self.base_url}: {exc}") from exc
        raise ImagesError("unreachable")  # pragma: no cover

    async def _extract(self, data: dict[str, Any]) -> bytes:
        items = data.get("data") or []
        if not items or not isinstance(items[0], dict):
            raise ImagesError(f"malformed images reply: {str(data)[:400]}")
        b64 = items[0].get("b64_json")
        if isinstance(b64, str) and b64.strip():
            try:
                return base64.b64decode(b64)
            except (ValueError, TypeError) as exc:
                raise ImagesError(f"invalid b64_json payload: {exc}") from exc
        url = items[0].get("url")
        if isinstance(url, str) and url.strip():
            # Absolute URLs pass through the client's base_url untouched.
            return await self._download(url.strip())
        raise ImagesError(f"reply carries neither b64_json nor url: {str(data)[:400]}")

    async def _download(self, url: str) -> bytes:
        for attempt in range(1, self.transport_retries + 2):
            try:
                resp = await self._client.get(url)
                if resp.status_code in (408, 429) or resp.status_code >= 500:
                    if attempt <= self.transport_retries:
                        await _backoff_sleep(attempt, _retry_after(resp)
                                             if resp.status_code == 429 else None,
                                             resp.status_code)
                        continue
                if resp.status_code != 200:
                    raise ImagesError(f"image URL {url} returned {resp.status_code}")
                if not resp.content:
                    raise ImagesError(f"image URL {url} returned an empty body")
                return resp.content
            except httpx.HTTPError as exc:
                if attempt <= self.transport_retries:
                    await _backoff_sleep(attempt, None, str(exc))
                    continue
                raise ImagesError(f"cannot download image from {url}: {exc}") from exc
        raise ImagesError("unreachable")  # pragma: no cover


class MockImagesClient:
    """Testing-only (--mockart): returns a fixed PNG, never touches the network."""

    name = "mock-images"

    async def generate(self, prompt: str, *, size: str = "1024x1024",
                       seed: int | None = None) -> bytes:
        return MOCK_PNG

    async def close(self) -> None:
        pass
