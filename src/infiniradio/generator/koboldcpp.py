"""KoboldCpp music generator, using its native music API.

KoboldCpp exposes a planning endpoint (``/api/extra/music/prepare``) and a
binary-audio endpoint (``/api/extra/music/generate``). The planning response
is passed back to the render endpoint; for YuE2 it includes the ABC score and
semantic prefix, while ACE-Step returns its song metadata (and optionally
audio codes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from ..abc_audit import Thresholds, abc_stats, audit_abc
from ..config import Config
from .base import GenRequest, GenResult, Generator, GenerationError

log = logging.getLogger(__name__)

MUSIC_FAMILIES = {"acestep", "yue2"}
AUDIO_FORMATS = {"wav", "mp3"}
STATUS_TIMEOUT_S = 4.0
STATUS_CACHE_TTL_S = 10.0


class KoboldCppGenerator(Generator):
    """Generate and download songs through a KoboldCpp server."""

    name = "koboldcpp"

    def __init__(self, cfg: Config, *, transport: httpx.AsyncBaseTransport | None = None):
        self.base = str(cfg.get("koboldcpp.url", "http://127.0.0.1:5001") or "").strip().rstrip("/")
        if not self.base:
            raise ValueError("[koboldcpp] url is empty")

        configured_family = str(cfg.get("koboldcpp.music_family", "auto") or "auto").strip().lower()
        if configured_family not in {"", "auto", *MUSIC_FAMILIES}:
            raise ValueError(
                "koboldcpp.music_family must be 'auto', 'acestep', or 'yue2', "
                f"got {configured_family!r}"
            )
        self.music_family = "" if configured_family in {"", "auto"} else configured_family

        self.audio_format = str(cfg.get("koboldcpp.audio_format", "wav") or "wav").strip().lower()
        if self.audio_format not in AUDIO_FORMATS:
            raise ValueError(
                f"koboldcpp.audio_format must be wav or mp3, got {self.audio_format!r}"
            )

        self.request_timeout_s = float(cfg.get("koboldcpp.request_timeout", 1800.0))
        if self.request_timeout_s <= 0:
            raise ValueError("koboldcpp.request_timeout must be positive")
        self.generate_codes = bool(cfg.get("koboldcpp.generate_codes", False))
        self.rewrite_caption = bool(cfg.get("koboldcpp.rewrite_caption", False))
        self.stereo = bool(cfg.get("koboldcpp.stereo", True))
        options = cfg.get("koboldcpp.options", {}) or {}
        if not isinstance(options, dict):
            raise ValueError("[koboldcpp.options] must be a TOML table")
        self.options: dict[str, Any] = dict(options)

        self._audit_enabled = bool(cfg.get("abc_audit.enabled", True))
        self._audit_attempts = max(1, int(cfg.get("abc_audit.max_attempts", 3)))
        self._audit_thresholds = Thresholds(
            max_chars=int(cfg.get("abc_audit.max_chars", 5000)),
            min_chars=int(cfg.get("abc_audit.min_chars", 300)),
        )

        headers = {"User-Agent": "infiniradio"}
        api_key = str(cfg.get("koboldcpp.api_key", "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = httpx.Timeout(
            connect=10.0, read=self.request_timeout_s,
            write=60.0, pool=10.0,
        )
        # Keep a trailing slash and use relative API paths so installations
        # mounted under a reverse-proxy path continue to work.
        self._client = httpx.AsyncClient(
            base_url=self.base + "/", headers=headers, timeout=timeout,
            transport=transport,
        )
        self._version_lock = asyncio.Lock()
        self._version_cache: tuple[float, dict[str, Any] | None, str | None] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _version_info(self) -> dict[str, Any]:
        """Read KoboldCpp's loaded music family, caching briefly for status UI."""
        now = time.monotonic()
        cached = self._version_cache
        if cached and now - cached[0] < STATUS_CACHE_TTL_S:
            if cached[1] is not None:
                return dict(cached[1])
            raise GenerationError(cached[2] or "KoboldCpp status request failed")

        async with self._version_lock:
            now = time.monotonic()
            cached = self._version_cache
            if cached and now - cached[0] < STATUS_CACHE_TTL_S:
                if cached[1] is not None:
                    return dict(cached[1])
                raise GenerationError(cached[2] or "KoboldCpp status request failed")

            try:
                response = await self._client.get(
                    "api/extra/version", timeout=httpx.Timeout(STATUS_TIMEOUT_S)
                )
                if response.status_code != 200:
                    raise GenerationError(
                        f"KoboldCpp /api/extra/version returned {response.status_code}: "
                        f"{_response_detail(response)}"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise GenerationError("KoboldCpp returned an invalid version response")
            except GenerationError as exc:
                self._version_cache = (time.monotonic(), None, str(exc))
                raise
            except (httpx.HTTPError, ValueError) as exc:
                message = f"cannot reach KoboldCpp at {self.base}: {exc}"
                self._version_cache = (time.monotonic(), None, message)
                raise GenerationError(message) from exc

            self._version_cache = (time.monotonic(), data, None)
            return dict(data)

    def _resolve_family(self, version: dict[str, Any]) -> str:
        if version.get("music") is False:
            raise GenerationError(
                "KoboldCpp is reachable, but reports no music model loaded "
                "(load an ACE-Step or YuE2 music model)"
            )

        reported = str(version.get("musicfamily", "") or "").strip().lower()
        if reported == "none":
            reported = ""
        if self.music_family and reported and reported != self.music_family:
            raise GenerationError(
                f"[koboldcpp] music_family = {self.music_family!r}, but the loaded "
                f"KoboldCpp music model is {reported!r}"
            )

        family = self.music_family or reported
        if family not in MUSIC_FAMILIES:
            raise GenerationError(
                "KoboldCpp did not report a loaded music family; set "
                "[koboldcpp] music_family to 'acestep' or 'yue2' and check the "
                "server's music model configuration"
            )
        return family

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.name, "url": self.base}
        try:
            version = await self._version_info()
            out["version"] = version.get("version")
            out["music"] = version.get("music")
            out["loaded_music_family"] = version.get("musicfamily")
            out["music_family"] = self._resolve_family(version)
            out["ok"] = True
        except GenerationError as exc:
            out["ok"] = False
            out["error"] = str(exc)
        return out

    def _request_payload(self, req: GenRequest, family: str, seed: int) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.options)
        payload.update({
            "caption": req.style,
            "lyrics": req.lyrics,
            "seed": int(seed),
            "musicfamily": family,
            "rewrite_caption": self.rewrite_caption,
            "stereo": self.stereo,
            "response_format": self.audio_format,
            "use_mp3": self.audio_format == "mp3",
        })
        if family == "yue2":
            # KoboldCpp's YuE2 planner returns the score and semantic prefix
            # needed by the subsequent render call.
            payload["gen_codes"] = True
            payload["cot"] = req.cot_mode if req.cot_mode in {"full", "melody", "off"} else "full"
            if req.abc and req.abc.strip():
                payload["abc"] = req.abc
            else:
                payload.pop("abc", None)
        else:
            payload["gen_codes"] = self.generate_codes
            payload.pop("abc", None)
            payload.pop("cot", None)
        return payload

    async def _prepare(self, req: GenRequest, family: str, seed: int) -> dict[str, Any]:
        payload = self._request_payload(req, family, seed)
        try:
            response = await self._client.post("api/extra/music/prepare", json=payload)
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"cannot reach KoboldCpp at {self.base} while preparing music: {exc}"
            ) from exc
        if response.status_code != 200:
            raise GenerationError(
                f"KoboldCpp /api/extra/music/prepare returned {response.status_code}: "
                f"{_response_detail(response)}"
            )
        try:
            plan = response.json()
        except ValueError as exc:
            raise GenerationError(
                "KoboldCpp /api/extra/music/prepare returned invalid JSON"
            ) from exc
        if not isinstance(plan, dict):
            raise GenerationError("KoboldCpp returned an invalid music plan")
        if plan.get("error"):
            raise GenerationError(f"KoboldCpp music planning failed: {plan['error']}")
        return plan

    @staticmethod
    def _score_from_plan(req: GenRequest, plan: dict[str, Any]) -> str:
        if req.abc and req.abc.strip():
            # Variations retain the parent track's score byte-for-byte.
            return req.abc
        score = plan.get("abc")
        return score if isinstance(score, str) else ""

    def _render_payload(
        self, req: GenRequest, plan: dict[str, Any], family: str, seed: int,
    ) -> dict[str, Any]:
        # The native API expects the prepared plan to be sent back to the
        # generate endpoint. Keep its family-specific fields (ABC, semantic
        # prefix, audio codes, BPM, duration, etc.) and reassert station-owned
        # values so the seed and user lyrics stay deterministic.
        payload = dict(plan)
        payload.update(self.options)
        payload.update({
            "caption": (plan.get("caption") if self.rewrite_caption else req.style) or req.style,
            "lyrics": req.lyrics if req.lyrics else str(plan.get("lyrics", "") or ""),
            "seed": int(seed),
            "musicfamily": family,
            "rewrite_caption": self.rewrite_caption,
            "stereo": self.stereo,
            "response_format": self.audio_format,
            "use_mp3": self.audio_format == "mp3",
        })
        if family == "yue2":
            payload["gen_codes"] = True
            payload["cot"] = req.cot_mode if req.cot_mode in {"full", "melody", "off"} else "full"
            score = self._score_from_plan(req, plan)
            if score:
                payload["abc"] = score
            else:
                payload.pop("abc", None)
        else:
            payload["gen_codes"] = self.generate_codes
            payload.pop("abc", None)
            payload.pop("cot", None)
        return payload

    async def _generate_audio(self, payload: dict[str, Any], workdir: Path) -> Path:
        workdir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            async with self._client.stream(
                "POST", "api/extra/music/generate", json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise GenerationError(
                        f"KoboldCpp /api/extra/music/generate returned "
                        f"{response.status_code}: {_body_detail(body)}"
                    )

                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if "json" in content_type:
                    body = await response.aread()
                    raise GenerationError(
                        "KoboldCpp /api/extra/music/generate returned an error response: "
                        f"{_body_detail(body)}"
                    )

                extension = _extension_from_content_type(content_type) or self.audio_format
                destination = workdir / f"original.{extension}"
                temporary = workdir / f"original.{extension}.part"
                total = 0
                prefix = bytearray()
                with temporary.open("wb") as output:
                    async for chunk in response.aiter_bytes(1 << 16):
                        if len(prefix) < 16:
                            prefix.extend(chunk[:16 - len(prefix)])
                        output.write(chunk)
                        total += len(chunk)

                if not total:
                    raise GenerationError(
                        "KoboldCpp returned an empty audio response; check that a complete "
                        "ACE-Step or YuE2 music model is loaded"
                    )

                first = bytes(prefix)
                if first.lstrip().startswith((b"{", b"[")):
                    raise GenerationError(
                        "KoboldCpp returned JSON instead of audio: "
                        f"{_body_detail(temporary.read_bytes()[:4096])}"
                    )
                detected = _extension_from_audio_header(first)
                if detected and detected != extension:
                    destination = workdir / f"original.{detected}"
                temporary.replace(destination)
                temporary = None
                return destination
        except GenerationError:
            raise
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"KoboldCpp music generation request failed at {self.base}: {exc}"
            ) from exc
        except OSError as exc:
            raise GenerationError(f"could not save KoboldCpp audio: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def generate(self, req: GenRequest, workdir: Path) -> GenResult:
        version = await self._version_info()
        family = self._resolve_family(version)
        seed = int(req.seed)
        attempts = 0
        last_reason = ""
        audited = (
            family == "yue2" and req.cot_mode == "full" and self._audit_enabled
            and req.abc is None and req.parent_track_id is None
        )

        if audited:
            plan: dict[str, Any] | None = None
            score = ""
            for attempt in range(1, self._audit_attempts + 1):
                attempts = attempt
                seed = int(req.seed) if attempt == 1 else random.getrandbits(31)
                planned_req = replace(req, seed=seed)
                plan = await self._prepare(planned_req, family, seed)
                score = self._score_from_plan(planned_req, plan)
                if not score:
                    raise GenerationError(
                        "KoboldCpp YuE2 planning returned no ABC score; check the loaded "
                        "music model and YuE2 score mode"
                    )
                accepted, last_reason = audit_abc(score, self._audit_thresholds)
                if accepted:
                    stats = abc_stats(score)
                    log.info(
                        "KoboldCpp YuE2 ABC audit: accepted on attempt %d/%d "
                        "(~%ds estimated, %d chars)",
                        attempt, self._audit_attempts,
                        stats["est_duration_s"], stats["chars"],
                    )
                    break
                log.info(
                    "KoboldCpp YuE2 ABC audit: attempt %d/%d rejected: %s",
                    attempt, self._audit_attempts, last_reason,
                )
            else:
                raise GenerationError(
                    f"ABC audit rejected {attempts} KoboldCpp candidate score(s); "
                    f"last rejection: {last_reason}"
                )
            assert plan is not None
        else:
            plan = await self._prepare(req, family, seed)
            score = self._score_from_plan(req, plan)
            if family == "yue2" and req.cot_mode != "off" and not score:
                raise GenerationError(
                    "KoboldCpp YuE2 planning returned no ABC score; check the loaded "
                    "music model and YuE2 score mode"
                )

        payload = self._render_payload(req, plan, family, seed)
        original = await self._generate_audio(payload, workdir)
        if score:
            (workdir / "score.abc").write_text(score, encoding="utf-8")

        info: dict[str, Any] = {
            "seed": seed,
            "koboldcpp_music_family": family,
        }
        if version.get("version"):
            info["koboldcpp_version"] = str(version["version"])
        if audited:
            info["abc_attempts"] = attempts
        return GenResult(original_path=original, abc_text=score, info=info)

    async def interrupt(self) -> bool:
        # KoboldCpp's documented abort endpoint controls text generation, not
        # the native music generation call. Don't send an unrelated abort.
        return False


def _extension_from_content_type(content_type: str) -> str:
    if content_type in {"audio/mpeg", "audio/mp3", "audio/x-mp3"}:
        return "mp3"
    if content_type in {"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"}:
        return "wav"
    return ""


def _extension_from_audio_header(data: bytes) -> str:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"ID3") or (
        len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
    ):
        return "mp3"
    return ""


def _body_detail(body: bytes) -> str:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body.decode("utf-8", errors="replace").strip()[:800] or "empty response"
    if isinstance(data, dict):
        detail = data.get("error") or data.get("detail") or data.get("msg") or data
        if isinstance(detail, dict):
            detail = detail.get("msg") or detail.get("error") or detail
        return str(detail)[:800]
    return str(data)[:800]


def _response_detail(response: httpx.Response) -> str:
    return _body_detail(response.content)
