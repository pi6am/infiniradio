"""KoboldCpp native music backend: API flow, formats, audit and factory dispatch."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest import mock

import httpx

import _helpers

_helpers.ensure_repo()

from infiniradio.app import make_generator, resolve_backend
from infiniradio.generator.base import GenRequest, GenerationError
from infiniradio.generator.koboldcpp import KoboldCppGenerator


def _config(base_cfg, *, family: str = "auto", audio_format: str = "wav",
            generate_codes: bool = False, options: dict | None = None,
            audit: bool = True, audit_attempts: int = 3):
    data = dict(base_cfg._d)
    data["generator"] = {**data.get("generator", {}), "backend": "koboldcpp"}
    data["library"] = {**data.get("library", {}),
                       "data_dir": str(base_cfg.path("library.data_dir"))}
    data["koboldcpp"] = {
        **data.get("koboldcpp", {}),
        "url": "http://koboldcpp.test:5001",
        "api_key": "test-password",
        "music_family": family,
        "audio_format": audio_format,
        "request_timeout": 60.0,
        "generate_codes": generate_codes,
        "rewrite_caption": False,
        "stereo": True,
        "options": options or {},
    }
    data["abc_audit"] = {
        **data.get("abc_audit", {}),
        "enabled": audit,
        "max_attempts": audit_attempts,
    }
    return type(base_cfg)(data, base_cfg.base_dir)


def _valid_abc() -> str:
    return (
        "X:1\nT:KoboldCpp test\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\n"
        + "CDEF GABc|" * 40
    )


def _version(family: str, *, music: bool = True) -> dict:
    return {
        "result": "KoboldCpp",
        "version": "2026.11.01",
        "music": music,
        "musicfamily": family if music else "none",
    }


async def _test_yue2_plan_render_and_status(base_cfg, tmp: Path) -> None:
    requests: list[httpx.Request] = []
    plan_score = _valid_abc()
    audio = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"test-audio"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/extra/version":
            return httpx.Response(200, json=_version("yue2"))
        if request.url.path == "/api/extra/music/prepare":
            body = json.loads(request.content)
            assert body["caption"] == "slow synthwave, instrumental"
            assert body["lyrics"] == "[Instrumental]"
            assert body["seed"] == 12345
            assert body["musicfamily"] == "yue2"
            assert body["cot"] == "full" and body["gen_codes"] is True
            assert "abc" not in body
            return httpx.Response(200, json={
                "caption": body["caption"], "lyrics": body["lyrics"],
                "seed": body["seed"], "musicfamily": "yue2",
                "abc": plan_score, "semantic_prefix": [11, 22, 33],
                "num_inference_steps": 8,
            })
        if request.url.path == "/api/extra/music/generate":
            body = json.loads(request.content)
            assert body["musicfamily"] == "yue2"
            assert body["seed"] == 12345
            assert body["cot"] == "full" and body["gen_codes"] is True
            assert body["abc"] == plan_score
            assert body["semantic_prefix"] == [11, 22, 33]
            assert body["response_format"] == "wav"
            return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})
        raise AssertionError(f"unexpected KoboldCpp request: {request.url}")

    cfg = _config(base_cfg)
    gen = KoboldCppGenerator(cfg, transport=httpx.MockTransport(handler))
    try:
        status = await gen.status()
        assert status["ok"] is True and status["music_family"] == "yue2", status
        result = await gen.generate(
            GenRequest(style="slow synthwave, instrumental", lyrics="[Instrumental]", seed=12345),
            tmp / "yue2-track",
        )
    finally:
        await gen.close()

    assert result.original_path.name == "original.wav"
    assert result.original_path.read_bytes() == audio
    assert result.abc_text == plan_score
    assert (tmp / "yue2-track" / "score.abc").read_text(encoding="utf-8") == plan_score
    assert result.info["seed"] == 12345
    assert result.info["koboldcpp_music_family"] == "yue2"
    assert result.info["abc_attempts"] == 1
    assert [r.url.path for r in requests].count("/api/extra/version") == 1
    assert all(r.headers.get("Authorization") == "Bearer test-password" for r in requests)
    print("KoboldCpp YuE2: auto-discovery, prepare/render payload, ABC and auth")


async def _test_acestep_mp3_and_options(base_cfg, tmp: Path) -> None:
    requests: list[httpx.Request] = []
    audio = b"ID3\x03\x00\x00" + b"test-mp3-audio"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/extra/version":
            return httpx.Response(200, json=_version("acestep"))
        if request.url.path == "/api/extra/music/prepare":
            body = json.loads(request.content)
            assert body["musicfamily"] == "acestep"
            assert body["gen_codes"] is True
            assert body["duration"] == 75
            assert body["inference_steps"] == 8
            return httpx.Response(200, json={
                "caption": body["caption"], "lyrics": body["lyrics"],
                "seed": body["seed"], "musicfamily": "acestep",
                "bpm": 112, "duration": 75, "keyscale": "D minor",
                "timesignature": "4",
            })
        if request.url.path == "/api/extra/music/generate":
            body = json.loads(request.content)
            assert body["musicfamily"] == "acestep"
            assert body["response_format"] == "mp3" and body["use_mp3"] is True
            assert body["gen_codes"] is True
            assert body["duration"] == 75 and body["bpm"] == 112
            assert "abc" not in body and "cot" not in body
            return httpx.Response(200, content=audio, headers={"content-type": "audio/mpeg"})
        raise AssertionError(f"unexpected KoboldCpp request: {request.url}")

    cfg = _config(base_cfg,
        family="acestep", audio_format="mp3", generate_codes=True,
        options={"duration": 75, "inference_steps": 8}, audit=True,
    )
    gen = KoboldCppGenerator(cfg, transport=httpx.MockTransport(handler))
    try:
        result = await gen.generate(
            GenRequest(style="warm electronic instrumental", lyrics="[Instrumental]", seed=444),
            tmp / "ace-track",
        )
    finally:
        await gen.close()

    assert result.original_path.name == "original.mp3"
    assert result.original_path.read_bytes() == audio
    assert result.abc_text == ""
    assert not (tmp / "ace-track" / "score.abc").exists()
    assert result.info["koboldcpp_music_family"] == "acestep"
    print("KoboldCpp ACE-Step: MP3 response and configurable API options")


async def _test_yue2_audit_retry(base_cfg, tmp: Path) -> None:
    prepared: list[dict] = []
    rendered: list[dict] = []
    valid = _valid_abc()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/extra/version":
            return httpx.Response(200, json=_version("yue2"))
        if request.url.path == "/api/extra/music/prepare":
            body = json.loads(request.content)
            prepared.append(body)
            score = "X:1\nK:C\nc" if len(prepared) == 1 else valid
            return httpx.Response(200, json={
                "musicfamily": "yue2", "seed": body["seed"],
                "abc": score, "semantic_prefix": [len(prepared)],
            })
        if request.url.path == "/api/extra/music/generate":
            rendered.append(json.loads(request.content))
            return httpx.Response(
                200, content=b"RIFF" + b"\x00\x00\x00\x00" + b"WAVEaudio",
                headers={"content-type": "audio/wav"},
            )
        raise AssertionError(f"unexpected KoboldCpp request: {request.url}")

    cfg = _config(base_cfg, family="yue2", audit=True, audit_attempts=2)
    gen = KoboldCppGenerator(cfg, transport=httpx.MockTransport(handler))
    try:
        with mock.patch(
            "infiniradio.generator.koboldcpp.random.getrandbits", return_value=67890
        ):
            result = await gen.generate(
                GenRequest(style="folk", lyrics="[Verse]", seed=123),
                tmp / "audit-track",
            )
    finally:
        await gen.close()

    assert len(prepared) == 2
    assert [p["seed"] for p in prepared] == [123, 67890]
    assert len(rendered) == 1 and rendered[0]["seed"] == 67890
    assert result.info["seed"] == 67890
    assert result.info["abc_attempts"] == 2
    assert result.abc_text == valid
    print("KoboldCpp YuE2: rejected score re-planned with a new seed")


async def _test_unavailable_music_model(base_cfg, tmp: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/extra/version"
        return httpx.Response(200, json=_version("none", music=False))

    cfg = _config(base_cfg)
    gen = KoboldCppGenerator(cfg, transport=httpx.MockTransport(handler))
    try:
        status = await gen.status()
        assert status["ok"] is False and "no music model loaded" in status["error"]
        try:
            await gen.generate(GenRequest(style="test", lyrics="", seed=1), tmp / "no-model")
            raise AssertionError("expected KoboldCpp model validation to fail")
        except GenerationError as exc:
            assert "no music model loaded" in str(exc)
    finally:
        await gen.close()
    print("KoboldCpp status: missing music model is reported clearly")


def run() -> None:
    base_cfg, tmp = _helpers.isolated_config()
    cfg = _config(base_cfg)
    try:
        assert resolve_backend(cfg, mockaudio=False) == "koboldcpp"
        factory = make_generator(cfg, "koboldcpp")
        assert isinstance(factory, KoboldCppGenerator)
        assert factory.name == "koboldcpp"
        asyncio.run(factory.close())
        print("generator dispatch: koboldcpp backend selected")

        asyncio.run(_test_yue2_plan_render_and_status(base_cfg, tmp))
        asyncio.run(_test_acestep_mp3_and_options(base_cfg, tmp))
        asyncio.run(_test_yue2_audit_retry(base_cfg, tmp))
        asyncio.run(_test_unavailable_music_model(base_cfg, tmp))
        print("test_koboldcpp: ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run()
