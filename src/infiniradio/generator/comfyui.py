"""ComfyUI generator: submits the YuE2 workflow over ComfyUI's HTTP API.

The workflow file (exported in API format) is used as a template and patched
in memory at submit time; it is never modified on disk:

  * style / lyrics / seed nodes get the request values
  * the "ABC planning" boolean selects fresh planning (true) vs re-rendering a
    supplied score (false) — the latter is how variations are made
  * the SaveAudioAdvanced node's format is set from config (flac | mp3)

Fresh songs run as TWO prompts (ABC audit, see abc_audit.py): the first
submits only the subgraph upstream of the ABC save node (the planner, its
inputs and the model it needs — the render branch is pruned away entirely),
so only the cheap text-only planning executes; the score is audited locally,
and on retryable rejections a new seed is planned. The accepted score is
then rendered by submitting the workflow the same way variations are
(plan=false, supplied ABC). Variations and audit-disabled setups use the
original single pass.

After completion, output files listed in /history are downloaded via /view
into the track directory, so infiniradio owns its own copies and nothing
assumes a shared filesystem with the ComfyUI machine.
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
from urllib.parse import quote

import httpx

from ..abc_audit import Thresholds, abc_stats, audit_abc
from ..config import Config
from .base import GenRequest, GenResult, Generator, GenerationError

log = logging.getLogger(__name__)

AUDIO_EXTS = {".flac", ".mp3", ".wav", ".m4a", ".ogg", ".opus"}


class ComfyUIGenerator(Generator):
    name = "comfyui"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = str(cfg.get("comfyui.url", "http://127.0.0.1:8188")).rstrip("/")
        self.vars: dict[str, str] = dict(cfg.get("comfyui.workflow_vars", {}) or {})
        self.audio_format = str(cfg.get("comfyui.audio_format", "flac")).lower()
        if self.audio_format not in ("flac", "mp3"):
            raise ValueError(f"comfyui.audio_format must be flac or mp3, got {self.audio_format!r}")
        self.poll_interval = float(cfg.get("comfyui.poll_interval", 2.0))
        self.deadline_s = float(cfg.get("comfyui.request_timeout", 1800.0))
        self.cleanup_url = str(cfg.get("comfyui.cleanup_url", "") or "").rstrip("/")
        self.cleanup_token = str(cfg.get("comfyui.cleanup_token", "") or "")
        self.cleanup_every_n = max(1, int(cfg.get("comfyui.cleanup_every_n", 1)))
        self._jobs_since_cleanup = 0
        # Two-pass quality gate (abc_audit): fresh songs plan + audit their
        # score before the expensive audio render. Needs the "abc_save" node
        # id so the planning pass can be pruned to its upstream subgraph.
        self._audit_enabled = bool(cfg.get("abc_audit.enabled", True))
        self._audit_attempts = max(1, int(cfg.get("abc_audit.max_attempts", 3)))
        self._audit_thresholds = Thresholds(
            max_chars=int(cfg.get("abc_audit.max_chars", 5000)),
            min_chars=int(cfg.get("abc_audit.min_chars", 300)),
        )
        if self._audit_enabled and "abc_save" not in self.vars:
            log.warning("abc_audit.enabled but no 'abc_save' node in "
                        "[comfyui.workflow_vars] — falling back to single-pass "
                        "(unaudited) generation")
        self._client_id = f"infiniradio-{random.getrandbits(48):012x}"
        auth = None
        user, pw = cfg.get("comfyui.username", ""), cfg.get("comfyui.password", "")
        if user:
            auth = httpx.BasicAuth(user, pw)
        self._client = httpx.AsyncClient(
            base_url=self.base, auth=auth,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
            headers={"User-Agent": "infiniradio"},
        )
        self._sidecar = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

    async def close(self) -> None:
        await self._client.aclose()
        await self._sidecar.aclose()

    # -- workflow templating -------------------------------------------------

    def _load_workflow(self) -> dict[str, Any]:
        path = self.cfg.path("comfyui.workflow")
        try:
            wf = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GenerationError(f"workflow file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise GenerationError(f"workflow file is not valid JSON: {path}: {exc}") from exc
        if not isinstance(wf, dict) or not wf:
            raise GenerationError(f"workflow file has no nodes: {path}")
        return wf

    def _node(self, wf: dict, var: str) -> dict:
        nid = self.vars.get(var)
        node = wf.get(str(nid)) if nid is not None else None
        if not isinstance(node, dict):
            raise GenerationError(
                f"workflow node for {var!r} (id {nid!r}) not found — check [comfyui.workflow_vars]"
            )
        return node

    def _patch_workflow(self, wf: dict, req: GenRequest) -> dict[str, Any]:
        self._node(wf, "style")["inputs"]["value"] = req.style
        self._node(wf, "lyrics")["inputs"]["value"] = req.lyrics
        self._node(wf, "seed")["inputs"]["seed"] = int(req.seed)
        # plan=true -> generate a fresh ABC; plan=false -> use the ABC text
        # wired into the "empty ABC" node (our variation hook, and the
        # second pass of audited fresh generation).
        self._node(wf, "plan")["inputs"]["value"] = req.abc is None
        self._node(wf, "abc")["inputs"]["value"] = req.abc or ""
        # Absent in the pruned planning-pass graph; patched only when present.
        audio_save = wf.get(str(self.vars.get("audio_save")))
        if isinstance(audio_save, dict):
            audio_inputs = audio_save["inputs"]
            audio_inputs["format"] = self.audio_format
            if self.audio_format == "mp3":
                audio_inputs["format.quality"] = str(self.cfg.get("comfyui.audio_quality", "V0"))
            else:
                audio_inputs.pop("format.quality", None)
        return wf

    def _abc_only_subgraph(self, wf: dict) -> dict[str, Any]:
        """The planning pass's graph: the ABC save node and its ancestors.

        Removing nodes from the in-memory prompt is exactly what ComfyUI's
        own "muted" API export does (verified against a hand-muted export):
        a node absent from the prompt cannot execute, on any build.
        """
        root = str(self.vars["abc_save"])
        if root not in wf:
            raise GenerationError(
                f"workflow node for 'abc_save' (id {root!r}) not found — check"
                f" [comfyui.workflow_vars]")
        keep: set[str] = set()
        stack = [root]
        while stack:
            nid = stack.pop()
            if nid in keep:
                continue
            keep.add(nid)
            for value in (wf[nid].get("inputs") or {}).values():
                if (isinstance(value, list) and len(value) == 2
                        and isinstance(value[0], str) and value[0] in wf):
                    stack.append(value[0])
        return {nid: node for nid, node in wf.items() if nid in keep}

    # -- sidecar ---------------------------------------------------------------

    async def cleanup_now(self) -> dict[str, Any] | None:
        if not self.cleanup_url:
            return None
        try:
            r = await self._sidecar.post(
                f"{self.cleanup_url}/cleanup", json={},
                headers={"X-Token": self.cleanup_token},
            )
            r.raise_for_status()
            data = r.json()
            log.info("sidecar cleanup: deleted %s files (%s bytes freed)",
                     data.get("deleted"), data.get("bytes_freed"))
            return data
        except Exception as exc:  # non-fatal by design
            log.warning("sidecar cleanup failed: %s", exc)
            return {"error": str(exc)}

    async def _maybe_cleanup(self) -> None:
        if not self.cleanup_url:
            return
        if self._jobs_since_cleanup < self.cleanup_every_n:
            return
        self._jobs_since_cleanup = 0
        await self.cleanup_now()

    async def sidecar_status(self) -> dict[str, Any] | None:
        if not self.cleanup_url:
            return None
        try:
            r = await self._sidecar.get(
                f"{self.cleanup_url}/status", headers={"X-Token": self.cleanup_token}
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": str(exc)}

    # -- generation --------------------------------------------------------------

    async def generate(self, req: GenRequest, workdir: Path) -> GenResult:
        await self._maybe_cleanup()
        self._jobs_since_cleanup += 1

        if req.abc is None and self._audit_enabled and "abc_save" in self.vars:
            return await self._generate_with_audit(req, workdir)
        return await self._run_full(req, workdir)

    async def _generate_with_audit(self, req: GenRequest, workdir: Path) -> GenResult:
        """(plan ABC -> audit) until a score passes, then render it.

        Retries plan with a fresh seed on rejection; a rejected candidate
        costs one cheap text-only pass instead of a full render. When every
        attempt is rejected the job fails (the DJ retries it later with a
        different prompt or seed) rather than airing a runaway score.
        """
        attempts, last_reason, abc_text, seed = 0, "", "", req.seed
        for attempt in range(1, self._audit_attempts + 1):
            attempts = attempt
            seed = req.seed if attempt == 1 else random.getrandbits(31)
            abc_text = await self._run_abc_pass(replace(req, seed=seed))
            ok, last_reason = audit_abc(abc_text, self._audit_thresholds)
            if ok:
                s = abc_stats(abc_text)
                log.info("ABC audit: score accepted on attempt %d/%d "
                         "(~%ds estimated, %d chars)",
                         attempt, self._audit_attempts, s["est_duration_s"], s["chars"])
                break
            log.info("ABC audit: attempt %d/%d rejected: %s",
                     attempt, self._audit_attempts, last_reason)
        else:
            raise GenerationError(
                f"ABC audit rejected {attempts} candidate score(s); "
                f"last rejection: {last_reason}")
        result = await self._run_full(replace(req, seed=seed, abc=abc_text), workdir)
        result.info["abc_attempts"] = attempts
        return result

    async def _run_abc_pass(self, req: GenRequest) -> str:
        """Cheap pass: plan an ABC score, no audio render. Returns the text."""
        wf = self._patch_workflow(self._abc_only_subgraph(self._load_workflow()), req)
        prompt_id = await self._submit(wf)
        entry = await self._wait_for_history(prompt_id)
        outputs = self._collect_outputs(entry, need_audio=False)
        if "abc" not in outputs:
            raise GenerationError("planning pass finished but no score "
                                  "file was found in /history")
        r = await self._client.get(
            "/view",
            params={"filename": outputs["abc"]["filename"],
                    "subfolder": outputs["abc"]["subfolder"], "type": "output"},
        )
        r.raise_for_status()
        return r.content.decode("utf-8", errors="replace")

    async def _run_full(self, req: GenRequest, workdir: Path) -> GenResult:
        """Single prompt: plan (or take the supplied ABC) and render audio."""
        wf = self._patch_workflow(self._load_workflow(), req)
        prompt_id = await self._submit(wf)
        entry = await self._wait_for_history(prompt_id)
        outputs = self._collect_outputs(entry)
        return await self._download(outputs, workdir, prompt_id)

    async def _submit(self, wf: dict) -> str:
        try:
            resp = await self._client.post(
                "/prompt", json={"prompt": wf, "client_id": self._client_id}
            )
        except httpx.HTTPError as exc:
            raise GenerationError(f"cannot reach ComfyUI at {self.base}: {exc}") from exc
        if resp.status_code != 200:
            detail = ""
            try:
                body = resp.json()
                detail = str(body.get("error", body))[:800]
                for node_errors in (body.get("node_errors") or {}).values():
                    for err in (node_errors or {}).get("errors", []):
                        detail += f"; {err.get('message')}: {err.get('details')}"
            except Exception:
                detail = resp.text[:800]
            raise GenerationError(f"ComfyUI rejected the workflow: {detail}")
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise GenerationError("ComfyUI did not return a prompt_id")
        return prompt_id

    async def _wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.deadline_s
        while time.monotonic() < deadline:
            try:
                r = await self._client.get(f"/history/{quote(prompt_id)}")
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as exc:
                raise GenerationError(f"history poll failed: {exc}") from exc
            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise GenerationError(f"execution error: {self._error_detail(entry)}")
                outputs = entry.get("outputs") or {}
                if outputs:
                    return entry
                # entry without outputs yet: keep polling
            await asyncio.sleep(self.poll_interval)
        raise GenerationError(f"timed out after {self.deadline_s:.0f}s waiting for ComfyUI job")

    @staticmethod
    def _error_detail(entry: dict) -> str:
        for msg in (entry.get("status") or {}).get("messages", []):
            if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                d = msg[1] if len(msg) > 1 else {}
                return (f"node {d.get('node_id')} ({d.get('node_type')}): "
                        f"{d.get('exception_message')}")
        return "unknown error (see ComfyUI server log)"

    @staticmethod
    def _collect_outputs(entry: dict, need_audio: bool = True) -> dict[str, dict[str, str]]:
        """-> {'audio': {filename, subfolder}, 'abc': {filename, subfolder}}"""
        found: dict[str, list[dict[str, str]]] = {"audio": [], "abc": []}
        for node_id, node_out in (entry.get("outputs") or {}).items():
            if not isinstance(node_out, dict):
                continue
            for entries in node_out.values():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict) or e.get("type") != "output":
                        continue
                    fname = str(e.get("filename", ""))
                    sub = str(e.get("subfolder", ""))
                    ext = Path(fname).suffix.lower()
                    if ext in AUDIO_EXTS:
                        found["audio"].append({"filename": fname, "subfolder": sub})
                    elif ext in (".txt", ".abc"):
                        found["abc"].append({"filename": fname, "subfolder": sub})
        out: dict[str, dict[str, str]] = {}
        for kind, lst in found.items():
            if lst:
                out[kind] = sorted(lst, key=lambda x: x["filename"])[0]
                if len(lst) > 1:
                    log.debug("multiple %s outputs found, using %s", kind, out[kind]["filename"])
        if need_audio and "audio" not in out:
            raise GenerationError("job finished but no audio output was found in /history")
        return out

    async def _download(self, outputs: dict, workdir: Path, prompt_id: str) -> GenResult:
        workdir.mkdir(parents=True, exist_ok=True)

        audio = outputs["audio"]
        original = workdir / f"original{Path(audio['filename']).suffix.lower()}"
        await self._download_file(audio, original)

        abc_text = ""
        if "abc" in outputs:
            r = await self._client.get(
                "/view",
                params={"filename": outputs["abc"]["filename"],
                        "subfolder": outputs["abc"]["subfolder"], "type": "output"},
            )
            r.raise_for_status()
            abc_text = r.content.decode("utf-8", errors="replace")
            (workdir / "score.abc").write_text(abc_text, encoding="utf-8")
        else:
            log.warning("job %s produced no ABC text file", prompt_id)

        return GenResult(original_path=original, abc_text=abc_text,
                         info={"comfy_prompt_id": prompt_id})

    async def _download_file(self, f: dict[str, str], dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            async with self._client.stream(
                "GET", "/view",
                params={"filename": f["filename"], "subfolder": f["subfolder"], "type": "output"},
            ) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    async for chunk in resp.aiter_bytes(1 << 16):
                        fh.write(chunk)
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    async def interrupt(self) -> bool:
        try:
            r = await self._client.post("/interrupt")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": "comfyui", "url": self.base}
        try:
            r = await self._client.get("/system_stats")
            r.raise_for_status()
            out["ok"] = True
            out["comfyui"] = r.json()
        except Exception as exc:
            out["ok"] = False
            out["error"] = str(exc)
        out["sidecar"] = await self.sidecar_status()
        return out
