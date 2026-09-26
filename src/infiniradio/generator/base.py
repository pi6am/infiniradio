"""Generator interface.

A Generator turns a GenRequest into artifacts inside a per-track directory:
the original audio file plus the ABC score text. Implementations talk to
whatever music backend they wrap (ComfyUI, a local mock, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenRequest:
    style: str
    lyrics: str
    seed: int
    cot_mode: str = "full"
    # When set, re-render this ABC score instead of planning a new one
    # (a "variation": different seed and/or style, same song).
    abc: str | None = None
    parent_track_id: int | None = None


@dataclass
class GenResult:
    original_path: Path
    abc_text: str
    info: dict[str, Any] = field(default_factory=dict)


class GenerationError(Exception):
    pass


class Generator(ABC):
    name: str = "abstract"

    @abstractmethod
    async def generate(self, req: GenRequest, workdir: Path) -> GenResult:
        """Produce original.<ext> + ABC text inside workdir. Raises GenerationError."""

    async def status(self) -> dict[str, Any]:
        """Backend health/info for the admin dashboard."""
        return {"backend": self.name, "ok": True}

    async def cleanup_now(self) -> dict[str, Any] | None:
        """Best-effort remote cleanup (sidecar). Returns response dict or None."""
        return None

    async def interrupt(self) -> bool:
        """Ask the backend to abort the currently running job (best effort)."""
        return False
