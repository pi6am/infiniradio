"""LLM client interface.

An LLMClient turns a list of chat messages into an assistant reply.
Implementations talk to whatever text backend they wrap (an OpenAI-compatible
HTTP endpoint, a scripted mock, ...). The interface is message-list based
rather than string-in/string-out because lyrics workflows carry conversations
across stages: a later stage can continue the thread an earlier stage
started by resubmitting the accumulated history.

Transport-level robustness (retries with backoff on 429/5xx/timeouts) is the
client's job; workflow-stage retries are about *content* (gate failures,
extraction misses). Keeping the two apart is what makes per-stage attempt
counters meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """One turn of a chat conversation. role: 'system' | 'user' | 'assistant'."""

    role: str
    content: str


@dataclass
class Reply:
    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)  # prompt/completion/total tokens
    truncated: bool = False        # finish_reason == "length": the reply was cut
                                   # off mid-stream by the output budget; the
                                   # caller (workflow engine) may continue it


class LLMError(Exception):
    """A chat call failed after exhausting transport retries."""


class LLMClient(ABC):
    name: str = "abstract"

    @abstractmethod
    async def chat(self, messages: list[Message], *, model: str | None = None,
                   temperature: float | None = None,
                   max_tokens: int | None = None) -> Reply:
        """One chat completion. Raises LLMError on failure."""

    async def status(self) -> dict[str, Any]:
        """Backend health/info for the admin dashboard."""
        return {"backend": self.name, "ok": True}

    async def close(self) -> None:
        return None
