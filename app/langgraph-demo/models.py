from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "TokenUsage":
        input_details = data.get("input_token_details") or {}
        output_details = data.get("output_token_details") or {}
        return cls(
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            cache_read_tokens=int(input_details.get("cache_read", 0) or 0),
            cache_write_tokens=int(input_details.get("cache_creation", 0) or 0),
            reasoning_tokens=int(output_details.get("reasoning", 0) or 0),
        )

    def merge(self, other: "TokenUsage") -> "TokenUsage":
        return replace(
            self,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class InferenceResult:
    message: AIMessage
    usage: TokenUsage
    ttft: float | None = None
    time_per_chunk: float | None = None


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    source_uri: str


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    snippet: str


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    question: str
    answer: str
    conversation_id: str
    user_id: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": {"question": self.question, "answer": self.answer},
            "metadata": {
                "conversation_id": self.conversation_id,
                "user_id": self.user_id,
                "created_at": self.created_at,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        content = data.get("content") or {}
        metadata = data.get("metadata") or {}
        return cls(
            id=str(data.get("id", "")),
            question=str(content.get("question", "")),
            answer=str(content.get("answer", "")),
            conversation_id=str(metadata.get("conversation_id", "")),
            user_id=str(metadata.get("user_id", "")),
            created_at=str(metadata.get("created_at", "")),
        )
