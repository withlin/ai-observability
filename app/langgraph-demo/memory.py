import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import SystemMessage

from context import RunContext
from models import MemoryRecord
from telemetry import Telemetry, json_str

STORE_ID = "traveler_memory"


class JsonMemoryStore:
    """Small atomic JSON store for conversation-scoped Q/A records."""

    def __init__(self, path: Path):
        self._path = path

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, store: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self._path)

    def search(self, conversation_id: str, limit: int) -> list[MemoryRecord]:
        store = self._read()
        records = [
            MemoryRecord.from_dict(record)
            for record in store.get(STORE_ID, [])
            if record.get("metadata", {}).get("conversation_id") == conversation_id
        ]
        records.sort(key=lambda record: record.created_at)
        return records[-limit:]

    def add(self, record: MemoryRecord) -> None:
        store = self._read()
        store.setdefault(STORE_ID, []).append(record.as_dict())
        self._write(store)


class MemoryService:
    def __init__(self, store: JsonMemoryStore, telemetry: Telemetry,
                 store_id: str = STORE_ID):
        self._store = store
        self._telemetry = telemetry
        self._store_id = store_id

    def search(self, context: RunContext, query: str, limit: int = 5) -> list[MemoryRecord]:
        with self._telemetry.memory_span("search_memory", self._store_id, context) as span:
            if self._telemetry.capture_content:
                span.set_attribute("gen_ai.memory.query.text", query)
            try:
                records = self._store.search(context.conversation_id, limit)
            except Exception as exc:
                span.set_error(exc)
                span.set_attribute("gen_ai.memory.record.count", 0)
                return []
            span.set_attribute("gen_ai.memory.record.count", len(records))
            if self._telemetry.capture_content:
                span.set_attribute(
                    "gen_ai.memory.records",
                    json_str([record.as_dict() for record in records]),
                )
            return records

    def add(self, context: RunContext, question: str, answer: str) -> str:
        record = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:16]}",
            question=question,
            answer=answer,
            conversation_id=context.conversation_id,
            user_id=context.user_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._telemetry.memory_span(
            "create_memory", self._store_id, context,
            attrs={"gen_ai.memory.record.id": record.id},
        ) as span:
            span.set_attribute("gen_ai.memory.record.count", 1)
            try:
                self._store.add(record)
            except Exception as exc:
                span.set_error(exc)
                return ""
            if self._telemetry.capture_content:
                span.set_attribute("gen_ai.memory.records", json_str([record.as_dict()]))
            return record.id

    @staticmethod
    def context_message(records: list[MemoryRecord]) -> SystemMessage | None:
        if not records:
            return None
        lines = ["Conversation memory from earlier runs:"]
        for record in records:
            lines.append(f"- Q: {record.question}\n  A: {record.answer}")
        return SystemMessage(content="\n".join(lines))
