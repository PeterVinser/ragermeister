"""Append-only event log — the source of truth.

The graph view is a materialized fold over this log; it is rebuildable by replay
(see ``GraphHouseKeeper.rebuild_from_log``). The graph is checkpointed every K
events for fast replay, but this log is authoritative.

Payloads are plain JSON-serialisable dicts so the log stays backend-agnostic. Each
``commit``/``retire``/``resolution`` payload carries everything replay needs to
reconstruct the graph deterministically — including frozen ``similar_to`` neighbour
ids and scores, so replay never needs to touch embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class LogEvent:
    seq: int
    type: str  # "insert" | "update" | "delete" | "resolve"
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"seq": self.seq, "type": self.type, "payload": self.payload},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "LogEvent":
        d = json.loads(line)
        return cls(seq=d["seq"], type=d["type"], payload=d["payload"])


class EventLog:
    """Append-only ``events.jsonl``. ``path=None`` keeps the log purely in memory
    (handy for tests). Seqs are 1-based and monotonic."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._events: list[LogEvent] = []
        if self._path is not None and self._path.exists():
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._events.append(LogEvent.from_json(line))

    @property
    def current_seq(self) -> int:
        """Seq of the last appended event (0 if empty)."""
        return self._events[-1].seq if self._events else 0

    def append(self, event_type: str, payload: dict[str, Any]) -> LogEvent:
        event = LogEvent(seq=self.current_seq + 1, type=event_type, payload=payload)
        self._events.append(event)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
        return event

    def read_all(self) -> list[LogEvent]:
        return list(self._events)

    def __iter__(self) -> Iterator[LogEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)
