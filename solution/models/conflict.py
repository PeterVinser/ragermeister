from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from solution.models.chunk import Chunk


class ConflictLabel(str, Enum):
    CLEAN = "clean"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    SUPERSEDES = "supersedes"
    NEEDS_HUMAN = "needs_human"


@dataclass
class JudgeResult:
    label: ConflictLabel
    implicated_ids: list[str]
    proposed_action: str
    rationale: str


@dataclass
class ConflictReport:
    report_id: str
    new_chunk: Chunk
    judge_result: JudgeResult


class DecisionAction(str, Enum):
    INSERT = "insert"
    UPDATE = "update"   # remove implicated + add new
    DELETE = "delete"   # remove implicated, discard new
    SKIP = "skip"       # discard new, keep implicated


@dataclass
class Decision:
    report_id: str
    action: DecisionAction
    new_chunk: Chunk | None = None
    chunk_ids_to_remove: list[str] = field(default_factory=list)
