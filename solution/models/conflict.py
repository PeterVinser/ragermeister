from enum import Enum
from pydantic import BaseModel, Field

from solution.models.chunk import Chunk


class ConflictLabel(str, Enum):
    CLEAN = "clean"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    SUPERSEDES = "supersedes"
    NEEDS_HUMAN = "needs_human"

class JudgeResult(BaseModel):
    label: ConflictLabel
    implicated_ids: list[str]
    proposed_action: str
    rationale: str

class ConflictReport(BaseModel):
    report_id: str
    new_chunk: Chunk
    judge_result: JudgeResult


class DecisionAction(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SKIP = "skip"

class Decision(BaseModel):
    report_id: str
    action: DecisionAction
    new_chunk: Chunk | None = None
    chunk_ids_to_remove: list[str] = Field(default_factory=list)
