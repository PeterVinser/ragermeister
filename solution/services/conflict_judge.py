import json
from solution.services.llm import LLM
from solution.models.chunk import Chunk
from solution.models.conflict import ConflictLabel, JudgeResult
from solution.models.message import Message

_SYSTEM_PROMPT = """\
You are a conflict-detection judge for a knowledge base. Given a new text chunk and its \
nearest neighbors already in the index, classify the relationship.

Respond ONLY with valid JSON:
{
  "label": "clean" | "duplicate" | "contradiction" | "supersedes" | "needs_human",
  "implicated_ids": ["<chunk_id>", ...],
  "proposed_action": "insert" | "replace_old" | "skip" | "flag_for_human",
  "rationale": "<one sentence>"
}"""


class ConflictJudge:
    def __init__(self) -> None:
        self._llm = LLM()

    def judge(self, new_chunk: Chunk, neighbors: list[Chunk]) -> JudgeResult:
        if not neighbors:
            return JudgeResult(
                label=ConflictLabel.CLEAN,
                implicated_ids=[],
                proposed_action="insert",
                rationale="No neighbors in index.",
            )
        user_content = self._build_user_content(new_chunk, neighbors)
        raw = self._llm.get_response(_SYSTEM_PROMPT, [Message(role="user", content=user_content)])
        return self._parse(raw, neighbors)

    def _build_user_content(self, new_chunk: Chunk, neighbors: list[Chunk]) -> str:
        neighbor_lines = "\n".join(
            f"  [chunk_id={n.chunk_id}] {n.text[:300]}" for n in neighbors
        )
        return (
            f"NEW CHUNK (chunk_id={new_chunk.chunk_id}):\n{new_chunk.text}\n\n"
            f"NEIGHBORS:\n{neighbor_lines}"
        )

    def _parse(self, raw: str, neighbors: list[Chunk]) -> JudgeResult:
        try:
            data = json.loads(raw)
            return JudgeResult(
                label=ConflictLabel(data["label"]),
                implicated_ids=list(data.get("implicated_ids", [])),
                proposed_action=str(data.get("proposed_action", "")),
                rationale=str(data.get("rationale", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return JudgeResult(
                label=ConflictLabel.NEEDS_HUMAN,
                implicated_ids=[n.chunk_id for n in neighbors],
                proposed_action="flag_for_human",
                rationale=f"Could not parse judge response: {raw[:150]}",
            )
