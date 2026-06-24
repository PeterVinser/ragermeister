import json
from solution.services.llm import LLM
from solution.models.chunk import Chunk
from solution.models.conflict import ConflictLabel, JudgeResult
from solution.models.message import Message

_SYSTEM_PROMPT = """\
You are a conflict-detection judge for a knowledge base.

You will receive:
1. A new text chunk.
2. A list of nearest-neighbor chunks already stored in the index.

Your task is to classify the relationship between the new chunk and the existing chunks.

Return ONLY a valid JSON object matching this exact structure:

{
  "label": "clean" | "duplicate" | "contradiction" | "supersedes" | "needs_human",
  "implicated_ids": ["<chunk_id>", "..."],
  "proposed_action": "insert" | "replace_old" | "skip" | "flag_for_human",
  "rationale": "<one concise sentence explaining the decision>"
}

Field rules:
- "label" must be exactly one of:
  - "clean": the new chunk does not materially conflict with or duplicate existing chunks.
  - "duplicate": the new chunk expresses the same information as one or more existing chunks.
  - "contradiction": the new chunk conflicts with one or more existing chunks and neither is clearly newer or authoritative.
  - "supersedes": the new chunk appears to replace or update one or more existing chunks.
  - "needs_human": the relationship is ambiguous, risky, or cannot be confidently classified.

- "implicated_ids" must contain the IDs of existing chunks involved in the decision.
  - Use an empty list [] when label is "clean".
  - Include one or more chunk IDs for "duplicate", "contradiction", "supersedes", or "needs_human" when relevant.

- "proposed_action" must be exactly one of:
  - "insert": insert the new chunk.
  - "replace_old": insert the new chunk and remove or deactivate implicated older chunks.
  - "skip": do not insert the new chunk.
  - "flag_for_human": do not decide automatically; send for review.

Recommended mapping:
- clean -> insert
- duplicate -> skip
- contradiction -> flag_for_human
- supersedes -> replace_old
- needs_human -> flag_for_human

- "rationale" must be a single concise sentence.
- Do not include markdown, comments, explanations, or any text outside the JSON object.
"""


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
        judge_result = self._llm.get_structured_response(_SYSTEM_PROMPT, [Message(role="user", content=user_content)], model_class=JudgeResult)
        
        return judge_result

    def _build_user_content(self, new_chunk: Chunk, neighbors: list[Chunk]) -> str:
        neighbor_lines = "\n".join(
            f"  [chunk_id={n.chunk_id}] {n.text[:300]}" for n in neighbors
        )
        return (
            f"NEW CHUNK (chunk_id={new_chunk.chunk_id}):\n{new_chunk.text}\n\n"
            f"NEIGHBORS:\n{neighbor_lines}"
        )