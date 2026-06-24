"""Entity-resolution adjudicator — the precision stage, fired ONLY in the gray band.

Given a mention and the blocking candidates, decide merge-to-existing vs create-new. This
is a SEPARATE component from the contradiction judge: different question (is this the same
real-world entity? vs does this chunk contradict that one?), different prompt, different
interface, different cache. Never fuse the two.

Invariant: when uncertain, return CREATE_NEW. A recoverable under-merge beats a corrupting
over-merge — once two distinct entities are fused, every chunk on both is mislinked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from solution.models.entity import (
    EntityCandidatesDecision,
    EntityCandidatesVerdict,
    EntityCandidate,
    EntityMention,
)
from solution.models.message import Message
from solution.services.llm import LLM

_SYSTEM_PROMPT = """\
You are an entity-resolution adjudicator for a knowledge-base entity index.

You are given ONE entity mention and a short list of existing canonical entities that a
recall-tuned blocking step judged plausibly the same. Decide whether the mention refers to
the SAME real-world entity as one of the candidates, or to a NEW entity.

Return ONLY a JSON object matching this structure:
{
  "decision": "merge" | "create_new",
  "candidate_id": "<id of the candidate to merge into, or null>",
  "confidence": <0.0-1.0>,
  "rationale": "<one concise sentence>"
}

Rules:
- "merge": the mention denotes the same real-world entity as exactly one candidate. Set
  "candidate_id" to that candidate's id.
- "create_new": the mention is a different entity (or you cannot tell). Set
  "candidate_id" to null.
- Same name does NOT imply same entity (two different people can share a name); different
  surface forms CAN be the same entity (alias, abbreviation, role vs name). Weigh the
  context, not just the string.
- If you are not confident it is the same entity, choose "create_new". Under-merging is
  recoverable; over-merging corrupts the index.
"""

def _create_new(rationale: str) -> EntityCandidatesVerdict:
    return EntityCandidatesVerdict(
        decision=EntityCandidatesDecision.CREATE_NEW,
        candidate_id=None,
        confidence=0.0,
        rationale=rationale,
    )


class EntityCandidatesJudge:
    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm if llm is not None else LLM()

    def judge(
        self, mention: EntityMention, candidates: list[EntityCandidate]
    ) -> EntityCandidatesVerdict:
        if not candidates:
            return _create_new("no candidates")
        content = self._build_content(mention, candidates)
        verdict = self._llm.get_structured_response(
            _SYSTEM_PROMPT,
            [Message(role="user", content=content)],
            model_class=EntityCandidatesVerdict,
        )
        if verdict is None:
            return _create_new("adjudicator returned nothing -> create_new")
        # Safety net for both invariants: a MERGE must name a real candidate; anything
        # else collapses to create_new.
        valid_ids = {c.canonical_id for c in candidates}
        if verdict.decision == EntityCandidatesDecision.MERGE and (
            verdict.candidate_id not in valid_ids
        ):
            return _create_new("merge named an unknown candidate -> create_new")
        return verdict

    def _build_content(
        self, mention: EntityMention, candidates: list[EntityCandidate]
    ) -> str:
        cand_lines = "\n".join(
            f"  [id={c.canonical_id}] {c.canonical_name} "
            f"(type={c.type.value}, aliases={c.aliases}, score={c.score:.3f})"
            for c in candidates
        )
        return (
            f"MENTION: {mention.surface_form!r} (type={mention.type.value})\n"
            f"CONTEXT: {mention.short_context}\n\n"
            f"CANDIDATES:\n{cand_lines}"
        )