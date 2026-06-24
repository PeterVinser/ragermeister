"""Entity extraction — one schema-constrained LLM pass over a chunk's text ONLY.

Type-aware, entity-only, with intra-chunk coreference resolved in the same pass (pronouns
and definite descriptions collapsed to their antecedent before the mention is emitted).
This never looks at the knowledge base — it reads the chunk text and nothing else.

Behind an interface so it is mockable (invariant: LLM behind interfaces; tests run with no
real model calls). It is shared across all baselines: every baseline that needs entities
reads the SAME extractor output, exactly as the contradiction judge is shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from solution.models.chunk import Chunk
from solution.models.entity import EntityMention, ExtractionResult
from solution.models.message import Message
from solution.services.llm import LLM

_SYSTEM_PROMPT = """\
You extract named entities from a single text chunk for a knowledge-base entity index.

Return ONLY entities, never relations or facts. For each entity emit:
  - surface_form: the entity as it should be indexed, with intra-chunk coreference
    already resolved. Replace pronouns and definite descriptions ("he", "she", "the
    office", "the dean") with the proper name they refer to in THIS chunk. If a proper
    name is never given, use the most specific noun phrase available.
  - type: one of "person", "org", "date", "topic", "other".
      person = a named individual.
      org    = an organisation, office, department, or institutional role-holder.
      date   = a calendar date or deadline (normalise to YYYY-MM-DD when the year is
               known; otherwise keep month-day).
      topic  = the subject the chunk is about (e.g. "application deadline").
      other  = a structured entity that fits none of the above.
  - short_context: 3-8 words from the chunk that disambiguate this mention. Never leave
    it empty for person/org; it is embedded together with the name.

Think step by step about who/what each pronoun refers to before emitting, but output ONLY
the final JSON object matching the schema. Do not invent entities not present in the text.
Do not extract relationships between entities.

Example:
  Text: "Dr. Elena Vance leads admissions. She moved the deadline to March 1."
  -> mentions:
     {surface_form: "Elena Vance", type: "person", short_context: "leads admissions"}
     {surface_form: "admissions office", type: "org", short_context: "Elena Vance leads"}
     {surface_form: "application deadline", type: "topic", short_context: "moved to March 1"}
     {surface_form: "March 1", type: "date", short_context: "deadline moved to"}
"""


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, chunk: Chunk) -> list[EntityMention]:
        """Pull typed, coref-resolved entity mentions out of one chunk's text."""


class LLMEntityExtractor(EntityExtractor):
    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm if llm is not None else LLM()

    def extract(self, chunk: Chunk) -> list[EntityMention]:
        result = self._llm.get_structured_response(
            _SYSTEM_PROMPT,
            [Message(role="user", content=chunk.text)],
            model_class=ExtractionResult,
        )
        return list(result.mentions) if result is not None else []