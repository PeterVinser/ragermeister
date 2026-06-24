"""Entity-resolution data models.

The entity layer is a *discovery scaffold*: canonical entity nodes let an arriving chunk
attach to the same identity an earlier chunk attached to, so structural traversal can
reach it. It carries NO contradictions and NO entity-to-entity / relation edges —
extraction is entity-only. The contradiction judge is a separate component entirely.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PERSON = "person"
    ORG = "org"  # Org / Office
    DATE = "date"
    TOPIC = "topic"
    OTHER = "other"  # structured-but-unhandled; ignored by resolution for now


# Only these go through the full embed -> block -> adjudicate pipeline. Dates take an
# exact-ISO shortcut; topics snap to a controlled vocabulary; OTHER is dropped.
RESOLVABLE_TYPES: frozenset[EntityType] = frozenset({EntityType.PERSON, EntityType.ORG})


class EntityMention(BaseModel):
    """One entity mention pulled from a single chunk. ``surface_form`` is the canonical
    in-chunk mention AFTER intra-chunk coreference (pronouns / "the office" collapsed to
    their antecedent). ``short_context`` is a few words around it — it disambiguates the
    mention when embedded, and must never be the bare surface form alone."""

    surface_form: str
    type: EntityType
    short_context: str = ""


class ExtractionResult(BaseModel):
    """Schema the LLM extractor is constrained to. Entity-only — no relations."""

    mentions: list[EntityMention] = Field(default_factory=list)


class EntityCandidate(BaseModel):
    """A canonical entity surfaced by blocking, with its cosine score to the mention."""

    canonical_id: str
    canonical_name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    score: float = 0.0


class EntityCandidatesDecision(str, Enum):
    MERGE = "merge"
    CREATE_NEW = "create_new"


class EntityCandidatesVerdict(BaseModel):
    """ER adjudicator output. ``candidate_id`` is set iff ``decision == MERGE``. When the
    adjudicator is uncertain it must return CREATE_NEW — a recoverable under-merge beats a
    corrupting over-merge."""

    decision: EntityCandidatesDecision
    candidate_id: str | None = None
    confidence: float = 0.0
    rationale: str = ""
