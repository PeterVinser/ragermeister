"""Canonical entity store + blocking — the shared candidate-generation layer for ER.

Holds one record per canonical entity: its name, type, alias set, and a normalized
embedding (name + type + context). Blocking is a cosine search WITHIN A TYPE, tuned for
recall — loose k, low floor. It is the recall ceiling of the whole resolver; the
adjudicator supplies precision afterwards, so we never tighten blocking to save calls.

Entity embeddings live here, NOT in the dependency graph (the graph stores no vectors —
chunk and entity nodes only reference ids). This index is also the shared identity layer:
every baseline that needs entities reads the same canonical ids out of it.

The corpus is small (~100-200 docs), so a dense per-type scan is trivially cheap and we
skip an ANN index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from solution.models.entity import EntityCandidate, EntityType


@dataclass
class EntityRecord:
    canonical_id: str
    canonical_name: str
    type: EntityType
    embedding: np.ndarray  # normalized
    aliases: set[str] = field(default_factory=set)


class EntityIndex:
    def __init__(self) -> None:
        self._by_id: dict[str, EntityRecord] = {}
        self._by_type: dict[EntityType, list[str]] = {}
        # Monotonic per-type counter for deterministic, replayable ids (no randomness).
        self._counters: dict[EntityType, int] = {}

    # ------------------------------------------------------------------ blocking

    def block(
        self,
        embedding: np.ndarray,
        type: EntityType,
        k: int,
        floor: float,
    ) -> list[EntityCandidate]:
        """Top-k canonical entities of the SAME type by cosine, above ``floor``. Recall
        is the goal here — keep k loose and the floor low."""
        ids = self._by_type.get(type, [])
        scored: list[tuple[float, EntityRecord]] = []
        for cid in ids:
            rec = self._by_id[cid]
            score = float(np.dot(embedding, rec.embedding))
            if score >= floor:
                scored.append((score, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            EntityCandidate(
                canonical_id=rec.canonical_id,
                canonical_name=rec.canonical_name,
                type=rec.type,
                aliases=sorted(rec.aliases),
                score=score,
            )
            for score, rec in scored[:k]
        ]

    # ------------------------------------------------------------------ mutation

    def create(
        self,
        canonical_name: str,
        type: EntityType,
        embedding: np.ndarray,
        surface_form: str | None = None,
    ) -> str:
        self._counters[type] = self._counters.get(type, 0) + 1
        canonical_id = f"{type.value}-{self._counters[type]}"
        aliases = {canonical_name}
        if surface_form:
            aliases.add(surface_form)
        self._by_id[canonical_id] = EntityRecord(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            type=type,
            embedding=embedding,
            aliases=aliases,
        )
        self._by_type.setdefault(type, []).append(canonical_id)
        return canonical_id

    def add_alias(self, canonical_id: str, surface_form: str) -> None:
        rec = self._by_id.get(canonical_id)
        if rec is not None and surface_form:
            rec.aliases.add(surface_form)

    def get(self, canonical_id: str) -> EntityRecord | None:
        return self._by_id.get(canonical_id)

    # ------------------------------------------------------------------ metric hook

    def alias_clusters(self) -> dict[str, dict]:
        """One row per canonical node: its name, type, and observed alias set. This is the
        scorable artifact — compared against a gold entity set it yields the
        false-discovery-rate of unresolved entities (over-/under-merge), which caps
        graph-only and metadata-only recall."""
        return {
            cid: {
                "canonical_name": rec.canonical_name,
                "type": rec.type.value,
                "aliases": sorted(rec.aliases),
            }
            for cid, rec in self._by_id.items()
        }
