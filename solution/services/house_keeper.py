"""HouseKeeper — the swappable monitor that sits on top of the KnowledgeBase.

The KnowledgeBase always embeds and stores chunks; it is the retrieval ground truth
and never reasons about consistency itself. A HouseKeeper watches arrivals and surfaces
*candidate* existing chunks that an arrival might conflict with — the KB then feeds those
candidates to the SHARED conflict judge. This is the ONLY thing that differs between
baselines (vector-only / graph-only / hybrid): each surfaces candidates a different way,
so any benchmark difference is attributable to candidate discovery, not to a different
judge (invariant #3).

A HouseKeeper may *read* the base's retrieval surface (the vector index, the docstore)
and build its own auxiliary structures — a graph, structural chunks, indexes — but it
MUST NOT mutate the base. It only monitors; it never owns the corpus.

The KB calls the lifecycle hooks below so a stateful HouseKeeper can keep its auxiliary
structures in step with what the base committed/removed. They default to no-ops, so a
stateless monitor (vector-only) implements only ``find_candidates``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from solution.models.chunk import Chunk
from solution.models.conflict import Decision


class HouseKeeper(ABC):
    @abstractmethod
    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        """Surface existing chunks the arriving ``chunk`` might be inconsistent with.

        This is the baseline-distinguishing step. The SHARED judge (owned by the KB)
        classifies whatever is returned; returning a too-wide set risks false flags, a
        too-narrow set risks misses — that trade-off IS the thing the benchmark measures.
        ``embedding`` is the arrival's (normalized) embedding, already computed by the KB.
        """

    # --------------------------------------------------------------- lifecycle hooks

    def on_commit(self, chunk: Chunk, embedding: np.ndarray) -> None:
        """The base committed ``chunk`` (it now has a ``vec_id``). Update auxiliary
        structures. No-op by default."""

    def on_retire(self, chunk_ids: list[str]) -> list[Chunk]:
        """The base is removing these chunks. Update auxiliary structures and return any
        still-live chunks whose consistency the removal may have changed, for the KB to
        re-judge (diachronic re-validation).

        Returns ``[]`` by default — a stateless monitor cannot find stale successors of a
        removed chunk. That blindness is a *banked* v1 limitation, not an oversight: it is
        precisely what the graph layer reclaims later via traversal."""
        return []

    def on_resolution(self, decision: Decision, new_chunk: Chunk | None) -> None:
        """A resolution committed (the new chunk, if any, is already committed via
        ``on_commit``). Record provenance in auxiliary structures. No-op by default."""
