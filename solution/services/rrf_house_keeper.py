"""RRF housekeeper — Reciprocal Rank Fusion of vector and graph discovery (ablation).

The "voting, not guidance" contrast to the hybrid. Where the hybrid lets the two signals
*guide one walk* (seeds mixed into a single PPR teleport vector, so a chunk hit by both
accumulates mass and the walk explores from their union), RRF runs vector-only and
graph-only discovery INDEPENDENTLY and fuses their ranked outputs by

    score(doc) = sum_lists 1 / (k + rank)        # k = 60

A doc only scores if a list already surfaced it — fusion can reorder, never reach
something neither parent found. Showing cascade/guidance beats fusion is itself a result.

Composes the two real housekeepers, so their lifecycle hooks (and thus their auxiliary
structures — the graph for graph-only, nothing for vector-only) stay maintained.
"""

from __future__ import annotations

import numpy as np

from solution.models.chunk import Chunk
from solution.models.conflict import Decision
from solution.services.graph_house_keeper import GraphHouseKeeper
from solution.services.house_keeper import HouseKeeper
from solution.services.vector_house_keeper import VectorHouseKeeper

_RRF_K = 60
_TOP_N = 5


class RRFHouseKeeper(HouseKeeper):
    def __init__(
        self,
        vector_hk: VectorHouseKeeper,
        graph_hk: GraphHouseKeeper,
        k: int = _RRF_K,
        top_n: int = _TOP_N,
    ) -> None:
        self._vector = vector_hk
        self._graph = graph_hk
        self._k = k
        self._top_n = top_n

    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        ranked_lists = [
            self._vector.find_candidates(chunk, embedding),
            self._graph.find_candidates(chunk, embedding),
        ]
        scores: dict[str, float] = {}
        objs: dict[str, Chunk] = {}
        for candidates in ranked_lists:
            for rank, cand in enumerate(candidates):
                scores[cand.chunk_id] = scores.get(cand.chunk_id, 0.0) + 1.0 / (
                    self._k + rank + 1
                )
                objs[cand.chunk_id] = cand
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [objs[cid] for cid in ranked[: self._top_n]]

    # Fan lifecycle out to both sub-housekeepers so each keeps its structures in step.
    def on_commit(self, chunk: Chunk, embedding: np.ndarray) -> None:
        self._vector.on_commit(chunk, embedding)
        self._graph.on_commit(chunk, embedding)

    def on_retire(self, chunk_ids: list[str]) -> list[Chunk]:
        vector_succ = self._vector.on_retire(chunk_ids)
        graph_succ = self._graph.on_retire(chunk_ids)
        # Only graph-only surfaces successors; vector-only returns [].
        return graph_succ or vector_succ

    def on_resolution(self, decision: Decision, new_chunk: Chunk | None) -> None:
        self._vector.on_resolution(decision, new_chunk)
        self._graph.on_resolution(decision, new_chunk)
