"""Vector-only housekeeper (v1 baseline).

Candidate discovery is a live KNN of the arriving chunk against the base's vector index
— nothing more. It holds NO auxiliary state, so every lifecycle hook is the inherited
no-op. In particular ``on_retire`` returns ``[]``: a vector-only monitor has no way to
find the stale *old* chunks left behind by a supersession. That is a banked limitation
(reclaimed by graph traversal later), and expressing it as "literally cannot implement
the hook" keeps the baseline honest.
"""

from __future__ import annotations

import numpy as np

from solution.models.chunk import Chunk
from solution.services.docstore import Docstore
from solution.services.house_keeper import HouseKeeper
from solution.services.vector_db import VectorDB

_NEIGHBOR_K = 5


class VectorHouseKeeper(HouseKeeper):
    def __init__(
        self,
        vector_db: VectorDB,
        docstore: Docstore,
        k: int = _NEIGHBOR_K,
    ) -> None:
        # Read-only use of the base's retrieval surface — never mutated here.
        self._vdb = vector_db
        self._docstore = docstore
        self._k = k

    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        vec_ids, _ = self._vdb.search(embedding.reshape(1, -1), self._k)
        return [
            c
            for vid in vec_ids
            if (c := self._docstore.get_by_vec_id(vid)) is not None
            and c.chunk_id != chunk.chunk_id  # exclude self if already committed
        ]
