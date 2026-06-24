"""Hybrid housekeeper — dual-seeded Personalized PageRank.

Builds the SAME dependency graph as graph-only (it subclasses ``GraphHouseKeeper``, so
construction — entity attachment, frozen ``similar_to``, derived_from, retire, resolution
edges — is identical and the comparison stays honest). The ONLY thing it changes is
discovery: how the teleport vector is seeded, and that the walk runs over the directed
walk graph.

It is the ONLY baseline permitted to use the incoming chunk's live embedding — and even
then only at SEEDING. The walk never embeds, never re-seeds, never KNNs mid-flight; the
only semantic signal inside the walk is the frozen ``similar_to`` topology.

Seeding (teleport vector), once per event, mass SUMMED onto nodes (addition, not score
fusion — a node hit by two sources simply accumulates more mass):
  * Vector seeds  — KNN the live embedding against the FAISS chunk index, land on chunk
                    nodes, weight = sim^2 (sharpen toward confident matches).
  * Entity seeds  — resolve the chunk's entities, land on entity nodes, weight w_entity,
                    down-weighted by node degree so a hub ("Northmoor") can't dominate.
  * Update-identity seeds — on an update, the prior version's (now tombstoned) chunk
                    nodes, weight w_old. Structural certainty: don't rediscover what we
                    already know the update affects.

Then one personalized-PageRank diffusion over the directed walk graph, restart ~0.2.
"""

from __future__ import annotations

import numpy as np

from solution.models.chunk import Chunk
from solution.models.graph import NodeType, make_node_id
from solution.services.docstore import Docstore
from solution.services.entity_resolver import EntityResolver
from solution.services.event_log import EventLog
from solution.services.graph_db import NetworkXGraphStore
from solution.services.graph_house_keeper import GraphHouseKeeper
from solution.services.vector_db import VectorDB

_TOP_N = 5
_K_VEC = 5
_W_ENTITY = 1.0
_W_OLD = 3.0  # update-identity: high — structural certainty
_RESTART = 0.2  # teleport probability (locality / precision dial)


class HybridHouseKeeper(GraphHouseKeeper):
    def __init__(
        self,
        vector_db: VectorDB,
        docstore: Docstore,
        resolver: EntityResolver,
        graph: NetworkXGraphStore | None = None,
        event_log: EventLog | None = None,
        checkpoint_path: str | None = None,
        top_n: int = _TOP_N,
        k_vec: int = _K_VEC,
        w_entity: float = _W_ENTITY,
        w_old: float = _W_OLD,
        restart: float = _RESTART,
    ) -> None:
        super().__init__(
            vector_db=vector_db,
            docstore=docstore,
            resolver=resolver,
            graph=graph,
            event_log=event_log,
            checkpoint_path=checkpoint_path,
        )
        self._top_n = top_n
        self._k_vec = k_vec
        self._w_entity = w_entity
        self._w_old = w_old
        self._restart = restart

    # Only discovery is overridden; commit / retire / resolution are inherited unchanged.
    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        teleport = self._seed(chunk, embedding)
        if not teleport:
            return []
        result_nodes = self._graph.ppr_personalized(
            teleport, self._top_n, restart=self._restart
        )
        return [
            c for nid in result_nodes if (c := self._node_to_chunk(nid)) is not None
        ]

    def _seed(self, chunk: Chunk, embedding: np.ndarray) -> dict[str, float]:
        """Mix the three seed populations into one teleport vector by ADDITION."""
        teleport: dict[str, float] = {}

        def add(node_id: str, mass: float) -> None:
            if mass > 0 and self._graph.get_node(node_id) is not None:
                teleport[node_id] = teleport.get(node_id, 0.0) + mass

        # Vector seeds: live KNN against the FAISS chunk index, weight = sim^2.
        if self._k_vec > 0:
            vec_ids, scores = self._vdb.search(embedding.reshape(1, -1), self._k_vec)
            for vid, sim in zip(vec_ids, scores):
                neighbor = self._docstore.get_by_vec_id(vid)
                if neighbor is None or neighbor.chunk_id == chunk.chunk_id:
                    continue
                add(make_node_id(NodeType.CHUNK, neighbor.chunk_id), max(float(sim), 0.0) ** 2)

        # Entity seeds: resolved canonical entities, hub-down-weighted by node degree.
        resolution = self._resolver.resolve(chunk)
        for mr in resolution.entities:
            if mr.canonical_id is None:
                continue
            node = make_node_id(NodeType.ENTITY, mr.canonical_id)
            add(node, self._w_entity / (1.0 + self._graph.degree(node)))

        # Update-identity seeds: the prior version's chunk nodes (tombstoned by the
        # update's retire, still present in the graph), high fixed weight.
        for node in self._graph.chunk_nodes_for_doc(chunk.doc_id):
            add(node, self._w_old)

        return teleport
