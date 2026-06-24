"""Graph-only housekeeper.

Maintains a typed dependency graph as an auxiliary structure *beside* the knowledge base
(it never writes to the base's index or docstore — it only reads them). Candidate
discovery is structural: an arriving chunk is run through the SHARED entity resolver to
attach it to canonical entity / date / topic anchors, and personalised PageRank surfaces
the existing chunks reachable from those anchors. This is NEVER a live KNN against the
corpus — that would be the hybrid baseline.

Entity attachment goes through the shared ``EntityResolver`` (extract -> block ->
adjudicate -> resolve), so the graph links chunks by *resolved canonical identity*, not by
raw surface string. The resolver is shared across baselines; its canonical store lives
outside the graph. Embeddings touch the graph in exactly one place: ``on_commit`` freezes
build-time chunk similarity into ``similar_to`` edges. The graph stores no vectors.

The append-only event log is this housekeeper's private source of truth — the graph view
is a fold over it (see ``rebuild_from_log``). The base is unaware any of this exists.
"""

from __future__ import annotations

import numpy as np

from solution.models.chunk import Chunk
from solution.models.conflict import Decision, DecisionAction
from solution.models.graph import NodeType, Relation, make_node_id
from solution.services.docstore import Docstore
from solution.services.entity_resolver import ChunkResolution, EntityResolver
from solution.services.event_log import EventLog
from solution.services.graph_db import NetworkXGraphStore, ChunkKeys
from solution.services.house_keeper import HouseKeeper
from solution.services.vector_db import VectorDB

_GRAPH_TOP_N = 5
_SIM_FREEZE_K = 5
_CHECKPOINT_EVERY = 50


class GraphHouseKeeper(HouseKeeper):
    def __init__(
        self,
        vector_db: VectorDB,
        docstore: Docstore,
        resolver: EntityResolver,
        graph: NetworkXGraphStore | None = None,
        event_log: EventLog | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        # Read-only handles onto the base's retrieval surface (used to resolve node ->
        # chunk and to freeze build-time similarity). Never mutated here.
        self._vdb = vector_db
        self._docstore = docstore
        # Shared graph-construction infrastructure (NOT baseline-specific).
        self._resolver = resolver
        # Auxiliary structures owned exclusively by this housekeeper.
        self._graph = graph if graph is not None else NetworkXGraphStore()
        self._log = event_log if event_log is not None else EventLog()
        self._checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------ discovery

    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        # Read-only resolve: surfaces matched canonical entities WITHOUT minting any (an
        # arriving chunk that gets flagged/skipped must not create entities).
        resolution = self._resolver.resolve(chunk)
        source_id = chunk.metadata.get("source_id") or chunk.doc_id
        anchors = self._candidate_anchors(resolution, source_id)
        if not anchors:
            return []
        candidate_nodes = self._graph.ppr(anchors, _GRAPH_TOP_N)
        return [
            c for nid in candidate_nodes if (c := self._node_to_chunk(nid)) is not None
        ]

    def _candidate_anchors(self, resolution: ChunkResolution, source_id: str) -> list[str]:
        """keys/resolved-entities -> existing anchor node ids. Dates/topics/source go
        through the symbolic index; entities resolve to canonical node ids directly."""
        keys = ChunkKeys(
            entities=[], dates=resolution.dates, topics=resolution.topics, source=source_id
        )
        anchors = list(self._graph.anchors_for(keys))
        for mr in resolution.entities:
            if mr.canonical_id is None:
                continue
            node = make_node_id(NodeType.ENTITY, mr.canonical_id)
            if self._graph.get_node(node) is not None and node not in anchors:
                anchors.append(node)
        return anchors

    def _node_to_chunk(self, chunk_node_id: str) -> Chunk | None:
        node = self._graph.get_node(chunk_node_id)
        if node is None:
            return None
        emb_id = node.attrs.get("embedding_id")
        if emb_id is None:
            return None
        return self._docstore.get_by_vec_id(int(emb_id))

    # ------------------------------------------------------------------ commit

    def on_commit(self, chunk: Chunk, embedding: np.ndarray) -> None:
        """Mirror a committed chunk into the graph: resolve + attach canonical anchors,
        add the Chunk node and structural edges, then freeze build-time ``similar_to``."""
        source_id = chunk.metadata.get("source_id") or chunk.doc_id
        # Mutating resolve: mint/merge canonical entities (reuses the read-path cache).
        resolution = self._resolver.commit(chunk)

        # Search the corpus and drop self: the base has ALREADY added this chunk to the
        # index by the time on_commit runs, so it would otherwise be its own top hit.
        row = embedding.reshape(1, -1)
        sim_vec_ids, sim_scores = self._vdb.search(row, _SIM_FREEZE_K + 1)
        frozen: list[tuple[str, float]] = []
        for vid, score in zip(sim_vec_ids, sim_scores):
            if vid == chunk.vec_id or score <= 0:
                continue
            neighbor = self._docstore.get_by_vec_id(vid)
            if neighbor is None:
                continue
            frozen.append((make_node_id(NodeType.CHUNK, neighbor.chunk_id), score))
        frozen = frozen[:_SIM_FREEZE_K]

        entities = self._entity_payload(resolution)
        derived_from = [d for d in (chunk.metadata.get("derived_from") or []) if d]
        seq = self._log.append(
            "commit",
            {
                "doc_id": chunk.doc_id,
                "source_id": source_id,
                "chunk": chunk.model_dump(),
                "similar_to": frozen,
                "entities": entities,
                "dates": resolution.dates,
                "topics": resolution.topics,
                "derived_from": derived_from,
            },
        ).seq

        chunk_node = self._add_chunk_with_anchors(
            chunk, resolution, entities, seq
        )
        # Artifact provenance: link this chunk to the source documents it was generated
        # from. Shared construction — graph-only and hybrid get the identical edge.
        for src_doc in derived_from:
            self._graph.add_derived_from(
                chunk_node, make_node_id(NodeType.DOCUMENT, src_doc), seq
            )
        if frozen:
            self._graph.freeze_similar_to(chunk_node, frozen, seq)
        self._maybe_checkpoint()

    def _add_chunk_with_anchors(
        self, chunk: Chunk, resolution: ChunkResolution, entities: list[dict], seq: int
    ) -> str:
        source_id = chunk.metadata.get("source_id") or chunk.doc_id
        # Dates/topics keep the symbolic-anchor path; entities are canonical-id keyed.
        keys = ChunkKeys(
            entities=[], dates=resolution.dates, topics=resolution.topics, source=source_id
        )
        chunk_node = self._graph.add_chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            embedding_id=chunk.vec_id,
            keys=keys,
            seq=seq,
        )
        for ent in entities:
            ent_node = self._graph.ensure_entity(
                ent["canonical_id"], ent["name"], ent["type"], seq, ent["aliases"]
            )
            self._graph.add_mention(chunk_node, ent_node, seq)
        return chunk_node

    def _entity_payload(self, resolution: ChunkResolution) -> list[dict]:
        """Resolved canonical entities for this chunk, deduped, with name+aliases pulled
        from the shared index. Logged so replay reconstructs nodes without re-resolving."""
        out: list[dict] = []
        seen: set[str] = set()
        for mr in resolution.entities:
            cid = mr.canonical_id
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            rec = self._resolver.index.get(cid)
            if rec is None:
                continue
            out.append(
                {
                    "canonical_id": cid,
                    "name": rec.canonical_name,
                    "type": rec.type.value,
                    "aliases": sorted(rec.aliases),
                }
            )
        return out

    # ------------------------------------------------------------------ retire

    def on_retire(self, chunk_ids: list[str]) -> list[Chunk]:
        """Tombstone the retired chunks' nodes (never hard-delete) and revoke their
        incident semantic edges. Return the still-live chunks on the other end of those
        revoked edges: a retirement may have invalidated an assertion that previously held,
        so the KB should re-judge them. This is the graph's diachronic-conflict win."""
        if not chunk_ids:
            return []
        seq = self._log.append("retire", {"chunk_ids": list(chunk_ids)}).seq
        revoked = []
        for chunk_id in chunk_ids:
            revoked.extend(
                self._graph.retire(make_node_id(NodeType.CHUNK, chunk_id), seq)
            )
        self._maybe_checkpoint()
        return self._live_successors(revoked, retired=set(chunk_ids))

    def _live_successors(self, revoked, retired: set[str]) -> list[Chunk]:
        seen: set[str] = set()
        out: list[Chunk] = []
        for edge in revoked:
            for endpoint in (edge.src, edge.dst):
                if endpoint in seen:
                    continue
                seen.add(endpoint)
                node = self._graph.get_node(endpoint)
                if (
                    node is not None
                    and node.status.value == "active"
                    and node.type is NodeType.CHUNK
                    and node.attrs.get("chunk_id") not in retired
                    and (chunk := self._node_to_chunk(endpoint)) is not None
                ):
                    out.append(chunk)
        return out

    # ------------------------------------------------------------------ resolution

    def on_resolution(self, decision: Decision, new_chunk: Chunk | None) -> None:
        """Persist the judge's verdict as provenance: a Decision node plus the semantic
        edge (new chunk -> each retired chunk) the action asserted."""
        if new_chunk is None or not decision.chunk_ids_to_remove:
            return
        # Reports aren't retained here, so we attach the relation implied by the action.
        relation = (
            Relation.SUPERSEDES_CONTENT
            if decision.action == DecisionAction.UPDATE
            else None
        )
        if relation is None:
            return
        new_node = make_node_id(NodeType.CHUNK, new_chunk.chunk_id)
        if self._graph.get_node(new_node) is None:
            return

        seq = self._log.append(
            "resolution",
            {
                "report_id": decision.report_id,
                "action": decision.action.value,
                "new_chunk_id": new_chunk.chunk_id,
                "remove": list(decision.chunk_ids_to_remove),
            },
        ).seq

        self._graph.add_decision_node(
            decision.report_id, seq, attrs={"action": decision.action.value}
        )
        for old_id in decision.chunk_ids_to_remove:
            old_node = make_node_id(NodeType.CHUNK, old_id)
            if self._graph.get_node(old_node) is None:
                continue
            self._graph.assert_semantic_edge(
                src=new_node,
                dst=old_node,
                relation=relation,
                decision_id=decision.report_id,
                snapshot_seq=seq,
                seq=seq,
            )
        self._maybe_checkpoint()

    # ------------------------------------------------------------------ persistence

    def _maybe_checkpoint(self) -> None:
        if self._checkpoint_path and self._log.current_seq % _CHECKPOINT_EVERY == 0:
            self._graph.checkpoint(self._checkpoint_path)

    def rebuild_from_log(self) -> None:
        """Reconstruct the graph view by replaying the authoritative event log. The commit
        events carry resolved canonical entities, frozen ``similar_to`` neighbours, and the
        resolution edges, so the rebuilt graph is faithful without re-embedding, re-judging,
        or re-resolving. Use after loading a checkpoint to catch up the tail of the log."""
        self._graph = NetworkXGraphStore()
        for ev in self._log.read_all():
            if ev.type == "commit":
                rec = ev.payload["chunk"]
                keys = ChunkKeys(
                    entities=[],
                    dates=ev.payload.get("dates", []),
                    topics=ev.payload.get("topics", []),
                    source=ev.payload.get("source_id"),
                )
                node = self._graph.add_chunk(
                    chunk_id=rec["chunk_id"],
                    doc_id=rec["doc_id"],
                    embedding_id=rec.get("vec_id", -1),
                    keys=keys,
                    seq=ev.seq,
                )
                for ent in ev.payload.get("entities", []):
                    ent_node = self._graph.ensure_entity(
                        ent["canonical_id"], ent["name"], ent["type"], ev.seq, ent["aliases"]
                    )
                    self._graph.add_mention(node, ent_node, ev.seq)
                for src_doc in ev.payload.get("derived_from", []):
                    self._graph.add_derived_from(
                        node, make_node_id(NodeType.DOCUMENT, src_doc), ev.seq
                    )
                frozen = [(n, s) for n, s in ev.payload.get("similar_to", [])]
                if frozen:
                    self._graph.freeze_similar_to(node, frozen, ev.seq)
            elif ev.type == "retire":
                for cid in ev.payload["chunk_ids"]:
                    self._graph.retire(make_node_id(NodeType.CHUNK, cid), ev.seq)
            elif ev.type == "resolution":
                if ev.payload["action"] != DecisionAction.UPDATE.value:
                    continue
                new_node = make_node_id(NodeType.CHUNK, ev.payload["new_chunk_id"])
                if self._graph.get_node(new_node) is None:
                    continue
                self._graph.add_decision_node(
                    ev.payload["report_id"], ev.seq, attrs={"action": ev.payload["action"]}
                )
                for old_id in ev.payload.get("remove", []):
                    old_node = make_node_id(NodeType.CHUNK, old_id)
                    if self._graph.get_node(old_node) is None:
                        continue
                    self._graph.assert_semantic_edge(
                        src=new_node,
                        dst=old_node,
                        relation=Relation.SUPERSEDES_CONTENT,
                        decision_id=ev.payload["report_id"],
                        snapshot_seq=ev.seq,
                        seq=ev.seq,
                    )
