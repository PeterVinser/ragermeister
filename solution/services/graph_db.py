"""Graph store for the graph-only housekeeper.

Three-layer store:
  1. Event log (``event_log.py``) — append-only source of truth, *outside* this module.
  2. Graph view — a NetworkX ``MultiDiGraph`` (multigraph so one node pair can hold both
     ``similar_to`` and ``contradicts``; the edge key IS the relation type).
  3. Symbolic indexes — plain dicts (entity/date/topic -> node_id, source -> doc_id).
     Their sole purpose is to let an arriving chunk attach by exact-key lookup, since it
     has no edges yet.

Belief vs structure: the graph grows monotonically in *structure* and prunes only in
*belief*. Nodes are tombstoned (never hard-deleted); structural edges are never revoked;
semantic edges are revoked. PPR walks *through* tombstoned nodes (supersession /
derived_from chains still propagate to live successors) but tombstoned nodes are excluded
from the candidate set handed to the judge.

A ``GraphStore`` interface sits in front so an embedded store (e.g. kuzu) could swap in
later without touching the housekeeper. The graph stores NO embeddings — Chunk nodes hold
an ``embedding_id`` referencing the existing FAISS ``IndexIDMap2``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from solution.models.graph import (
    EdgeStatus,
    EdgeTier,
    GraphEdge,
    GraphNode,
    NODE_PREFIX,
    NodeStatus,
    NodeType,
    Relation,
    make_node_id,
    tier_of,
)
from solution.services.entity_resolver import (
    normalize_entity,
    normalize_topic,
)

# Personalised-PageRank edge-weight table. derived_from / supersedes propagate strongly;
# mentions / on_date are weak (a date is a hub, a mention is cheap). similar_to is scaled
# by its frozen build-time confidence (weight * confidence).
RELATION_WEIGHTS: dict[Relation, float] = {
    Relation.SUPERSEDES: 3.0,
    Relation.SUPERSEDES_CONTENT: 3.0,
    Relation.VERSION_OF: 3.0,
    Relation.DERIVED_FROM: 2.5,
    Relation.CONTRADICTS: 2.0,
    Relation.DUPLICATES: 2.0,
    Relation.CONTAINS: 1.5,
    Relation.SIMILAR_TO: 1.0,  # * confidence
    Relation.HAS_TOPIC: 0.7,
    Relation.MENTIONS: 0.5,
    Relation.ON_DATE: 0.4,
}


@dataclass
class ChunkKeys:
    entities: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)  # ISO-8601 (YYYY-MM-DD)
    topics: list[str] = field(default_factory=list)
    source: str | None = None  # source_id, from event metadata

class GraphStore(ABC):
    """DB-agnostic interface. Keep envelopes free of NetworkX so kuzu could swap in."""

    @abstractmethod
    def anchors_for(self, keys: ChunkKeys) -> list[str]:
        """Symbolic-index lookup: arriving chunk's keys -> existing anchor node ids."""

    @abstractmethod
    def ppr(self, seeds: list[str], top_n: int, exclude: set[str] | None = None) -> list[str]:
        """Personalised PageRank seeded at ``seeds``; returns top-N active chunk node ids."""

    @abstractmethod
    def retire(self, node_id: str, seq: int) -> list[GraphEdge]:
        """Tombstone a node; revoke incident semantic edges; return them for re-judge."""

    @abstractmethod
    def revoke(self, src: str, dst: str, relation: Relation, seq: int) -> None:
        """Mark a single semantic edge revoked (belief pruned, structure kept)."""

    @abstractmethod
    def active_view(self) -> nx.MultiDiGraph:
        """Read-only view of active nodes + active edges."""

    @abstractmethod
    def state_at(self, seq: int) -> nx.MultiDiGraph:
        """Active-status view as of event ``seq`` (temporal reconstruction)."""


class NetworkXGraphStore(GraphStore):
    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        # Symbolic indexes (layer 3). Keys are normalised for exact-match lookup.
        self._entity_index: dict[str, str] = {}  # normalized name -> ent node id
        self._date_index: dict[str, str] = {}  # iso date -> date node id
        self._topic_index: dict[str, str] = {}  # topic slug -> topic node id
        self._source_index: dict[str, str] = {}  # source_id -> doc node id

    # ------------------------------------------------------------------ nodes

    def _put_node(self, node: GraphNode) -> None:
        self._g.add_node(node.node_id, env=node)

    def get_node(self, node_id: str) -> GraphNode | None:
        data = self._g.nodes.get(node_id)
        return data["env"] if data else None

    def _get_or_create_anchor(
        self, node_type: NodeType, key: str, seq: int, attrs: dict | None = None
    ) -> str:
        node_id = make_node_id(node_type, key)
        if node_id not in self._g:
            self._put_node(
                GraphNode(
                    node_id=node_id,
                    type=node_type,
                    valid_from=seq,
                    created_by_event=seq,
                    attrs=attrs or {},
                )
            )
        return node_id

    def ensure_document(self, doc_id: str, seq: int, source_id: str | None = None) -> str:
        node_id = make_node_id(NodeType.DOCUMENT, doc_id)
        if node_id not in self._g:
            self._put_node(
                GraphNode(
                    node_id=node_id,
                    type=NodeType.DOCUMENT,
                    valid_from=seq,
                    created_by_event=seq,
                    attrs={"doc_id": doc_id, "source_id": source_id},
                )
            )
        if source_id is not None:
            self._source_index[source_id] = node_id
        return node_id

    def add_decision_node(self, decision_id: str, seq: int, attrs: dict | None = None) -> str:
        node_id = make_node_id(NodeType.DECISION, decision_id)
        if node_id not in self._g:
            self._put_node(
                GraphNode(
                    node_id=node_id,
                    type=NodeType.DECISION,
                    valid_from=seq,
                    created_by_event=seq,
                    attrs=attrs or {},
                )
            )
        return node_id

    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        embedding_id: int,
        keys: ChunkKeys,
        seq: int,
    ) -> str:
        """Add an active Chunk node plus its deterministic structural edges:
        ``contains`` (doc -> chunk), ``mentions`` (chunk -> entity), ``on_date``,
        ``has_topic``. Anchor nodes are created on demand and the symbolic indexes are
        updated so future arrivals can attach to them."""
        chunk_node = make_node_id(NodeType.CHUNK, chunk_id)
        self._put_node(
            GraphNode(
                node_id=chunk_node,
                type=NodeType.CHUNK,
                valid_from=seq,
                created_by_event=seq,
                attrs={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "embedding_id": embedding_id,  # FAISS id; graph stores NO vectors
                },
            )
        )

        doc_node = self.ensure_document(doc_id, seq, source_id=keys.source)
        self._add_structural_edge(doc_node, chunk_node, Relation.CONTAINS, seq)

        for name in keys.entities:
            norm = normalize_entity(name)
            if not norm:
                continue
            ent_node = self._get_or_create_anchor(
                NodeType.ENTITY, norm, seq, attrs={"name": name}
            )
            self._entity_index[norm] = ent_node
            self._add_structural_edge(chunk_node, ent_node, Relation.MENTIONS, seq)

        for iso in keys.dates:
            date_node = self._get_or_create_anchor(NodeType.DATE, iso, seq)
            self._date_index[iso] = date_node
            self._add_structural_edge(chunk_node, date_node, Relation.ON_DATE, seq)

        for topic in keys.topics:
            slug = normalize_topic(topic)
            if not slug:
                continue
            topic_node = self._get_or_create_anchor(
                NodeType.TOPIC, slug, seq, attrs={"label": topic}
            )
            self._topic_index[slug] = topic_node
            self._add_structural_edge(chunk_node, topic_node, Relation.HAS_TOPIC, seq)

        return chunk_node

    def ensure_entity(
        self,
        canonical_id: str,
        canonical_name: str,
        entity_type: str,
        seq: int,
        aliases: list[str] | None = None,
    ) -> str:
        """Ensure a canonical Entity node keyed by the resolver's ``canonical_id`` (NOT a
        raw surface form). Idempotent: re-mentioning an entity refreshes its alias set
        rather than minting a duplicate. Entities are a discovery scaffold — no embeddings
        and no entity-to-entity edges live here."""
        node_id = make_node_id(NodeType.ENTITY, canonical_id)
        node = self.get_node(node_id)
        if node is None:
            self._put_node(
                GraphNode(
                    node_id=node_id,
                    type=NodeType.ENTITY,
                    valid_from=seq,
                    created_by_event=seq,
                    attrs={
                        "canonical_id": canonical_id,
                        "name": canonical_name,
                        "entity_type": entity_type,
                        "aliases": sorted(set(aliases or [])),
                    },
                )
            )
        elif aliases:
            merged = set(node.attrs.get("aliases", [])) | set(aliases)
            node.attrs["aliases"] = sorted(merged)
        # Canonical-id keyed so a resolved arrival can look the node up directly.
        self._entity_index[canonical_id] = node_id
        return node_id

    def add_mention(self, chunk_node: str, entity_node: str, seq: int) -> None:
        """``mentions`` edge: chunk -> canonical entity (structural, never revoked)."""
        self._add_structural_edge(chunk_node, entity_node, Relation.MENTIONS, seq)

    def add_derived_from(self, artifact_node: str, source_node: str, seq: int) -> None:
        """``derived_from`` edge: artifact chunk -> the source doc/chunk it was generated
        from (structural). Stored forward (artifact -> source); the hybrid walk graph makes
        it reverse-traversable (source -> artifact) so a change to a source can reach its
        stale dependents. Skips silently if either endpoint is absent."""
        if artifact_node not in self._g or source_node not in self._g:
            return
        self._add_structural_edge(artifact_node, source_node, Relation.DERIVED_FROM, seq)

    def chunk_nodes_for_doc(self, doc_id: str) -> list[str]:
        """Chunk nodes contained by a document, INCLUDING tombstoned ones. Used for the
        hybrid's update-identity seeds: the prior version's now-retired chunks are the
        structural certainty of what an update affects."""
        doc_node = make_node_id(NodeType.DOCUMENT, doc_id)
        if doc_node not in self._g:
            return []
        out: list[str] = []
        for _src, dst, key in self._g.out_edges(doc_node, keys=True):
            if key == Relation.CONTAINS.value:
                node = self.get_node(dst)
                if node is not None and node.type is NodeType.CHUNK:
                    out.append(dst)
        return out

    def degree(self, node_id: str) -> int:
        """Total incident-edge count (a coarse hub measure for seed down-weighting)."""
        return self._g.degree(node_id) if node_id in self._g else 0

    # ------------------------------------------------------------------ edges

    def _add_structural_edge(
        self, src: str, dst: str, relation: Relation, seq: int
    ) -> None:
        edge = GraphEdge(
            src=src,
            dst=dst,
            relation=relation,
            tier=EdgeTier.STRUCTURAL,
            weight=RELATION_WEIGHTS[relation],
        )
        # Edge key = relation, so a node pair can hold several relation types at once.
        self._g.add_edge(
            src,
            dst,
            key=relation.value,
            env=edge,
            weight=edge.weight,
            created_seq=seq,
            revoked_seq=None,
        )

    def freeze_similar_to(
        self, chunk_node: str, neighbors: Iterable[tuple[str, float]], seq: int
    ) -> None:
        """Freeze build-time similarity as ``similar_to`` edges (both directions so PPR
        propagates symmetrically). ``neighbors`` is ``(neighbor_chunk_node, confidence)``.
        Embeddings are used ONLY here, at build time — never to attach the arriving chunk.
        """
        for neighbor_node, confidence in neighbors:
            if neighbor_node == chunk_node or neighbor_node not in self._g:
                continue
            weight = RELATION_WEIGHTS[Relation.SIMILAR_TO] * float(confidence)
            for a, b in ((chunk_node, neighbor_node), (neighbor_node, chunk_node)):
                edge = GraphEdge(
                    src=a,
                    dst=b,
                    relation=Relation.SIMILAR_TO,
                    tier=EdgeTier.SEMANTIC,
                    weight=weight,
                    confidence=float(confidence),
                )
                self._g.add_edge(
                    a,
                    b,
                    key=Relation.SIMILAR_TO.value,
                    env=edge,
                    weight=weight,
                    created_seq=seq,
                    revoked_seq=None,
                )

    def assert_semantic_edge(
        self,
        src: str,
        dst: str,
        relation: Relation,
        decision_id: str,
        snapshot_seq: int,
        seq: int,
        confidence: float = 1.0,
        rationale: str | None = None,
    ) -> None:
        """Record a judge-asserted semantic relation (contradicts / duplicates /
        supersedes_content). Carries provenance: who asserted it, against which snapshot."""
        if tier_of(relation) is not EdgeTier.SEMANTIC:
            raise ValueError(f"{relation} is structural, not semantically assertable")
        edge = GraphEdge(
            src=src,
            dst=dst,
            relation=relation,
            tier=EdgeTier.SEMANTIC,
            weight=RELATION_WEIGHTS[relation] * float(confidence),
            confidence=float(confidence),
            asserted_by=make_node_id(NodeType.DECISION, decision_id),
            asserted_against=snapshot_seq,
            rationale=rationale,
        )
        self._g.add_edge(
            src,
            dst,
            key=relation.value,
            env=edge,
            weight=edge.weight,
            created_seq=seq,
            revoked_seq=None,
        )

    def revoke(self, src: str, dst: str, relation: Relation, seq: int) -> None:
        data = self._g.get_edge_data(src, dst, key=relation.value)
        if data is None:
            return
        env: GraphEdge = data["env"]
        if env.tier is EdgeTier.STRUCTURAL:
            raise ValueError("structural edges are never revoked")
        env.status = EdgeStatus.REVOKED
        data["revoked_seq"] = seq

    # ------------------------------------------------------------------ retire

    def retire(self, node_id: str, seq: int) -> list[GraphEdge]:
        """Tombstone (never hard-delete) and prune belief: walk the node's incident
        semantic edges, revoke the still-active ones (their ``asserted_against`` snapshot
        no longer holds once the node changes), and return them so the caller can
        re-judge the live successor. Structural edges are left untouched."""
        node = self.get_node(node_id)
        if node is None or node.status is NodeStatus.TOMBSTONED:
            return []
        node.status = NodeStatus.TOMBSTONED
        node.valid_to = seq
        node.retired_by_event = seq

        revoked: list[GraphEdge] = []
        # incident = out-edges + in-edges
        incident = list(self._g.out_edges(node_id, keys=True, data=True)) + list(
            self._g.in_edges(node_id, keys=True, data=True)
        )
        for src, dst, _key, data in incident:
            env: GraphEdge = data["env"]
            if env.tier is EdgeTier.SEMANTIC and env.status is EdgeStatus.ACTIVE:
                env.status = EdgeStatus.REVOKED
                data["revoked_seq"] = seq
                revoked.append(env)
        return revoked

    # ------------------------------------------------------------------ discovery

    def anchors_for(self, keys: ChunkKeys) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()

        def push(node_id: str | None) -> None:
            if node_id and node_id in self._g and node_id not in seen:
                seen.add(node_id)
                anchors.append(node_id)

        if keys.source is not None:
            push(self._source_index.get(keys.source))
        for name in keys.entities:
            push(self._entity_index.get(normalize_entity(name)))
        for iso in keys.dates:
            push(self._date_index.get(iso))
        for topic in keys.topics:
            push(self._topic_index.get(normalize_topic(topic)))
        return anchors

    def _ppr_graph(self) -> nx.Graph:
        """Undirected weighted projection over active (non-revoked) edges. Built fresh per
        call (corpus is ~100-200 docs). Tombstoned NODES are kept so chains propagate
        through them; revoked EDGES are dropped (belief pruned)."""
        h: nx.Graph = nx.Graph()
        h.add_nodes_from(self._g.nodes())
        for src, dst, data in self._g.edges(data=True):
            env: GraphEdge = data["env"]
            if env.status is EdgeStatus.REVOKED:
                continue
            w = data.get("weight", 1.0)
            if h.has_edge(src, dst):
                h[src][dst]["weight"] += w  # parallel relations reinforce
            else:
                h.add_edge(src, dst, weight=w)
        return h

    def ppr(
        self, seeds: list[str], top_n: int, exclude: set[str] | None = None
    ) -> list[str]:
        exclude = exclude or set()
        present = [s for s in seeds if s in self._g]
        if not present or top_n <= 0:
            return []

        h = self._ppr_graph()
        # Down-weight high-degree hub anchors so a generic date/entity doesn't dominate.
        personalization = {n: 0.0 for n in h.nodes()}
        for s in present:
            personalization[s] = 1.0 / (1.0 + h.degree(s))
        total = sum(personalization.values())
        if total <= 0:
            return []
        personalization = {n: v / total for n, v in personalization.items()}

        scores = _personalized_pagerank(h, personalization)

        seed_set = set(present)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[str] = []
        for node_id, score in ranked:
            if score <= 0 or node_id in seed_set or node_id in exclude:
                continue
            node = self.get_node(node_id)
            # Tombstoned nodes are walked through but excluded from the candidate set.
            if (
                node is not None
                and node.type is NodeType.CHUNK
                and node.status is NodeStatus.ACTIVE
            ):
                out.append(node_id)
                if len(out) >= top_n:
                    break
        return out

    def build_walk_digraph(
        self,
        forward: dict[Relation, float] | None = None,
        reverse: dict[Relation, float] | None = None,
    ) -> nx.DiGraph:
        """Directed walk graph for the hybrid's dual-seeded PPR. Each stored relation gets
        a forward transition weight and an EXPLICIT typed reverse edge (so consequence
        edges — derived_from, supersedes — are traversable both ways; we do not silently
        inherit NetworkX directedness). ``similar_to`` is already stored in both directions
        (frozen at build time, confidence-scaled), so it is copied with its stored weight
        and not given a synthetic reverse. Revoked edges are dropped; tombstoned nodes are
        kept so chains stay intact."""
        fwd = forward if forward is not None else RELATION_WEIGHTS
        rev = reverse if reverse is not None else RELATION_WEIGHTS
        dg: nx.DiGraph = nx.DiGraph()
        dg.add_nodes_from(self._g.nodes())

        def accumulate(a: str, b: str, weight: float) -> None:
            if weight <= 0:
                return
            if dg.has_edge(a, b):
                dg[a][b]["weight"] += weight
            else:
                dg.add_edge(a, b, weight=weight)

        for src, dst, data in self._g.edges(data=True):
            env: GraphEdge = data["env"]
            if env.status is EdgeStatus.REVOKED:
                continue
            if env.relation is Relation.SIMILAR_TO:
                accumulate(src, dst, float(data.get("weight", 1.0)))  # already bidirectional
                continue
            accumulate(src, dst, fwd.get(env.relation, 1.0))
            accumulate(dst, src, rev.get(env.relation, 1.0))  # typed reverse edge
        return dg

    def ppr_personalized(
        self,
        personalization: dict[str, float],
        top_n: int,
        *,
        restart: float = 0.2,
        exclude: set[str] | None = None,
        forward: dict[Relation, float] | None = None,
        reverse: dict[Relation, float] | None = None,
    ) -> list[str]:
        """Run the SHARED PPR kernel over the directed walk graph from an arbitrary
        teleport vector (mass mixed across chunk and entity seeds — the dual-seed payoff is
        addition into this vector, not score fusion). ``restart`` is the teleport
        probability (the locality / precision dial); the kernel's damping is ``1-restart``.
        Returns top-N active CHUNK nodes by visit probability. Seeds are NOT excluded — a
        vector- or update-seeded chunk is itself a legitimate candidate; only ``exclude``
        (e.g. the arriving chunk) and tombstoned/non-chunk nodes are dropped."""
        exclude = exclude or set()
        if top_n <= 0:
            return []
        seeds = {n: m for n, m in personalization.items() if m > 0 and n in self._g}
        if not seeds:
            return []

        dg = self.build_walk_digraph(forward, reverse)
        pers = {n: 0.0 for n in dg.nodes()}
        total = 0.0
        for node_id, mass in seeds.items():
            pers[node_id] += mass
            total += mass
        if total <= 0:
            return []
        pers = {n: v / total for n, v in pers.items()}

        scores = _personalized_pagerank(dg, pers, alpha=1.0 - restart)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[str] = []
        for node_id, score in ranked:
            if score <= 0 or node_id in exclude:
                continue
            node = self.get_node(node_id)
            if (
                node is not None
                and node.type is NodeType.CHUNK
                and node.status is NodeStatus.ACTIVE
            ):
                out.append(node_id)
                if len(out) >= top_n:
                    break
        return out

    def bfs_candidates(
        self, seeds: list[str], hops: int, top_n: int, exclude: set[str] | None = None
    ) -> list[str]:
        """In-baseline ablation: plain k-hop BFS + type filter instead of PPR. Selectable
        so the benchmark can isolate what PPR's weighting actually buys."""
        exclude = exclude or set()
        h = self._ppr_graph()  # same active-edge projection, unweighted traversal
        frontier = {s for s in seeds if s in h}
        visited = set(frontier)
        out: list[str] = []
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for n in frontier:
                for nbr in h.neighbors(n):
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    nxt.add(nbr)
                    node = self.get_node(nbr)
                    if (
                        node is not None
                        and node.type is NodeType.CHUNK
                        and node.status is NodeStatus.ACTIVE
                        and nbr not in exclude
                    ):
                        out.append(nbr)
            frontier = nxt
            if not frontier or len(out) >= top_n:
                break
        return out[:top_n]

    # ------------------------------------------------------------------ snapshot

    def snapshot(self) -> dict:
        """Compact temporal snapshot of current graph state for run logging.

        Returns counts per node type, edge tier/status counts, per-entity
        active-chunk tallies, and the size distribution of connected components
        in the active-chunk subgraph. Cheap at corpus scale (~100-200 docs).
        """
        from collections import defaultdict

        node_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"active": 0, "tombstoned": 0})
        for _, d in self._g.nodes(data=True):
            env: GraphNode = d["env"]
            status = "active" if env.status is NodeStatus.ACTIVE else "tombstoned"
            node_counts[env.type.value][status] += 1

        struct = sem_active = sem_revoked = 0
        for _, _, d in self._g.edges(data=True):
            env: GraphEdge = d["env"]
            if env.tier is EdgeTier.STRUCTURAL:
                struct += 1
            elif env.status is EdgeStatus.ACTIVE:
                sem_active += 1
            else:
                sem_revoked += 1

        # Per-entity: how many active chunks MENTION it (arrival-graph richness).
        entity_clusters = []
        for n, d in self._g.nodes(data=True):
            env: GraphNode = d["env"]
            if env.type is not NodeType.ENTITY or env.status is not NodeStatus.ACTIVE:
                continue
            active_chunks = sum(
                1
                for src, _, key in self._g.in_edges(n, keys=True)
                if key == Relation.MENTIONS.value
                and (cn := self.get_node(src)) is not None
                and cn.type is NodeType.CHUNK
                and cn.status is NodeStatus.ACTIVE
            )
            entity_clusters.append({
                "id": n,
                "name": env.attrs.get("name", ""),
                "type": env.attrs.get("entity_type", ""),
                "active_chunk_count": active_chunks,
            })
        entity_clusters.sort(key=lambda x: -x["active_chunk_count"])

        # Connected components of active-chunk subgraph (over non-revoked edges).
        active_chunk_ids = {
            n for n, d in self._g.nodes(data=True)
            if d["env"].type is NodeType.CHUNK and d["env"].status is NodeStatus.ACTIVE
        }
        g_chunks = nx.Graph()
        g_chunks.add_nodes_from(active_chunk_ids)
        for u, v, d in self._g.edges(data=True):
            env_e: GraphEdge = d["env"]
            if u in active_chunk_ids and v in active_chunk_ids and env_e.status is not EdgeStatus.REVOKED:
                g_chunks.add_edge(u, v)
        components = sorted(
            (len(c) for c in nx.connected_components(g_chunks)), reverse=True
        )

        return {
            "nodes": {k: dict(v) for k, v in node_counts.items()},
            "edges": {
                "structural": struct,
                "semantic_active": sem_active,
                "semantic_revoked": sem_revoked,
            },
            "entity_clusters": entity_clusters,
            "chunk_components": components,
        }

    # ------------------------------------------------------------------ views

    def active_chunk_node_ids(self) -> list[str]:
        return [
            n
            for n, d in self._g.nodes(data=True)
            if d["env"].type is NodeType.CHUNK
            and d["env"].status is NodeStatus.ACTIVE
        ]

    def active_view(self) -> nx.MultiDiGraph:
        active_nodes = [
            n for n, d in self._g.nodes(data=True) if d["env"].status is NodeStatus.ACTIVE
        ]
        sub = self._g.subgraph(active_nodes).copy()
        revoked = [
            (u, v, k)
            for u, v, k, d in sub.edges(keys=True, data=True)
            if d["env"].status is EdgeStatus.REVOKED
        ]
        sub.remove_edges_from(revoked)
        return sub

    def state_at(self, seq: int) -> nx.MultiDiGraph:
        nodes_at = [
            n for n, d in self._g.nodes(data=True) if d["env"].is_active_at(seq)
        ]
        sub = self._g.subgraph(nodes_at).copy()
        stale = [
            (u, v, k)
            for u, v, k, d in sub.edges(keys=True, data=True)
            if d.get("created_seq", 0) > seq
            or (d.get("revoked_seq") is not None and d["revoked_seq"] <= seq)
        ]
        sub.remove_edges_from(stale)
        return sub

    # ------------------------------------------------------------------ persistence

    def checkpoint(self, path: str | Path) -> None:
        """Serialise the graph + indexes to node-link-ish JSON for fast replay. The event
        log remains authoritative; this is purely an optimisation."""
        nodes = [_node_to_dict(d["env"]) for _, d in self._g.nodes(data=True)]
        edges = []
        for u, v, k, d in self._g.edges(keys=True, data=True):
            rec = _edge_to_dict(d["env"])
            rec["created_seq"] = d.get("created_seq")
            rec["revoked_seq"] = d.get("revoked_seq")
            edges.append(rec)
        blob = {
            "nodes": nodes,
            "edges": edges,
            "indexes": {
                "entity": self._entity_index,
                "date": self._date_index,
                "topic": self._topic_index,
                "source": self._source_index,
            },
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NetworkXGraphStore":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        for nd in blob["nodes"]:
            store._put_node(_node_from_dict(nd))
        for ed in blob["edges"]:
            env = _edge_from_dict(ed)
            store._g.add_edge(
                env.src,
                env.dst,
                key=env.relation.value,
                env=env,
                weight=env.weight,
                created_seq=ed.get("created_seq"),
                revoked_seq=ed.get("revoked_seq"),
            )
        idx = blob.get("indexes", {})
        store._entity_index = dict(idx.get("entity", {}))
        store._date_index = dict(idx.get("date", {}))
        store._topic_index = dict(idx.get("topic", {}))
        store._source_index = dict(idx.get("source", {}))
        return store


# ----------------------------------------------------------------------- ppr kernel


def _personalized_pagerank(
    h: nx.Graph,
    personalization: dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Power-iteration personalised PageRank over a small weighted graph (undirected for
    graph-only's projection, directed for the hybrid walk graph — the SAME kernel serves
    both; we never write a second PPR).

    Done in numpy rather than ``networkx.pagerank`` to avoid a scipy dependency — at this
    corpus scale (~100-200 docs) the dense iteration is trivially cheap. Dangling mass and
    teleport both redistribute to the personalization vector.
    """
    nodes = list(h.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}

    directed = h.is_directed()
    w = np.zeros((n, n), dtype=np.float64)
    for u, v, data in h.edges(data=True):
        weight = float(data.get("weight", 1.0))
        w[idx[u], idx[v]] += weight
        if not directed:
            w[idx[v], idx[u]] += weight  # undirected edges iterate once

    out = w.sum(axis=1)
    dangling = out == 0
    safe_out = np.where(dangling, 1.0, out)
    transition = w / safe_out[:, None]  # row-normalised: transition[j, i] = P(j -> i)

    p = np.array([personalization.get(node, 0.0) for node in nodes], dtype=np.float64)
    if p.sum() <= 0:
        return {}
    p = p / p.sum()

    r = p.copy()
    for _ in range(max_iter):
        prev = r
        leaked = alpha * prev[dangling].sum()
        r = (1.0 - alpha) * p + alpha * (transition.T @ prev) + leaked * p
        if np.abs(r - prev).sum() < tol:
            break
    return {node: float(r[idx[node]]) for node in nodes}


# ----------------------------------------------------------------------- (de)serialise


def _node_to_dict(n: GraphNode) -> dict:
    d = asdict(n)
    d["type"] = n.type.value
    d["status"] = n.status.value
    return d


def _node_from_dict(d: dict) -> GraphNode:
    return GraphNode(
        node_id=d["node_id"],
        type=NodeType(d["type"]),
        status=NodeStatus(d["status"]),
        merged_into=d.get("merged_into"),
        valid_from=d.get("valid_from", 0),
        valid_to=d.get("valid_to"),
        created_by_event=d.get("created_by_event", 0),
        retired_by_event=d.get("retired_by_event"),
        attrs=d.get("attrs", {}),
    )


def _edge_to_dict(e: GraphEdge) -> dict:
    d = asdict(e)
    d["relation"] = e.relation.value
    d["tier"] = e.tier.value
    d["status"] = e.status.value
    return d


def _edge_from_dict(d: dict) -> GraphEdge:
    return GraphEdge(
        src=d["src"],
        dst=d["dst"],
        relation=Relation(d["relation"]),
        tier=EdgeTier(d["tier"]),
        status=EdgeStatus(d["status"]),
        weight=d.get("weight", 1.0),
        confidence=d.get("confidence"),
        asserted_by=d.get("asserted_by"),
        asserted_against=d.get("asserted_against"),
        rationale=d.get("rationale"),
    )
