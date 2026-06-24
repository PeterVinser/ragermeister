"""Graph node/edge envelopes for the graph-only housekeeper.

These envelopes are deliberately DB-agnostic: NetworkX is the in-process backend
today, but an embedded store (e.g. kuzu) could swap in later without the
housekeeper noticing. The graph stores NO embeddings — Chunk nodes only hold an
``embedding_id`` referencing the existing FAISS ``IndexIDMap2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"
    DATE = "date"
    TOPIC = "topic"
    ARTIFACT = "artifact"
    DECISION = "decision"
    # CLAIM is deferred. It slots between CHUNK and ENTITY later (chunk -expresses->
    # claim -mentions-> entity) without restructuring these envelopes.


# Typed prefixes for node ids. A stable, human-readable id keeps the event log and
# graph checkpoints diffable.
NODE_PREFIX: dict[NodeType, str] = {
    NodeType.DOCUMENT: "doc",
    NodeType.CHUNK: "chunk",
    NodeType.ENTITY: "ent",
    NodeType.DATE: "date",
    NodeType.TOPIC: "topic",
    NodeType.ARTIFACT: "art",
    NodeType.DECISION: "dec",
}


def make_node_id(node_type: NodeType, key: str) -> str:
    """``chunk:<key>``, ``ent:<key>`` ... — prefix is derived from the type."""
    return f"{NODE_PREFIX[node_type]}:{key}"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    TOMBSTONED = "tombstoned"
    MERGED = "merged"  # merged_into another node (entity resolution); see merged_into


class EdgeTier(str, Enum):
    STRUCTURAL = "structural"  # deterministic, never revoked
    SEMANTIC = "semantic"  # judge-asserted (or frozen build-time similar_to)


class EdgeStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class Relation(str, Enum):
    # --- structural (deterministic, never revoked) ---
    CONTAINS = "contains"  # document -> chunk
    MENTIONS = "mentions"  # chunk -> entity
    ON_DATE = "on_date"  # chunk -> date
    HAS_TOPIC = "has_topic"  # chunk -> topic
    DERIVED_FROM = "derived_from"  # artifact -> chunk/doc
    VERSION_OF = "version_of"  # doc version -> prior doc version
    SUPERSEDES = "supersedes"  # structural version supersession (doc-level)

    # --- semantic (judge-asserted, revocable; similar_to is frozen build-time) ---
    CONTRADICTS = "contradicts"
    DUPLICATES = "duplicates"
    SUPERSEDES_CONTENT = "supersedes_content"  # content-level supersession
    SIMILAR_TO = "similar_to"  # frozen at build time from embeddings


STRUCTURAL_RELATIONS: frozenset[Relation] = frozenset(
    {
        Relation.CONTAINS,
        Relation.MENTIONS,
        Relation.ON_DATE,
        Relation.HAS_TOPIC,
        Relation.DERIVED_FROM,
        Relation.VERSION_OF,
        Relation.SUPERSEDES,
    }
)

SEMANTIC_RELATIONS: frozenset[Relation] = frozenset(
    {
        Relation.CONTRADICTS,
        Relation.DUPLICATES,
        Relation.SUPERSEDES_CONTENT,
        Relation.SIMILAR_TO,
    }
)


def tier_of(relation: Relation) -> EdgeTier:
    return (
        EdgeTier.STRUCTURAL
        if relation in STRUCTURAL_RELATIONS
        else EdgeTier.SEMANTIC
    )


@dataclass
class GraphNode:
    """Node envelope. ``valid_from``/``valid_to`` are event seqs (a temporal view);
    ``status`` carries belief. The graph grows monotonically in structure and prunes
    only in belief — nodes are tombstoned, never hard-deleted."""

    node_id: str
    type: NodeType
    status: NodeStatus = NodeStatus.ACTIVE
    merged_into: str | None = None  # set when status == MERGED
    valid_from: int = 0  # event seq this node became active
    valid_to: int | None = None  # event seq it was retired (None == still valid)
    created_by_event: int = 0
    retired_by_event: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)  # type-specific payload

    def is_active_at(self, seq: int) -> bool:
        if self.created_by_event > seq:
            return False
        if self.retired_by_event is not None and self.retired_by_event <= seq:
            return False
        return True


@dataclass
class GraphEdge:
    """Edge envelope. ``key`` in the MultiDiGraph is the relation, so the same node
    pair can simultaneously hold e.g. ``similar_to`` and ``contradicts``."""

    src: str
    dst: str
    relation: Relation
    tier: EdgeTier
    status: EdgeStatus = EdgeStatus.ACTIVE
    weight: float = 1.0

    # semantic-only provenance (None on structural edges)
    confidence: float | None = None
    asserted_by: str | None = None  # decision node id (dec:<...>)
    asserted_against: int | None = None  # snapshot seq the assertion was made against
    rationale: str | None = None
