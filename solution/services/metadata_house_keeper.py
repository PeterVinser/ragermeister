"""Metadata-only housekeeper — faceted attribute-match candidate discovery.

The fourth candidate-discovery strategy behind the shared ``HouseKeeper`` interface. It
reuses, unchanged, the same machinery as graph-only: the entity resolver (canonical
entities — so facets are post-resolution identities, not raw surface strings), the shared
contradiction judge (owned by the KB), and the resolution manager. ONLY discovery differs.

Artifact: one inverted index per facet (plain dicts) built from the resolver's
extract+resolve output:
    entity_id   -> {chunk_ids}
    iso_date    -> {chunk_ids}
    topic       -> {chunk_ids}
    source_id   -> {doc_ids}      (document-level facet)
    title_token -> {chunk_ids}

Discovery, per arriving chunk:
  1. Extract + resolve its facets (NO KB retrieval — same pass graph-only runs).
  2. Look each facet up in its index; UNION the posting lists into a candidate set.
  3. Rank by attribute specificity: each candidate scores the summed inverse document
     frequency (IDF) of the facets it SHARES with the arrival. A rare shared entity is
     strong; a shared source_id is weak. Specificity-weighted OR + top-N, not a hard AND
     threshold (AND would miss single-shared-attribute cases like the entity-mediated 7).
  4. Hand top-N to the shared judge.

Honest-baseline constraints (deliberate, do NOT patch):
  * Reads attribute *values* only. It holds no typed edges (supersedes / derived_from /
    contradicts) and never traverses — that is graph-only's contribution.
  * SINGLE HOP, no transitivity. It structurally cannot follow a supersession chain
    (seq 1->5->10) or a derived_from dependency (seq 8). ``on_retire`` therefore returns
    ``[]``: with no edges it cannot find stale successors.
  * Capped by the SAME entity-resolution quality as graph-only (under-merge => chunks that
    should share a canonical entity don't => missed). Expected and shared.
"""

from __future__ import annotations

import math
import re

import numpy as np

from solution.models.chunk import Chunk
from solution.services.docstore import Docstore
from solution.services.entity_resolver import ChunkResolution, EntityResolver
from solution.services.house_keeper import HouseKeeper

_METADATA_TOP_N = 5  # candidate-size dial — tune comparably to vector-k and graph-PPR-N

# Tiny stoplist so generic title words don't become spurious shared facets.
_TITLE_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "in", "on", "at", "by", "with",
    "is", "are", "as", "or",
}

# A facet is a (kind, value) pair; kinds map to the per-facet inverted indexes.
Facet = tuple[str, str]


class MetadataHouseKeeper(HouseKeeper):
    def __init__(
        self,
        docstore: Docstore,
        resolver: EntityResolver,
        top_n: int = _METADATA_TOP_N,
    ) -> None:
        # Read-only handle on the base (parity with the other housekeepers; discovery here
        # never actually needs it since we keep our own chunk store).
        self._docstore = docstore
        self._resolver = resolver  # shared ER infrastructure — canonical entities
        self._top_n = top_n

        # Inverted indexes (one dict per facet). Entity/date/topic/title post chunk_ids;
        # source posts doc_ids (a document-level attribute).
        self._postings: dict[str, dict[str, set[str]]] = {
            "entity": {}, "date": {}, "topic": {}, "title": {}
        }
        self._source_index: dict[str, set[str]] = {}  # source_id -> {doc_ids}

        # Bookkeeping for scoring + retirement.
        self._chunks: dict[str, Chunk] = {}
        self._chunk_facets: dict[str, set[Facet]] = {}
        self._chunk_doc: dict[str, str] = {}
        self._doc_chunks: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ discovery

    def find_candidates(self, chunk: Chunk, embedding: np.ndarray) -> list[Chunk]:
        # Read-only resolve (no minting): same extraction pass graph-only uses. The
        # embedding is unused — discovery here is purely attribute match.
        resolution = self._resolver.resolve(chunk)
        facets = self._facets_of(chunk, resolution)
        if not facets:
            return []

        # Union the posting lists of each facet into the candidate set.
        candidates: set[str] = set()
        for kind, value in facets:
            if kind == "source":
                for doc_id in self._source_index.get(value, ()):
                    candidates |= self._doc_chunks.get(doc_id, set())
            else:
                candidates |= self._postings[kind].get(value, set())
        candidates.discard(chunk.chunk_id)
        if not candidates:
            return []

        # Rank by summed IDF of SHARED facets (specificity weighting), then chunk_id for
        # a deterministic tie-break. Top-N go to the judge.
        scored: list[tuple[float, str]] = []
        for cid in candidates:
            other = self._chunk_facets.get(cid)
            if other is None:
                continue
            shared = facets & other
            if not shared:
                continue
            score = sum(self._idf(kind, value) for kind, value in shared)
            scored.append((score, cid))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._chunks[cid] for _, cid in scored[: self._top_n] if cid in self._chunks]

    def _idf(self, kind: str, value: str) -> float:
        """Smoothed inverse document frequency of a facet. Rare facet -> high weight.
        Source frequency is measured in documents; the rest in chunks."""
        if kind == "source":
            df = len(self._source_index.get(value, ()))
            n = max(len(self._doc_chunks), 1)
        else:
            df = len(self._postings[kind].get(value, ()))
            n = max(len(self._chunks), 1)
        return math.log((n + 1) / (df + 1)) + 1.0

    def _facets_of(self, chunk: Chunk, resolution: ChunkResolution) -> set[Facet]:
        facets: set[Facet] = set()
        for mr in resolution.entities:
            if mr.canonical_id is not None:  # unresolved-new mentions have no postings
                facets.add(("entity", mr.canonical_id))
        for iso in resolution.dates:
            facets.add(("date", iso))
        for topic in resolution.topics:
            facets.add(("topic", topic))
        source_id = chunk.metadata.get("source_id")
        if source_id:
            facets.add(("source", source_id))
        for token in self._title_tokens(chunk):
            facets.add(("title", token))
        return facets

    @staticmethod
    def _title_tokens(chunk: Chunk) -> set[str]:
        title = chunk.metadata.get("title") or ""
        toks = re.findall(r"[a-z0-9]+", title.lower())
        return {t for t in toks if len(t) > 2 and t not in _TITLE_STOP}

    # ------------------------------------------------------------------ lifecycle

    def on_commit(self, chunk: Chunk, embedding: np.ndarray) -> None:
        """Index the committed chunk's facets. ``commit`` mints/merges canonical entities
        (reusing the read-path cache), so entity facets are post-resolution ids."""
        resolution = self._resolver.commit(chunk)
        facets = self._facets_of(chunk, resolution)

        self._chunks[chunk.chunk_id] = chunk
        self._chunk_facets[chunk.chunk_id] = facets
        self._chunk_doc[chunk.chunk_id] = chunk.doc_id
        self._doc_chunks.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

        for kind, value in facets:
            if kind == "source":
                self._source_index.setdefault(value, set()).add(chunk.doc_id)
            else:
                self._postings[kind].setdefault(value, set()).add(chunk.chunk_id)

    def on_retire(self, chunk_ids: list[str]) -> list[Chunk]:
        """Drop the retired chunks from every index. Returns ``[]`` — a single-hop,
        edge-free matcher structurally cannot find stale successors of a removed chunk.
        That blindness is the intended contrast with graph-only; it is not patched."""
        for cid in chunk_ids:
            facets = self._chunk_facets.pop(cid, set())
            self._chunks.pop(cid, None)
            doc_id = self._chunk_doc.pop(cid, None)
            source_id: str | None = None
            for kind, value in facets:
                if kind == "source":
                    source_id = value
                else:
                    posting = self._postings[kind].get(value)
                    if posting is not None:
                        posting.discard(cid)
            if doc_id is not None:
                doc_chunks = self._doc_chunks.get(doc_id)
                if doc_chunks is not None:
                    doc_chunks.discard(cid)
                    if not doc_chunks:  # doc has no live chunks -> drop its source posting
                        self._doc_chunks.pop(doc_id, None)
                        if source_id is not None:
                            src = self._source_index.get(source_id)
                            if src is not None:
                                src.discard(doc_id)
        return []
