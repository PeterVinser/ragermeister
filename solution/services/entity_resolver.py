"""Entity resolver — the shared extract -> block -> adjudicate -> attach pipeline.

Built ONCE and shared across all baselines (metadata-only and graph-only read the same
canonical entities, exactly as they share the contradiction judge). Resolution quality
must not vary across baselines, so nothing here is baseline-specific.

Two entry points, deliberately split read from write:

  * ``resolve(chunk)``  — read-only. Extract (cached), embed + block + adjudicate the gray
    band. Returns which canonical entities the chunk's mentions match. Mutates only the
    verdict/extraction caches, never the canonical store. Used for discovery, where an
    arriving (possibly-flagged, possibly-skipped) chunk must NOT create entities.
  * ``commit(chunk)``   — mutating. Applies the resolution: appends aliases on a merge,
    mints a canonical node on create, and logs each as a reversible resolve-type event.
    Reuses the cached ``resolve`` output, so it costs no extra extraction/adjudication.

Three-band adjudication on the top blocking score: ``>= tau_high`` auto-merge, ``< tau_low``
auto-create, in between the adjudicator decides. Every gray-band verdict is cached on
``(mention_signature, top_candidate_id)`` so no pair is judged twice and re-ingestion is
idempotent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from solution.models.chunk import Chunk
from solution.models.entity import (
    EntityCandidatesDecision,
    EntityCandidatesVerdict,
    EntityCandidate,
    EntityMention,
    EntityType,
    RESOLVABLE_TYPES,
)
from solution.services.embedder import Embedder
from solution.services.entity_candidates_judge import EntityCandidatesJudge
from solution.services.entity_extractor import EntityExtractor
from solution.services.entity_index import EntityIndex
from solution.services.event_log import EventLog

# Recall-tuned blocking defaults; precision comes from the adjudicator, not from these.
_BLOCK_K = 10
_BLOCK_FLOOR = 0.30
_TAU_HIGH = 0.86
_TAU_LOW = 0.55

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_MONTH_DAY = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b", re.IGNORECASE
)


def normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def normalize_topic(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.strip().lower()).strip("-")

@dataclass
class MentionResolution:
    """The fate of one resolvable (person/org) mention. ``canonical_id`` is the matched
    entity, or ``None`` until ``commit`` mints one. ``band`` records why."""

    mention: EntityMention
    canonical_id: str | None
    band: str  # "high" | "low" | "gray"
    score: float
    embedding: np.ndarray | None
    verdict: EntityCandidatesVerdict | None = None


@dataclass
class ChunkResolution:
    entities: list[MentionResolution] = field(default_factory=list)  # person/org
    dates: list[str] = field(default_factory=list)  # canonical date keys
    topics: list[str] = field(default_factory=list)  # controlled-vocab slugs


class EntityResolver:
    def __init__(
        self,
        extractor: EntityExtractor,
        adjudicator: EntityCandidatesJudge,
        embedder: Embedder | None = None,
        index: EntityIndex | None = None,
        event_log: EventLog | None = None,
        topic_vocab: dict[str, list[str]] | None = None,
        tau_high: float = _TAU_HIGH,
        tau_low: float = _TAU_LOW,
        block_k: int = _BLOCK_K,
        block_floor: float = _BLOCK_FLOOR,
    ) -> None:
        self._extractor = extractor
        self._adjudicator = adjudicator
        self._embedder = embedder if embedder is not None else Embedder()
        self._index = index if index is not None else EntityIndex()
        self._log = event_log if event_log is not None else EventLog()
        # slug -> surface patterns. None => fall back to a normalized free slug.
        self._topic_vocab = topic_vocab
        self._tau_high = tau_high
        self._tau_low = tau_low
        self._block_k = block_k
        self._block_floor = block_floor

        self._extract_cache: dict[str, list[EntityMention]] = {}
        self._resolution_cache: dict[str, ChunkResolution] = {}
        self._verdict_cache: dict[tuple[str, str], EntityCandidatesVerdict] = {}
        self._embed_cache: dict[str, np.ndarray] = {}
        self._committed: set[str] = set()

    @property
    def index(self) -> EntityIndex:
        return self._index

    # ------------------------------------------------------------------ read path

    def resolve(self, chunk: Chunk) -> ChunkResolution:
        """Resolve the chunk's mentions against the CURRENT canonical store, read-only.
        Cached per chunk so ``commit`` (and any re-judge) reuses it for free."""
        cached = self._resolution_cache.get(chunk.chunk_id)
        if cached is not None:
            return cached

        res = ChunkResolution()
        for mention in self._extract(chunk):
            if mention.type is EntityType.DATE:
                iso = self._normalize_date(mention.surface_form)
                if iso and iso not in res.dates:
                    res.dates.append(iso)
            elif mention.type is EntityType.TOPIC:
                slug = self._snap_topic(mention.surface_form)
                if slug and slug not in res.topics:
                    res.topics.append(slug)
            elif mention.type in RESOLVABLE_TYPES:
                res.entities.append(self._resolve_one(mention))
            # EntityType.OTHER is intentionally dropped (no shortcut, no pipeline).

        self._resolution_cache[chunk.chunk_id] = res
        return res

    def _resolve_one(self, mention: EntityMention) -> MentionResolution:
        emb = self._embed_mention(mention)
        candidates = self._index.block(
            emb, mention.type, self._block_k, self._block_floor
        )
        top = candidates[0].score if candidates else 0.0

        if not candidates or top < self._tau_low:
            return MentionResolution(mention, None, "low", top, emb)
        if top >= self._tau_high:
            return MentionResolution(mention, candidates[0].canonical_id, "high", top, emb)

        verdict = self._adjudicate(mention, candidates)
        matched = (
            verdict.candidate_id
            if verdict.decision is EntityCandidatesDecision.MERGE
            else None
        )
        return MentionResolution(mention, matched, "gray", top, emb, verdict)

    def _adjudicate(
        self, mention: EntityMention, candidates: list[EntityCandidate]
    ) -> EntityCandidatesVerdict:
        # Cache on (mention signature, top-candidate id): same pair is never re-judged.
        key = (self._signature(mention), candidates[0].canonical_id)
        cached = self._verdict_cache.get(key)
        if cached is not None:
            return cached
        verdict = self._adjudicator.judge(mention, candidates)
        self._verdict_cache[key] = verdict
        return verdict

    # ------------------------------------------------------------------ write path

    def commit(self, chunk: Chunk) -> ChunkResolution:
        """Apply the resolution to the canonical store and log it. Idempotent per chunk."""
        res = self.resolve(chunk)
        if chunk.chunk_id in self._committed:
            return res
        self._committed.add(chunk.chunk_id)

        # New entities minted within THIS chunk, keyed by signature, so a mention the
        # extractor failed to corefer doesn't spawn two canonical nodes in one chunk.
        minted: dict[str, str] = {}
        for mr in res.entities:
            if mr.canonical_id is not None:
                self._index.add_alias(mr.canonical_id, mr.mention.surface_form)
                self._log_merge(chunk, mr)
                continue
            sig = self._signature(mr.mention)
            existing = minted.get(sig)
            if existing is not None:
                mr.canonical_id = existing
                continue
            assert mr.embedding is not None  # set for every resolvable mention
            cid = self._index.create(
                canonical_name=mr.mention.surface_form,
                type=mr.mention.type,
                embedding=mr.embedding,
                surface_form=mr.mention.surface_form,
            )
            mr.canonical_id = cid
            minted[sig] = cid
            self._log_create(chunk, mr)
        return res

    # ------------------------------------------------------------------ helpers

    def _extract(self, chunk: Chunk) -> list[EntityMention]:
        cached = self._extract_cache.get(chunk.chunk_id)
        if cached is None:
            cached = self._extractor.extract(chunk)
            self._extract_cache[chunk.chunk_id] = cached
        return cached

    def _embed_mention(self, mention: EntityMention) -> np.ndarray:
        sig = self._signature(mention)
        cached = self._embed_cache.get(sig)
        if cached is not None:
            return cached
        # Embed name + type + context — never the bare surface form.
        text = (
            f"{mention.surface_form} [{mention.type.value}] {mention.short_context}".strip()
        )
        emb = self._embedder.embed([text])[0].astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        self._embed_cache[sig] = emb
        return emb

    def _signature(self, mention: EntityMention) -> str:
        return f"{mention.type.value}|{normalize_entity(mention.surface_form)}"

    def _normalize_date(self, surface: str) -> str | None:
        m = _ISO_DATE.search(surface)
        if m:
            return m.group(0)
        m = _MONTH_DAY.search(surface)
        if m:
            return f"{_MONTHS[m.group(1).lower()]}-{int(m.group(2)):02d}"
        slug = normalize_entity(surface).replace(" ", "-")
        return slug or None

    def _snap_topic(self, surface: str) -> str | None:
        low = normalize_entity(surface)
        if self._topic_vocab is None:
            return normalize_topic(surface) or None
        for slug, patterns in self._topic_vocab.items():
            if any(p in low for p in patterns):
                return slug
        return None  # off-vocabulary topics are dropped (controlled vocabulary)

    def _log_merge(self, chunk: Chunk, mr: MentionResolution) -> None:
        self._log.append(
            "entity_merge",
            {
                "chunk_id": chunk.chunk_id,
                "canonical_id": mr.canonical_id,
                "surface_form": mr.mention.surface_form,
                "type": mr.mention.type.value,
                "band": mr.band,
                "confidence": mr.verdict.confidence if mr.verdict else 1.0,
                "rationale": mr.verdict.rationale if mr.verdict else f"auto-merge ({mr.band})",
            },
        )

    def _log_create(self, chunk: Chunk, mr: MentionResolution) -> None:
        self._log.append(
            "entity_create",
            {
                "chunk_id": chunk.chunk_id,
                "canonical_id": mr.canonical_id,
                "surface_form": mr.mention.surface_form,
                "type": mr.mention.type.value,
                "band": mr.band,
            },
        )

    # ------------------------------------------------------------------ metric hook

    def alias_clusters(self) -> dict[str, dict]:
        return self._index.alias_clusters()
