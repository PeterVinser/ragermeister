import uuid
from typing import Callable

import numpy as np

from solution.services.embedder import Embedder
from solution.models.chunk import Chunk
from solution.models.conflict import (
    ConflictLabel,
    ConflictReport,
    Decision,
    DecisionAction,
)
from solution.models.event import EventType, IngestEvent
from solution.services.conflict_judge import ConflictJudge
from solution.services.docstore import Docstore
from solution.services.house_keeper import HouseKeeper
from solution.services.vector_db import VectorDB


class KnowledgeBase:
    """Baseline-agnostic knowledge base. It always embeds and stores chunks — it is the
    retrieval ground truth — and it never reasons about consistency itself.

    Consistency monitoring is delegated to an injected, swappable ``HouseKeeper``: before
    committing an arrival the KB asks the housekeeper for candidate neighbours and feeds
    them to the SHARED judge. Baselines (vector-only / graph-only / hybrid) differ ONLY in
    which housekeeper is wired in; the embed/judge/commit/route machinery here is identical
    for all of them, which is what keeps the eventual comparison honest (invariant #3).
    The housekeeper reads the base but never mutates it.

    Wiring note: KnowledgeBase and ResolutionManager are mutually dependent. Construct the
    KB with ``conflict_sink=None``, create the manager pointing at ``apply_decision``, then
    set ``kb.conflict_sink = manager.submit``.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        docstore: Docstore,
        judge: ConflictJudge,
        housekeeper: HouseKeeper,
        conflict_sink: Callable[[ConflictReport], None] | None = None,
        revalidate_on_retire: bool = False,
    ) -> None:
        self._embedder = Embedder()
        self._vdb = vector_db
        self._docstore = docstore
        self._judge = judge  # SHARED across baselines — never specialised here
        self._housekeeper = housekeeper
        self.conflict_sink = conflict_sink
        self._revalidate_on_retire = revalidate_on_retire
        self._pending_embeddings: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ query

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        emb = self._embedder.embed([text])
        vec_ids, _ = self._vdb.search(emb, k)
        return [
            chunk
            for vid in vec_ids
            if (chunk := self._docstore.get_by_vec_id(vid)) is not None
        ]

    # ------------------------------------------------------------------ ingest

    def ingest(self, event: IngestEvent) -> None:
        if event.event_type == EventType.DELETE:
            self._retire_doc(event.doc_id)
            return

        if event.event_type == EventType.UPDATE:
            self._retire_doc(event.doc_id)

        # Preserve all event attributes (title, etc.) on the chunk — housekeepers read
        # them as facets — while guaranteeing source_id is always populated.
        metadata = dict(event.metadata)
        metadata["source_id"] = event.metadata.get("source_id") or event.doc_id
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=event.doc_id,
            text=event.text,
            metadata=metadata,
        )

        embeddings = self._embedder.embed([chunk.text])
        if embeddings.shape[0] == 0:
            return
        emb = embeddings[0]

        self._detect(chunk, emb)

    def _detect(self, chunk: Chunk, emb: np.ndarray) -> None:
        """Detection ALWAYS runs and ALWAYS emits a signal (invariant #2): clean commits,
        flagged routes a report. The housekeeper supplies candidates; the shared judge
        labels them."""
        candidates = self._housekeeper.find_candidates(chunk, emb)
        result = self._judge.judge(chunk, candidates)

        if result.label == ConflictLabel.CLEAN:
            self._commit(chunk, emb)
        else:
            self._pending_embeddings[chunk.chunk_id] = emb
            if self.conflict_sink is None:
                raise RuntimeError(
                    "conflict_sink not set — cannot route conflict report"
                )
            self.conflict_sink(
                ConflictReport(
                    report_id=str(uuid.uuid4()),
                    new_chunk=chunk,
                    judge_result=result,
                )
            )

    # ------------------------------------------------------------------ decisions

    def apply_decision(self, decision: Decision) -> None:
        successors = self._retire_chunks(decision.chunk_ids_to_remove)

        if decision.action in (DecisionAction.INSERT, DecisionAction.UPDATE):
            chunk = decision.new_chunk
            if chunk is not None:
                emb = self._pending_embeddings.pop(chunk.chunk_id, None)
                if emb is None:
                    emb = self._embedder.embed([chunk.text])[0]
                self._commit(chunk, emb)
                self._housekeeper.on_resolution(decision, chunk)
        elif decision.action in (DecisionAction.DELETE, DecisionAction.SKIP):
            if decision.new_chunk:
                self._pending_embeddings.pop(decision.new_chunk.chunk_id, None)

        # Diachronic re-check: a committed change may retroactively conflict with chunks
        # the housekeeper flags as affected. Re-judge them through the same path. Only the
        # graph housekeeper surfaces successors here; vector-only returns none.
        if self._revalidate_on_retire and successors:
            self._revalidate(successors)

    def _revalidate(self, successors: list[Chunk]) -> None:
        for chunk in successors:
            emb = self._embedder.embed([chunk.text])[0]
            candidates = [
                n
                for n in self._housekeeper.find_candidates(chunk, emb)
                if n.chunk_id != chunk.chunk_id
            ]
            result = self._judge.judge(chunk, candidates)
            if result.label != ConflictLabel.CLEAN and self.conflict_sink is not None:
                self.conflict_sink(
                    ConflictReport(
                        report_id=str(uuid.uuid4()),
                        new_chunk=chunk,
                        judge_result=result,
                    )
                )

    # ------------------------------------------------------------------ commit / retire

    def _commit(self, chunk: Chunk, emb: np.ndarray) -> None:
        [vec_id] = self._vdb.add(emb.reshape(1, -1))
        chunk.vec_id = vec_id
        self._docstore.add(chunk)
        # Housekeeper mirrors the commit into its own structures AFTER the base has it.
        self._housekeeper.on_commit(chunk, emb)

    def _retire_doc(self, doc_id: str) -> None:
        self._retire_chunks(self._docstore.get_doc_chunk_ids(doc_id))

    def _retire_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        # Notify the housekeeper first (so it can read soon-to-be-removed chunks), then
        # remove from the base. Returns any live chunks the housekeeper wants re-judged.
        successors = self._housekeeper.on_retire(list(chunk_ids))
        for chunk_id in chunk_ids:
            chunk = self._docstore.remove_chunk(chunk_id)
            if chunk and chunk.vec_id >= 0:
                self._vdb.remove([chunk.vec_id])
        return successors
