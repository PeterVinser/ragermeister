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
from solution.services.vector_db import VectorDB

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64
_NEIGHBOR_K = 5


def _split(text: str, doc_id: str) -> list[Chunk]:
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=text[start:end],
                metadata={"char_start": start, "char_end": min(end, len(text))},
            )
        )
        if end >= len(text):
            break
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


class KnowledgeBase:
    """
    Wiring note: KnowledgeBase and ResolutionManager are mutually dependent.
    Construct KB first with conflict_sink=None, then create the manager,
    then set kb.conflict_sink = manager.submit.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        docstore: Docstore,
        judge: ConflictJudge,
        conflict_sink: Callable[[ConflictReport], None] | None = None,
    ) -> None:
        self._embedder = Embedder()
        self._vdb = vector_db
        self._docstore = docstore
        self._judge = judge
        self.conflict_sink = conflict_sink
        self._pending_embeddings: dict[str, np.ndarray] = {}

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        emb = self._embedder.embed([text])
        vec_ids, _ = self._vdb.search(emb, k)
        return [
            chunk
            for vid in vec_ids
            if (chunk := self._docstore.get_by_vec_id(vid)) is not None
        ]

    def ingest(self, event: IngestEvent) -> None:
        if event.event_type == EventType.DELETE:
            self._delete_doc(event.doc_id)
            return

        if event.event_type == EventType.UPDATE:
            self._delete_doc(event.doc_id)

        chunks = _split(event.text, event.doc_id)
        if not chunks:
            return

        embeddings = self._embedder.embed([c.text for c in chunks])

        for chunk, emb in zip(chunks, embeddings):
            vec_ids, _ = self._vdb.search(emb.reshape(1, -1), _NEIGHBOR_K)
            neighbors = [
                n
                for vid in vec_ids
                if (n := self._docstore.get_by_vec_id(vid)) is not None
            ]
            result = self._judge.judge(chunk, neighbors)

            if result.label == ConflictLabel.CLEAN:
                self._commit(chunk, emb)
            else:
                self._pending_embeddings[chunk.chunk_id] = emb
                if self.conflict_sink is None:
                    raise RuntimeError("conflict_sink not set — cannot route conflict report")
                self.conflict_sink(
                    ConflictReport(
                        report_id=str(uuid.uuid4()),
                        new_chunk=chunk,
                        judge_result=result,
                    )
                )

    def apply_decision(self, decision: Decision) -> None:
        for chunk_id in decision.chunk_ids_to_remove:
            chunk = self._docstore.remove_chunk(chunk_id)
            if chunk and chunk.vec_id >= 0:
                self._vdb.remove([chunk.vec_id])

        if decision.action in (DecisionAction.INSERT, DecisionAction.UPDATE):
            chunk = decision.new_chunk
            if chunk is not None:
                emb = self._pending_embeddings.pop(chunk.chunk_id, None)
                if emb is None:
                    emb = self._embedder.embed([chunk.text])[0]
                self._commit(chunk, emb)
        elif decision.action in (DecisionAction.DELETE, DecisionAction.SKIP):
            if decision.new_chunk:
                self._pending_embeddings.pop(decision.new_chunk.chunk_id, None)

    def _commit(self, chunk: Chunk, emb: np.ndarray) -> None:
        [vec_id] = self._vdb.add(emb.reshape(1, -1))
        chunk.vec_id = vec_id
        self._docstore.add(chunk)

    def _delete_doc(self, doc_id: str) -> None:
        removed = self._docstore.remove_doc(doc_id)
        vec_ids = [c.vec_id for c in removed if c.vec_id >= 0]
        if vec_ids:
            self._vdb.remove(vec_ids)
