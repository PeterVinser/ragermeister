from solution.models.chunk import Chunk

class Docstore:
    def __init__(self) -> None:
        self._doc_chunks: dict[str, list[str]] = {}
        self._chunks: dict[str, Chunk] = {}
        self._vec_to_chunk: dict[int, str] = {}

    def add(self, chunk: Chunk) -> None:
        self._chunks[chunk.chunk_id] = chunk
        self._vec_to_chunk[chunk.vec_id] = chunk.chunk_id
        self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def remove_chunk(self, chunk_id: str) -> Chunk | None:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return None
        self._vec_to_chunk.pop(chunk.vec_id, None)
        try:
            self._doc_chunks.get(chunk.doc_id, []).remove(chunk_id)
        except ValueError:
            pass
        return chunk

    def remove_doc(self, doc_id: str) -> list[Chunk]:
        chunk_ids = self._doc_chunks.pop(doc_id, [])
        return [c for cid in chunk_ids if (c := self.remove_chunk(cid)) is not None]

    def get_by_vec_id(self, vec_id: int) -> Chunk | None:
        chunk_id = self._vec_to_chunk.get(vec_id)
        return self._chunks.get(chunk_id) if chunk_id else None

    def get_doc_chunk_ids(self, doc_id: str) -> list[str]:
        return list(self._doc_chunks.get(doc_id, []))
