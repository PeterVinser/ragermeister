import numpy as np
import faiss


class VectorDB:
    def __init__(self, dim: int) -> None:
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._next_id = 0

    def add(self, embeddings: np.ndarray) -> list[int]:
        """Add rows of normalized embeddings; return the assigned FAISS IDs."""
        n = embeddings.shape[0]
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        self._index.add_with_ids(embeddings, ids)
        return ids.tolist()

    def remove(self, vec_ids: list[int]) -> None:
        ids = np.array(vec_ids, dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(ids))

    def search(self, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """query shape: (1, dim). Returns (vec_ids, scores) for top-k results."""
        if self._index.ntotal == 0:
            return [], []
        k = min(k, int(self._index.ntotal))
        scores, ids = self._index.search(query, k)
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        if not pairs:
            return [], []
        id_list, score_list = zip(*pairs)
        return list(id_list), list(score_list)
