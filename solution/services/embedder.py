import numpy as np
from sentence_transformers import SentenceTransformer

from solution.interfaces.embedder import EmbedderInterface


class SentenceTransformerEmbedder(EmbedderInterface):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(
            self._model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )
