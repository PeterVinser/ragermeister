from abc import ABC, abstractmethod

import numpy as np


class EmbedderInterface(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings, shape (len(texts), dim)."""
