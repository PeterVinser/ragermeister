from openai import AzureOpenAI
import numpy as np

from solution.services import cost_meter

class Embedder:
    def __init__(self):
        self.client = AzureOpenAI()

    def embed(self, text: list[str]) -> np.ndarray:
        cost_meter.record("embed_calls", 1)
        cost_meter.record("embed_texts", len(text))

        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )

        embeddings = [e.embedding for e in response.data]

        return np.array(embeddings, dtype=np.float32)