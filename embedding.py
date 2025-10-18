from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

Vector = List[float]
Matrix = List[Vector]


class BaseEmbedder(ABC):
    """
    Abstract interface for text embedding backends used during clustering.
    """

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> Matrix:
        """
        Return a matrix of shape (len(texts), embedding_dim).

        Implementations are expected to return floating point vectors
        (lists or tuples). Callers handle any required normalisation.
        """


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]
