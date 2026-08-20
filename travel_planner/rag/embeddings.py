"""Local dense and Chinese-friendly sparse embedding providers."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from travel_planner.rag.chunking import tokenize_search_text
from travel_planner.rag.config import RagConfig


class LocalHybridEmbeddings:
    """Lazy FastEmbed dense model plus deterministic hashed sparse vectors."""

    def __init__(self, config: RagConfig):
        self.config = config
        self._dense_model = None

    def _load_dense_model(self):
        if self._dense_model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required; install qdrant-client[fastembed]"
                ) from exc
            cache_dir = Path(self.config.model_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._dense_model = TextEmbedding(
                model_name=self.config.dense_model,
                cache_dir=str(cache_dir),
            )
        return self._dense_model

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        model = self._load_dense_model()
        return [
            vector.tolist()
            for vector in model.passage_embed(
                values,
                batch_size=self.config.embedding_batch_size,
            )
        ]

    def embed_query(self, text: str) -> list[float]:
        model = self._load_dense_model()
        return list(model.query_embed(text))[0].tolist()

    @staticmethod
    def _token_index(token: str) -> int:
        # Qdrant sparse indices are uint32. BLAKE2 keeps IDs stable across runs.
        return int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(),
            "big",
        )

    def embed_sparse(self, text: str, *, query: bool = False) -> tuple[list[int], list[float]]:
        counts = Counter(tokenize_search_text(text))
        if not counts:
            return [], []

        weighted: dict[int, float] = {}
        for token, frequency in counts.items():
            index = self._token_index(token)
            # Saturating TF; Qdrant's IDF modifier supplies corpus-level rarity.
            value = 1.0 if query else (frequency * 2.2) / (frequency + 1.2)
            weighted[index] = weighted.get(index, 0.0) + value

        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        ordered = sorted(weighted.items())
        return [index for index, _ in ordered], [value / norm for _, value in ordered]
