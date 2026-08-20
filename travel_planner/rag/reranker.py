"""对接用户提供的 RAG-Retrieval Cross-Encoder。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from travel_planner import logging as tp_logging
from travel_planner.rag.config import RagConfig


logger = tp_logging.get_logger(__name__)


class LocalRAGRetrievalReranker:
    """懒加载本地 RAG-Retrieval 模型，避免应用启动时占用数 GB 内存。"""

    def __init__(self, config: RagConfig):
        self.config = config
        self._ranker = None

    def _load(self):
        if self._ranker is not None:
            return self._ranker

        library_path = Path(self.config.rag_retrieval_path)
        model_path = Path(self.config.reranker_model_path)
        if not library_path.is_dir():
            raise FileNotFoundError(f"RAG-Retrieval library not found: {library_path}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Reranker model not found: {model_path}")

        library_value = str(library_path)
        if library_value not in sys.path:
            sys.path.insert(0, library_value)

        from rag_retrieval import Reranker

        logger.info("Loading local RAG-Retrieval reranker from %s", model_path)
        self._ranker = Reranker(
            str(model_path),
            model_type="cross-encoder",
            device=self.config.reranker_device,
            dtype=self.config.reranker_dtype,
            verbose=0,
        )
        if self._ranker is None:
            raise RuntimeError("RAG-Retrieval returned no reranker instance")
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        ranker = self._load()
        ranked = ranker.rerank(
            query,
            [candidate["content"] for candidate in candidates],
            batch_size=min(8, len(candidates)),
            max_length=self.config.reranker_max_length,
            normalize=True,
            long_doc_process_strategy="max_length_truncation",
        )

        output: list[dict[str, Any]] = []
        for result in ranked.top_k(top_k):
            candidate = dict(candidates[int(result.doc_id)])
            candidate["rerank_score"] = float(result.score) if result.score is not None else None
            output.append(candidate)
        return output
