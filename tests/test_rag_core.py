import unittest
from dataclasses import replace
from unittest.mock import patch

from travel_planner.rag.chunking import build_search_text, chunk_text
from travel_planner.rag.config import get_rag_config
from travel_planner.rag.embeddings import LocalHybridEmbeddings
from travel_planner.rag.query_processing import QueryProcessor
from travel_planner.rag.retriever import (
    TravelKnowledgeRetriever,
    deduplicate_candidates,
    reciprocal_rank_fusion,
)
from travel_planner.rag.schemas import MetadataConditions, QueryPlan, QueryRewriteResult


def candidate(chunk_id, document_id, title, content):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "title": title,
        "content": content,
        "city": "北京",
        "topic": "family",
        "metadata": {},
    }


class FakeStore:
    def __init__(self):
        self.calls = []

    def ensure_collection(self):
        self.calls.append(("ensure",))

    def search_dense(self, query, *, filters=None, limit=20):
        self.calls.append(("dense", query, filters, limit))
        values = [
            candidate("c1", "d1", "故宫亲子路线", "故宫上午参观，需要预约。"),
            candidate("c2", "d2", "北京交通", "乘坐地铁前往天坛，避开晚高峰。"),
            candidate("c4", "d4", "北京住宿", "建议住在地铁站附近。"),
        ]
        for score, item in zip((0.9, 0.8, 0.7), values, strict=True):
            item["dense_score"] = score
        return values

    def search_sparse(self, query, *, filters=None, limit=20):
        self.calls.append(("sparse", query, filters, limit))
        values = [
            candidate("c2", "d2", "北京交通", "乘坐地铁前往天坛，避开晚高峰。"),
            candidate("c3", "d3", "北京美食", "晚餐可以安排北京烤鸭。"),
            candidate("c1", "d1", "故宫亲子路线", "故宫上午参观，需要预约。"),
        ]
        for score, item in zip((4.0, 3.0, 2.0), values, strict=True):
            item["sparse_score"] = score
        return values


class FakeReranker:
    def rerank(self, query, candidates, *, top_k):
        output = []
        for index, value in enumerate(candidates[:top_k]):
            item = dict(value)
            item["rerank_score"] = 1.0 - index / 10
            output.append(item)
        return output


class FakeQueryProcessor:
    def plan(self, query, *, overrides=None):
        values = {"city": "北京", "topic": "family"}
        values.update({key: value for key, value in (overrides or {}).items() if value})
        return QueryPlan(
            original_query=query,
            rewritten_query="北京 三日 亲子 行程 故宫 地铁",
            filters=MetadataConditions.model_validate(values),
        )


class RagCoreTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            get_rag_config(),
            query_rewrite_enabled=False,
            metadata_extraction_enabled=False,
        )

    def test_chunking_and_chinese_search_tokens(self):
        chunks = chunk_text("北京亲子旅行。" * 200, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertIn("北京", build_search_text("北京亲子旅行"))
        self.assertIn("亲子", build_search_text("北京亲子旅行"))

    def test_sparse_embedding_is_stable_and_nonempty(self):
        embeddings = LocalHybridEmbeddings(self.config)
        first = embeddings.embed_sparse("北京亲子三日游 故宫预约")
        second = embeddings.embed_sparse("北京亲子三日游 故宫预约")
        self.assertEqual(first, second)
        self.assertGreater(len(first[0]), 3)

    def test_query_processing_fallback_rewrites_then_extracts_metadata(self):
        processor = QueryProcessor(self.config)
        plan = processor.plan("请安排北京亲子三日游，重点是历史文化和地铁交通")
        self.assertIn("北京", plan.rewritten_query)
        self.assertEqual("北京", plan.filters.city)
        self.assertEqual("family", plan.filters.topic)

    def test_llm_query_rewrite_precedes_llm_metadata_extraction(self):
        processor = QueryProcessor(
            replace(
                self.config,
                query_rewrite_enabled=True,
                metadata_extraction_enabled=True,
            ),
            model=object(),
        )
        with patch(
            "travel_planner.rag.query_processing.safe_structured_output",
            side_effect=[
                QueryRewriteResult(rewritten_query="北京 五日 亲子 故宫 长城 地铁"),
                MetadataConditions(city="北京", topic="family"),
            ],
        ) as structured:
            plan = processor.plan("带两个孩子去北京玩五天")
        self.assertEqual(2, structured.call_count)
        self.assertEqual("北京 五日 亲子 故宫 长城 地铁", plan.rewritten_query)
        self.assertEqual("北京", plan.filters.city)

    def test_rrf_boosts_chunks_seen_by_both_branches(self):
        dense = [candidate("a", "d1", "A", "A"), candidate("b", "d2", "B", "B")]
        sparse = [candidate("b", "d2", "B", "B"), candidate("c", "d3", "C", "C")]
        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
        self.assertEqual("b", fused[0]["chunk_id"])
        self.assertEqual(2, fused[0]["dense_rank"])
        self.assertEqual(1, fused[0]["sparse_rank"])

    def test_dedup_removes_same_document_and_near_duplicate(self):
        values = [
            candidate("a", "d1", "A", "故宫上午参观需要提前预约门票"),
            candidate("b", "d1", "B", "天坛下午参观"),
            candidate("c", "d2", "C", "故宫上午参观需要提前预约门票"),
            candidate("d", "d3", "D", "晚餐安排北京烤鸭"),
        ]
        result = deduplicate_candidates(values, similarity_threshold=0.86, by_document=True)
        self.assertEqual(["a", "d"], [item["chunk_id"] for item in result])

    def test_complete_pipeline_and_trace(self):
        store = FakeStore()
        retriever = TravelKnowledgeRetriever(
            self.config,
            store=store,
            reranker=FakeReranker(),
            query_processor=FakeQueryProcessor(),
        )
        result = retriever.search_with_trace("帮我规划北京亲子游", top_k=5)
        self.assertEqual("北京 三日 亲子 行程 故宫 地铁", result.trace.rewritten_query)
        self.assertEqual({"city": "北京", "topic": "family"}, result.trace.filters)
        self.assertEqual(3, result.trace.dense_count)
        self.assertEqual(3, result.trace.sparse_count)
        self.assertEqual(4, result.trace.fused_count)
        self.assertEqual(4, result.trace.reranked_count)
        self.assertEqual(4, result.trace.final_count)
        self.assertEqual(20, store.calls[1][3])
        self.assertEqual(20, store.calls[2][3])
        self.assertIsNotNone(result.evidence[0].rrf_score)
        self.assertIsNotNone(result.evidence[0].rerank_score)


if __name__ == "__main__":
    unittest.main()
