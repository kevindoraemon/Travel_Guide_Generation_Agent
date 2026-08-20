import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from travel_planner.rag.evaluation import RetrievalEvalCase, evaluate_retriever
from travel_planner.rag.schemas import RetrievalResult, RetrievalTrace, RetrievedChunk
from travel_planner.utils import gather_with_concurrency
from scripts.evaluate_rag import load_cases


class FakeRetriever:
    def search_with_trace(self, query, **kwargs):
        evidence = [
            RetrievedChunk(
                chunk_id="c2",
                document_id="d2",
                title="second",
                content="second result",
            ),
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                title="first",
                content="first relevant result",
            ),
        ]
        return RetrievalResult(
            evidence=evidence,
            trace=RetrievalTrace(
                original_query=query,
                rewritten_query=query,
                final_count=2,
            ),
        )


class EvaluationTest(unittest.TestCase):
    def test_metrics_are_calculated_from_explicit_labels(self):
        report = evaluate_retriever(
            FakeRetriever(),
            [
                RetrievalEvalCase(
                    case_id="hit-at-two",
                    query="query one",
                    relevant_document_ids=["d1"],
                ),
                RetrievalEvalCase(
                    case_id="miss",
                    query="query two",
                    relevant_chunk_ids=["c3"],
                ),
            ],
            k=5,
        )
        self.assertEqual(0.5, report.recall_at_k)
        self.assertEqual(0.25, report.mrr)
        self.assertEqual(0.5, report.hit_rate_at_k)
        self.assertEqual(2, report.results[0].first_relevant_rank)

    def test_unlabeled_cases_are_rejected(self):
        with self.assertRaises(ValidationError):
            RetrievalEvalCase(case_id="invalid", query="no labels")

    def test_graded_relevance_produces_ndcg(self):
        report = evaluate_retriever(
            FakeRetriever(),
            [
                RetrievalEvalCase(
                    case_id="graded",
                    query="graded query",
                    graded_relevance={"chunk:c1": 3, "chunk:c2": 1},
                )
            ],
            k=2,
        )
        self.assertEqual(1.0, report.recall_at_k)
        self.assertGreater(report.ndcg_at_k, 0.0)
        self.assertLess(report.ndcg_at_k, 1.0)

    def test_unanswerable_qa_cases_are_excluded_from_retrieval_metrics(self):
        rows = [
            {"case_id": "answerable", "query": "q1", "relevant_chunk_ids": ["c1"]},
            {"case_id": "no-answer", "query": "q2", "answerable": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            cases, dataset_hash, excluded = load_cases(path)
        self.assertEqual(1, len(cases))
        self.assertEqual(1, excluded)
        self.assertEqual(64, len(dataset_hash))


class ConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_hard_concurrency_limit(self):
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker(value):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return value

        factories = [lambda value=value: worker(value) for value in range(9)]
        results = await gather_with_concurrency(3, factories)
        self.assertEqual(list(range(9)), results)
        self.assertEqual(3, peak)


if __name__ == "__main__":
    unittest.main()
