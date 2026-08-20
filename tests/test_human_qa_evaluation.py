import unittest
from pathlib import Path

from travel_planner.rag.human_evaluation import (
    HumanAnswerAnnotation,
    TravelQAEvalCase,
    aggregate_annotations,
    load_qa_cases,
    quadratic_weighted_kappa,
)


class HumanQAEvaluationTests(unittest.TestCase):
    def test_pilot_dataset_is_valid_but_not_claimed_as_human_labeled(self):
        cases = load_qa_cases(Path("data/eval/travel_qa_eval_pilot.jsonl"))
        self.assertEqual(10, len(cases))
        self.assertTrue(all(case.answerable for case in cases))
        self.assertTrue(all(case.human_label_status == "draft_needs_review" for case in cases))
        self.assertTrue(all(case.to_retrieval_case().relevant_keys() for case in cases))

    def test_weighted_kappa_is_one_for_identical_labels(self):
        self.assertEqual(1.0, quadratic_weighted_kappa([(1, 1), (3, 3), (5, 5)]))

    def test_double_annotation_aggregation(self):
        common = {
            "case_id": "case-1",
            "blind_answer_id": "blind-1",
            "completeness": 4,
            "faithfulness": 5,
            "citation_correctness": 4,
            "practicality": 4,
            "constraint_satisfaction": 5,
            "fluency": 4,
            "critical_error": False,
        }
        annotations = [
            HumanAnswerAnnotation(
                assignment_id="a-1", annotator_id="annotator-a", **common
            ),
            HumanAnswerAnnotation(
                assignment_id="b-1", annotator_id="annotator-b", **common
            ),
        ]
        report = aggregate_annotations(annotations, {"blind-1": "hybrid_rag"})
        self.assertEqual(1, report["systems"]["hybrid_rag"]["answers"])
        self.assertEqual(0.0, report["systems"]["hybrid_rag"]["critical_error_rate"])
        self.assertTrue(all(value == 1.0 for value in report["inter_annotator_agreement"].values()))

    def test_aggregation_rejects_single_annotator(self):
        annotation = HumanAnswerAnnotation(
            assignment_id="a-1",
            annotator_id="annotator-a",
            case_id="case-1",
            blind_answer_id="blind-1",
            completeness=4,
            faithfulness=4,
            citation_correctness=4,
            practicality=4,
            constraint_satisfaction=4,
            fluency=4,
            critical_error=False,
        )
        with self.assertRaises(ValueError):
            aggregate_annotations([annotation], {"blind-1": "hybrid_rag"})

    def test_unanswerable_case_does_not_require_positive_relevance(self):
        case = TravelQAEvalCase(
            case_id="no-answer",
            query="语料中不存在的问题",
            category="unanswerable",
            answerable=False,
            graded_relevance={"chunk:candidate": 0},
            source_snapshot="snapshot-v1",
        )
        self.assertFalse(case.answerable)


if __name__ == "__main__":
    unittest.main()
