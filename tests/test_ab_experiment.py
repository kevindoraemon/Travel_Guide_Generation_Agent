import csv
import json

import pytest

from scripts.ab_test_ttd_dr import (
    BenchmarkResult,
    OPENAI_DR_REPORT_PROMPT,
    PairwiseVerdict,
    RESEARCH_PLAN_PROMPT,
    _quality_dict,
    ab_metrics,
    human_metrics,
    pairwise_metrics,
    prepare_blind_review,
    summarize_records,
)
from travel_planner.prompts import DRAFT_ITINERARY_PROMPT


def _record(task_id: str, openai_score: float, ttd_score: float, pairwise=None) -> dict:
    common = {"route_conflict": False, "budget_omission": False, "fact_gap": False}
    record = {
        "task_id": task_id,
        "query": "测试任务",
        "initial": {"trip_brief": "测试简报", "latency_seconds": 1},
        "variants": {
            "openai_dr": {"report": "openai", "evidence": ["a"], "quality": {"overall": openai_score}, "defects": common, "gate_passed": True, "revision_rounds": 0, "search_rounds": 2, "scout_calls": 4, "end_to_end_latency_seconds": 10, "completion_status": "passed", "stop_reason": "plan_execute_synthesize"},
            "ttd_dr": {"report": "ttd", "evidence": ["b"], "initial_draft": "skeleton", "initial_quality": {"overall": 50.0}, "quality": {"overall": ttd_score}, "defects": common, "gate_passed": True, "revision_rounds": 2, "search_rounds": 2, "scout_calls": 4, "end_to_end_latency_seconds": 20, "completion_status": "passed", "stop_reason": "quality_gate_passed", "quality_trajectory": [{"iteration": 1, "score": 5.0}, {"iteration": 2, "score": 7.5}]},
        },
    }
    if pairwise:
        record["pairwise"] = pairwise
    return record


def test_summary_only_reports_initial_to_final_gain_for_ttd():
    records = [_record("a", 60.0, 70.0), _record("b", 70.0, 80.0)]
    records[0]["variants"]["ttd_dr"].update(stop_reason="revision_budget_reached", completion_status="degraded")
    summary = summarize_records(records)
    assert "avg_score_gain" not in summary["openai_dr"]
    assert summary["ttd_dr"]["avg_score_gain"] == 25.0
    assert summary["ttd_dr"]["avg_revision_rounds"] == 2
    assert summary["ttd_dr"]["max_iteration_reached_rate"] == 0.5
    assert summary["ttd_dr"]["degradation_rate_after_max_iterations"] == 1.0
    assert summary["ttd_dr"]["coordinator_declared_complete_rate"] == 0.5
    assert summary["ttd_dr"]["avg_internal_refine_gain"] == 2.5
    assert summary["openai_dr"]["avg_search_rounds"] == 2
    assert summary["openai_dr"]["avg_scout_calls"] == 4
    comparison = ab_metrics(records)
    assert comparison["ttd_minus_openai_final_score"] == 10
    assert comparison["ttd_absolute_score_win_rate"] == 1
    assert comparison["equal_actual_scout_calls_rate"] == 1
    assert comparison["ttd_to_openai_latency_ratio"] == 2


def test_benchmark_score_is_a_100_point_sum():
    result = BenchmarkResult(
        feasibility_score=28,
        budget_score=20,
        factuality_score=18,
        usability_score=12,
        route_conflict=False,
        route_conflict_reason="none",
        budget_omission=False,
        budget_omission_reason="none",
        fact_gap=True,
        fact_gap_reason="one unsupported price",
        critical_issues=["verify price"],
        direct_usable=False,
        reason="needs verification",
    )
    quality = _quality_dict(result)
    assert quality["overall"] == 71
    assert quality["factuality"] == 12
    assert quality["usability"] == 11


def test_initial_draft_is_an_unresearched_skeleton():
    assert "而不是最终路书" in DRAFT_ITINERARY_PROMPT
    assert "不得编造 URL" in DRAFT_ITINERARY_PROMPT


def test_openai_dr_is_plan_execute_without_a_draft():
    assert "只制定调研计划，不写路书" in RESEARCH_PLAN_PROMPT
    assert "没有草稿可参考" in OPENAI_DR_REPORT_PROMPT


def test_pairwise_metrics_counts_wins_and_position_disagreement():
    records = [
        _record("a", 80, 80, pairwise={"preferred": "ttd_dr", "consistent": True, "passes": [{"confidence": 0.8}, {"confidence": 0.9}]}),
        _record("b", 80, 80, pairwise={"preferred": "openai_dr", "consistent": True, "passes": [{"confidence": 0.7}]}),
        _record("c", 80, 80, pairwise={"preferred": "inconsistent_tie", "consistent": False, "passes": [{"confidence": 0.5}]}),
    ]
    metrics = pairwise_metrics(records)
    assert metrics["pairs"] == 3
    assert metrics["decided_pairs"] == 2
    assert metrics["position_flip_disagreement_rate"] == 1 / 3
    assert metrics["ttd_dr_win_rate_tie_adjusted"] == (1 + 0.5) / 3
    assert metrics["openai_dr_win_rate_tie_adjusted"] == (1 + 0.5) / 3
    assert metrics["avg_confidence"] == pytest.approx((0.8 + 0.9 + 0.7 + 0.5) / 4)


def test_blind_review_and_human_metrics(tmp_path):
    records = [_record("a", 60.0, 70.0)]
    prepare_blind_review(records, tmp_path)
    manifest = json.loads((tmp_path / "blind_manifest.json").read_text(encoding="utf-8"))
    review_path = tmp_path / "human_review.csv"
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pair_id = rows[0]["pair_id"]
    ttd_label = next(label for label, variant in manifest[pair_id].items() if variant == "ttd_dr")
    rows[0]["preferred"] = ttd_label
    rows[0][f"usable_{ttd_label.lower()}"] = "yes"
    with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = human_metrics(tmp_path)
    assert metrics["ttd_dr_preference_win_rate_tie_adjusted"] == 1.0
    assert metrics["ttd_dr_direct_usable_rate"] == 1.0


def test_pairwise_verdict_schema_requires_labels():
    verdict = PairwiseVerdict(preferred="tie", confidence=0.5, fact_check="equal", reasons="none")
    assert verdict.preferred == "tie"
