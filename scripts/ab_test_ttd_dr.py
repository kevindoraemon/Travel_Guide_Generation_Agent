"""Run a resumable OpenAI-DeepResearch-style vs TTD-DR itinerary experiment.

Variant A (openai_dr):  plan -> agentic two-pass browsing (LLM decides follow-up
                        queries from round-1 gaps) -> one-shot final report.
                        No draft skeleton, no iterative refinement loop.
Variant B (ttd_dr):     draft skeleton -> evidence-driven denoising loop
                        (coordinator + refine + evaluator + critic) -> final.

Scoring: held-out absolute rubric per report + LLM pairwise blind comparison
(two shuffled passes, position-bias checked) + human blind review CSV.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal


EXPERIMENT_WRITER_TOKENS = 1800
MAX_FINDING_CHARS = 3500
BENCHMARK_PASS_SCORE = 60
FOLLOW_UP_CONCURRENCY = 1
DEFAULT_SCOUT_BUDGET = 4
EXPERIMENT_SEARCH_RESULTS = 1
VARIANTS = ("openai_dr", "ttd_dr")
RUBRIC = "evidence_100_v1"


class BenchmarkResult(BaseModel):
    feasibility_score: int = Field(ge=0, le=35)
    budget_score: int = Field(ge=0, le=25)
    factuality_score: int = Field(ge=0, le=25)
    usability_score: int = Field(ge=0, le=15)
    route_conflict: bool = Field(description="Whether the itinerary has an obvious route, timing, closure, or transport conflict")
    route_conflict_reason: str
    budget_omission: bool = Field(description="Whether required budget categories or the total budget are missing")
    budget_omission_reason: str
    fact_gap: bool = Field(description="Whether important prices, opening hours, booking rules, or transport facts are absent or unverifiable")
    fact_gap_reason: str
    critical_issues: list[str]
    direct_usable: bool
    reason: str


class ResearchPlan(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3, description="Three non-overlapping executable research tasks")


class FollowUpQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=1, description="The single highest-value evidence gap")


class PairwiseVerdict(BaseModel):
    preferred: Literal["X", "Y", "tie"]
    confidence: float = Field(ge=0.0, le=1.0)
    fact_check: str = Field(description="Which report's key facts (prices, hours, booking rules) are better supported by the evidence pool")
    reasons: str


OPENAI_DR_REPORT_PROMPT = """你刚刚以自主浏览的方式完成了对一个出行需求的多轮网络调研（已规划搜索、阅读网页、按需补搜）。
以下是全部调研笔记。请基于这些笔记，针对出行需求一次性撰写最终旅游路书。没有草稿可参考，也没有后续修订机会。

<出行需求简报>
{trip_brief}
</出行需求简报>

今天的日期是 {date}。

以下是您搜集到的旅游情报：
<情报>
{findings}
</情报>

请确保答案与用户输入的语言一致。

该路书应：
1. 结构清晰，标题使用规范（# 标题，## 章节，### 子章节）
2. 按天组织行程，每天标注上午/下午/晚上安排、交通方式、预计用时与人均费用
3. 只使用情报中出现过的具体事实（门票价格、开放时间、班次、人均消费、地址等），不得编造
4. 在事实陈述后使用 [编号] 引用资料来源
5. 在路书末尾添加"参考资料"部分，列出所有引用的链接
6. 确保各天行程逻辑连贯（避免回头路、时间冲突、预算超支、景点闭馆）
7. 涉及预算时提供详细费用明细表；多方案对比时提供汇总表
8. 各章节以段落形式撰写，不要以要点罗列事实

<引用规则>
- 为每个唯一的 URL 分配唯一引用编号，按顺序编号（1、2、3……），中间不能有空隙。
- URL 只出现在"### 来源列表"中，正文仅使用 [编号]。
</引用规则>

切勿在路书中提及自己是作者或说明工作过程。只输出路书本体。
"""


RESEARCH_PLAN_PROMPT = """你是深度调研规划器。请先分析需求，再输出恰好3个彼此不重叠、可独立执行的网络调研任务。
三个任务合起来必须覆盖：景点开放/预约规则，路线交通/时间冲突，住宿餐饮/完整预算，以及用户的特殊约束。
每个任务都要包含目的地、日期或季节、人数和需要核验的关键事实，优先要求官方来源。此阶段只制定调研计划，不写路书。

<出行需求>
{trip_brief}
</出行需求>
{draft_context}
请使用与出行需求相同的语言。"""


FOLLOW_UP_QUERY_PROMPT = """你是深度调研 Agent 的浏览规划器。已有第一轮调研笔记如下：

<出行需求>
{trip_brief}
</出行需求>

<已有情报摘要>
{digest}
</已有情报摘要>

请判断当前情报中哪些关键事实缺口会直接影响路书可执行性（如价格、预约规则、末班交通、闭馆日、天气风险），
只给出1条价值最高、与已有情报不重复的补充搜索查询。查询必须具体、可直接搜索。

用与出行需求相同的语言输出查询。"""


PAIRWISE_PROMPT = """你是A/B测试的留出裁判，不参与任何一份路书的生成。两份候选路书（X 与 Y）针对同一出行需求，
由两种不同的研究方法独立产出。请基于需求与证据池，判定哪一份更好。

<出行需求>
{trip_brief}
</出行需求>

<候选路书X>
{report_x}
</候选路书X>

<候选路书Y>
{report_y}
</候选路书Y>

<证据池（用于核查事实，两份路书共用）>
{evidence}
</证据池>

判定规则：
1. 先做事实核查：对照证据池，逐项检查两份路书中的价格、开放时间、预约规则、班次；证据不支持的关键事实视为缺陷。
2. 再比可执行性：路线冲突、闭馆冲突、末班交通不可达、预算超支或遗漏。
3. 再比需求覆盖与直接可用性：是否覆盖全部硬性需求（同行人、预算、节奏、特殊约束）。
4. 篇幅长、表格多、语言流畅本身不加分；证据支持才加分。
5. 若两份质量实质相当，判 tie；不要为了避免平局而强行选边。

输出 preferred（X/Y/tie）、confidence（0-1）、fact_check 与 reasons。"""


def load_tasks(path: Path, limit: int = 0) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tasks[:limit] if limit else tasks


def retry_sync_evaluator(model, schema, messages):
    """Held-out judge call wrapped in retry for provider rate limits (429 etc.)."""
    import time as _time
    from travel_planner.llm import safe_structured_output as _sso

    delay = 8.0
    for attempt in range(5):
        try:
            return _sso(model, schema, messages)
        except Exception:
            if attempt == 4:
                raise
            _time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _metric_field(metric: dict | object, key: str):
    if isinstance(metric, dict):
        return metric.get(key)
    return getattr(metric, key, None)


def _quality_dict(result: BenchmarkResult) -> dict:
    raw = {
        "feasibility": result.feasibility_score,
        "budget": result.budget_score,
        "factuality": result.factuality_score,
        "usability": result.usability_score,
    }
    scores = dict(raw)
    adjustments: list[str] = []
    for defect, dimension, ceiling in (
        (result.route_conflict, "feasibility", 20),
        (result.budget_omission, "budget", 10),
        (result.fact_gap, "factuality", 12),
        (not result.direct_usable, "usability", 11),
    ):
        if defect and scores[dimension] > ceiling:
            adjustments.append(f"{dimension}: {scores[dimension]} -> {ceiling}")
            scores[dimension] = ceiling
    return {
        "rubric": RUBRIC,
        **scores,
        "overall": sum(scores.values()),
        "raw_scores": raw,
        "score_adjustments": adjustments,
        "direct_usable": result.direct_usable,
        "critical_issues": result.critical_issues,
        "reason": result.reason,
    }


def _nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_records(records: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [record["variants"][variant] for record in records if variant in record.get("variants", {})]
        if not rows:
            continue
        final_scores = [row["quality"]["overall"] for row in rows]
        maxed = [row for row in rows if row.get("stop_reason") == "revision_budget_reached"]
        metrics: dict = {
            "cases": len(rows),
            "quality_scale": 100,
            "final_avg_score": statistics.fmean(final_scores),
            "quality_gate_pass_rate": sum(row["gate_passed"] for row in rows) / len(rows),
            "route_conflict_rate": sum(row["defects"]["route_conflict"] for row in rows) / len(rows),
            "budget_omission_rate": sum(row["defects"]["budget_omission"] for row in rows) / len(rows),
            "fact_gap_rate": sum(row["defects"]["fact_gap"] for row in rows) / len(rows),
            "avg_revision_rounds": statistics.fmean(row["revision_rounds"] for row in rows),
            "avg_search_rounds": statistics.fmean(row.get("search_rounds", 1) for row in rows),
            "avg_scout_calls": statistics.fmean(row.get("scout_calls", 0) for row in rows),
            "p95_end_to_end_latency_seconds": _nearest_rank_p95([row["end_to_end_latency_seconds"] for row in rows]),
        }
        if variant == "ttd_dr":
            initial = [row["initial_quality"]["overall"] for row in rows]
            metrics["initial_avg_score"] = statistics.fmean(initial)
            metrics["avg_score_gain"] = statistics.fmean(
                final - start for final, start in zip(final_scores, initial)
            )
            metrics["max_iteration_reached_rate"] = len(maxed) / len(rows)
            metrics["degradation_rate_after_max_iterations"] = (
                sum(row.get("completion_status") == "degraded" for row in maxed) / len(maxed) if maxed else None
            )
            metrics["coordinator_declared_complete_rate"] = (
                sum(row.get("stop_reason") == "quality_gate_passed" for row in rows) / len(rows)
            )
            gains: list[float] = []
            for row in rows:
                trajectory = [point["score"] for point in row.get("quality_trajectory", [])]
                if len(trajectory) >= 2:
                    gains.append(trajectory[-1] - trajectory[0])
            metrics["avg_internal_refine_gain"] = statistics.fmean(gains) if gains else None
        summary[variant] = metrics
    return summary


async def pairwise_judge(record: dict, passes: int) -> dict:
    """Two shuffled passes with order flipped between passes (position-bias check)."""
    brief = record["initial"]["trip_brief"]
    evidence_parts: list[str] = []
    for variant in VARIANTS:
        evidence_parts.extend(record["variants"].get(variant, {}).get("evidence", []))
    evidence = "\n\n".join(part[:MAX_FINDING_CHARS] for part in evidence_parts)

    from travel_planner.llm import get_chat_model, safe_structured_output
    from travel_planner.utils import gather_with_concurrency, retry_async

    judge_model = get_chat_model("benchmark_evaluator")
    outcomes: list[dict] = []
    for pass_index in range(passes):
        digest = hashlib.sha256(record["task_id"].encode() + f"pw{pass_index}".encode()).hexdigest()
        ttd_is_x = (int(digest, 16) % 2 == 0) ^ (pass_index % 2 == 1)  # flip order each pass
        mapping = {"X": "ttd_dr" if ttd_is_x else "openai_dr", "Y": "openai_dr" if ttd_is_x else "ttd_dr"}
        prompt = PAIRWISE_PROMPT.format(
            trip_brief=brief,
            report_x=record["variants"][mapping["X"]]["report"],
            report_y=record["variants"][mapping["Y"]]["report"],
            evidence=evidence,
        )
        verdict = await retry_async(lambda p=prompt: asyncio.to_thread(
            safe_structured_output, judge_model, PairwiseVerdict, [HumanMessage(content=p)]
        ))
        outcomes.append({
            "pass": pass_index,
            "x_is": mapping["X"],
            "preferred_label": verdict.preferred,
            "preferred_variant": mapping.get(verdict.preferred) if verdict.preferred != "tie" else "tie",
            "confidence": verdict.confidence,
            "fact_check": verdict.fact_check,
            "reasons": verdict.reasons,
        })
    votes = [outcome["preferred_variant"] for outcome in outcomes]
    consistent = len(set(votes)) == 1
    if consistent:
        preferred = votes[0]
    else:
        openai_votes = votes.count("openai_dr")
        ttd_votes = votes.count("ttd_dr")
        preferred = "openai_dr" if openai_votes > ttd_votes else "ttd_dr" if ttd_votes > openai_votes else "inconsistent_tie"
    return {"passes": outcomes, "votes": votes, "consistent": consistent, "preferred": preferred}


def pairwise_metrics(records: list[dict]) -> dict | None:
    judged = [record["pairwise"] for record in records if record.get("pairwise")]
    if not judged:
        return None
    total = len(judged)
    wins = {"openai_dr": 0, "ttd_dr": 0}
    ties = 0
    inconsistent = 0
    for result in judged:
        if result["preferred"] == "openai_dr":
            wins["openai_dr"] += 1
        elif result["preferred"] == "ttd_dr":
            wins["ttd_dr"] += 1
        elif result["preferred"] == "inconsistent_tie":
            ties += 1
        else:
            ties += 1
        if not result["consistent"]:
            inconsistent += 1
    confidences = [outcome["confidence"] for result in judged for outcome in result["passes"]]
    decided = total - ties
    return {
        "pairs": total,
        "position_flip_disagreement_rate": inconsistent / total,
        "openai_dr_win_rate_tie_adjusted": (wins["openai_dr"] + 0.5 * ties) / total,
        "ttd_dr_win_rate_tie_adjusted": (wins["ttd_dr"] + 0.5 * ties) / total,
        "tie_rate": ties / total,
        "decided_pairs": decided,
        "avg_confidence": statistics.fmean(confidences) if confidences else None,
    }


def ab_metrics(records: list[dict]) -> dict | None:
    paired = [record for record in records if set(VARIANTS) <= set(record.get("variants", {}))]
    if not paired:
        return None
    score_deltas = [
        record["variants"]["ttd_dr"]["quality"]["overall"]
        - record["variants"]["openai_dr"]["quality"]["overall"]
        for record in paired
    ]
    openai_latency = statistics.fmean(
        record["variants"]["openai_dr"]["end_to_end_latency_seconds"] for record in paired
    )
    ttd_latency = statistics.fmean(
        record["variants"]["ttd_dr"]["end_to_end_latency_seconds"] for record in paired
    )
    return {
        "ttd_minus_openai_final_score": statistics.fmean(score_deltas),
        "ttd_absolute_score_win_rate": sum(delta > 0 for delta in score_deltas) / len(paired),
        "equal_actual_scout_calls_rate": sum(
            record["variants"]["ttd_dr"].get("scout_calls")
            == record["variants"]["openai_dr"].get("scout_calls")
            for record in paired
        ) / len(paired),
        "ttd_to_openai_latency_ratio": ttd_latency / openai_latency if openai_latency else None,
    }


def prepare_blind_review(records: list[dict], output_dir: Path) -> None:
    blind_dir = output_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "human_review.csv"
    existing: dict[str, dict] = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as stream:
            existing = {row["pair_id"]: row for row in csv.DictReader(stream)}
    manifest: dict[str, dict] = {}
    rows: list[dict] = []
    for record in records:
        task_id = record["task_id"]
        pair_id = hashlib.sha256(task_id.encode()).hexdigest()[:10]
        ttd_is_a = int(pair_id, 16) % 2 == 0
        mapping = {"A": "ttd_dr" if ttd_is_a else "openai_dr", "B": "openai_dr" if ttd_is_a else "ttd_dr"}
        manifest[pair_id] = mapping
        for label, variant in mapping.items():
            (blind_dir / f"{pair_id}_{label}.md").write_text(
                f"# 任务\n\n{record['query']}\n\n# 路书方案 {label}\n\n{record['variants'][variant]['report']}",
                encoding="utf-8",
            )
        row = {
            "pair_id": pair_id,
            "task": record["query"],
            "response_a": str(blind_dir / f"{pair_id}_A.md"),
            "response_b": str(blind_dir / f"{pair_id}_B.md"),
            "preferred": "",
            "usable_a": "",
            "usable_b": "",
            "notes": "",
        }
        for field in ("preferred", "usable_a", "usable_b", "notes"):
            row[field] = existing.get(pair_id, {}).get(field, "")
        rows.append(row)
    _write_json(output_dir / "blind_manifest.json", manifest)
    with review_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["pair_id"])
        writer.writeheader()
        writer.writerows(rows)


def human_metrics(output_dir: Path) -> dict | None:
    review_path = output_dir / "human_review.csv"
    manifest_path = output_dir / "blind_manifest.json"
    if not review_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    preference_rows = [row for row in all_rows if row.get("preferred", "").strip().upper() in {"A", "B", "TIE"}]
    if not preference_rows and not any(row.get("usable_a") or row.get("usable_b") for row in all_rows):
        return None
    wins = {"openai_dr": 0, "ttd_dr": 0}
    ties = 0
    usable = {"openai_dr": 0, "ttd_dr": 0}
    usable_labeled = {"openai_dr": 0, "ttd_dr": 0}
    for row in all_rows:
        mapping = manifest[row["pair_id"]]
        preference = row["preferred"].strip().upper()
        if preference == "TIE":
            ties += 1
        elif preference in {"A", "B"}:
            wins[mapping[preference]] += 1
        for label in ("A", "B"):
            answer = row.get(f"usable_{label.lower()}", "").strip().lower()
            if answer in {"yes", "y", "是", "1", "no", "n", "否", "0"}:
                usable_labeled[mapping[label]] += 1
            if answer in {"yes", "y", "是", "1"}:
                usable[mapping[label]] += 1
    total = len(preference_rows)
    return {
        "labeled_pairs": total,
        "ttd_dr_preference_win_rate_tie_adjusted": (wins["ttd_dr"] + 0.5 * ties) / total if total else None,
        "openai_dr_preference_win_rate_tie_adjusted": (wins["openai_dr"] + 0.5 * ties) / total if total else None,
        "ttd_dr_direct_usable_rate": usable["ttd_dr"] / usable_labeled["ttd_dr"] if usable_labeled["ttd_dr"] else None,
        "openai_dr_direct_usable_rate": usable["openai_dr"] / usable_labeled["openai_dr"] if usable_labeled["openai_dr"] else None,
    }


def write_summary(records: list[dict], output_dir: Path, config: dict) -> dict:
    result = {
        "config": config,
        "automated": summarize_records(records),
        "comparison": ab_metrics(records),
        "pairwise": pairwise_metrics(records),
        "human": human_metrics(output_dir),
    }
    _write_json(output_dir / "summary.json", result)
    lines = [
        "# OpenAI-DR vs TTD-DR A/B 测试",
        "",
        "## 实验配置",
        "",
        *[f"- {key}: {value}" for key, value in config.items()],
        "",
    ]
    for name, metrics in result["automated"].items():
        lines.extend([f"## {name}", "", *[f"- {key}: {value}" for key, value in metrics.items()], ""])
    lines.extend(["## 直接对比", "", json.dumps(result["comparison"], ensure_ascii=False, indent=2), ""])
    lines.extend(["## LLM 成对盲评（两轮位置对调）", "", json.dumps(result["pairwise"], ensure_ascii=False, indent=2) if result["pairwise"] else "暂无。", ""])
    lines.extend(["## 人工盲测", "", json.dumps(result["human"], ensure_ascii=False) if result["human"] else "等待填写 human_review.csv。", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


async def _prepare_brief(task: dict) -> tuple[str, float]:
    from travel_planner.agents.briefing_agent import plan_trip_brief

    started = time.perf_counter()
    update = await asyncio.to_thread(plan_trip_brief, {"messages": [HumanMessage(content=task["query"])]})
    return update["trip_brief"], time.perf_counter() - started


async def _research_plan(trip_brief: str, fallback: list[str], draft: str = "") -> list[str]:
    from travel_planner.llm import get_chat_model

    draft_context = f"\n<当前待去噪骨架>\n{draft}\n</当前待去噪骨架>" if draft else ""
    try:
        result = await asyncio.to_thread(
            retry_sync_evaluator,
            get_chat_model("coordinator"),
            ResearchPlan,
            [HumanMessage(content=RESEARCH_PLAN_PROMPT.format(
                trip_brief=trip_brief, draft_context=draft_context
            ))],
        )
        queries = list(dict.fromkeys(query.strip() for query in result.queries if query.strip()))
        if len(queries) == 3:
            return queries
    except Exception:
        pass
    return fallback[:3]


async def _run_scouts(topics: list[str]) -> list[str]:
    import travel_planner.agents.scout_agent as scout_module
    import travel_planner.tools.tool as tool_module
    from travel_planner.utils import gather_with_concurrency, retry_async

    scout_module.MAX_TOOL_CALL_ITERATIONS = 2
    _, _, search_defaults = tool_module._ensure_search_runtime()
    search_defaults["max_results"] = EXPERIMENT_SEARCH_RESULTS

    factories = [
        lambda topic=topic: retry_async(lambda t=topic: scout_module.scout_agent.ainvoke({
            "scout_messages": [HumanMessage(content=t)], "travel_topic": t
        }))
        for topic in topics
    ]
    results = await gather_with_concurrency(FOLLOW_UP_CONCURRENCY, factories)
    return [str(result.get("compressed_intel", ""))[:MAX_FINDING_CHARS] for result in results]


async def _follow_up_query(trip_brief: str, findings: list[str], context: str = "") -> str:
    from travel_planner.llm import get_chat_model

    digest = "\n\n".join([*findings[-3:], context])[-10000:]
    result = await asyncio.to_thread(
        retry_sync_evaluator,
        get_chat_model("research_planner"),
        FollowUpQueries,
        [HumanMessage(content=FOLLOW_UP_QUERY_PROMPT.format(trip_brief=trip_brief, digest=digest))],
    )
    return result.queries[0]


async def _evaluate_report(trip_brief: str, report: str, evidence: list[str]) -> tuple[dict, dict, list[str]]:
    from travel_planner.agents.coordinator import validate_itinerary_contract
    from travel_planner.llm import get_chat_model

    evidence_text = "\n\n".join(item[:MAX_FINDING_CHARS] for item in evidence)
    prompt = f"""你是A/B测试的留出裁判，不参与路书生成或TTD-DR终止决策。根据需求、候选路书和外部检索证据进行严格的100分评估。篇幅长、表格多或带URL本身不加分。
<需求>{trip_brief}</需求>
<路书>{report}</路书>
<外部证据>{evidence_text}</外部证据>

从0分开始按以下量表给分：
- 可执行性0-35：0-10不可执行；11-20有明显冲突；21-27修订后可用；28-31可直接执行；32-34经充分核验；35近乎无缺陷。出现明显路线、闭馆、时间或交通冲突时不得超过20分。
- 预算0-25：主要类别或总计缺失不得超过10分；有完整分项、总计、余量和超支预案才可达到20分以上。
- 事实与证据0-25：只有外部证据明确支持的价格、开放时间、预约、班次和政策才能得分；路书自己的URL或引用编号不算证据。关键事实不可验证时不得超过12分。
- 直接使用性0-15：需求覆盖、备选、风险提示清晰且用户无需补充研究才能执行，才可达到12分以上。

校准规则：60分表示“最低可用且仍需修改”，70分表示“基本可直接使用”，80分表示“质量较强”，90分只用于关键事实逐项有证据且几乎无缺陷的方案。必须指出具体扣分证据；不要因为语言流畅而宽松打分。
路线冲突包括时间重叠、明显回头路、闭馆日冲突、末班车或跨区耗时不现实；预算遗漏指缺少主要类别或合计；事实缺失指关键事实无外部证据支持且没有明确标为待确认。"""
    result = await asyncio.to_thread(
        retry_sync_evaluator,
        get_chat_model("benchmark_evaluator"),
        BenchmarkResult,
        [HumanMessage(content=prompt)],
    )
    hard_issues = validate_itinerary_contract(trip_brief, report)
    defects = {key: getattr(result, key) for key in (
        "route_conflict", "route_conflict_reason", "budget_omission", "budget_omission_reason", "fact_gap", "fact_gap_reason"
    )}
    return _quality_dict(result), defects, hard_issues


async def _openai_dr(
    trip_brief: str,
    fallback_topics: list[str],
    follow_up_rounds: int,
    scout_budget: int,
) -> dict:
    """OpenAI DR control: plan -> execute/pivot -> one-shot synthesis, with no draft."""
    from travel_planner.llm import get_chat_model
    from travel_planner.utils import get_today_str, retry_async

    started = time.perf_counter()
    plan = await _research_plan(trip_brief, fallback_topics)
    initial_topics = plan[:min(len(plan), scout_budget)]
    findings = await _run_scouts(initial_topics)
    follow_up_queries: list[str] = []
    for _ in range(follow_up_rounds):
        if len(findings) >= scout_budget:
            break
        query = await _follow_up_query(trip_brief, findings)
        follow_up_queries.append(query)
        findings.extend(await _run_scouts([query]))
    prompt = OPENAI_DR_REPORT_PROMPT.format(
        trip_brief=trip_brief,
        findings="\n\n".join(findings),
        date=get_today_str(),
    )
    writer = get_chat_model("writer", max_tokens=EXPERIMENT_WRITER_TOKENS)
    response = await retry_async(lambda: writer.ainvoke([HumanMessage(content=prompt)]))
    return {
        "report": str(response.content),
        "research_plan": plan,
        "follow_up_queries": follow_up_queries,
        "evidence": findings,
        "search_rounds": 1 + len(follow_up_queries),
        "scout_calls": len(findings),
        "latency_seconds": time.perf_counter() - started,
    }


async def _ttd_dr(
    trip_brief: str,
    fallback_topics: list[str],
    max_iterations: int,
    scout_budget: int,
) -> dict:
    """TTD-DR: draft first, then evidence-driven evaluate/critic/refine rounds."""
    import importlib

    from travel_planner.agents.briefing_agent import write_draft_itinerary
    from travel_planner.agents.coordinator import validate_itinerary_contract
    from travel_planner.agents.critic_agent import critic_node
    from travel_planner.agents.evaluator_agent import evaluate_itinerary_quality
    from travel_planner.llm import get_chat_model

    started = time.perf_counter()
    draft = (await asyncio.to_thread(write_draft_itinerary, {
        "messages": [HumanMessage(content=trip_brief)], "trip_brief": trip_brief
    }))["draft_itinerary"]
    initial_draft = draft
    plan = await _research_plan(trip_brief, fallback_topics, draft)
    findings = await _run_scouts(plan[:min(len(plan), scout_budget)])
    trajectory: list[dict] = []
    follow_up_queries: list[str] = []
    critiques: list[str] = []
    tool_module = importlib.import_module("travel_planner.tools.tool")
    tool_module.writer_model = get_chat_model("writer", max_tokens=EXPERIMENT_WRITER_TOKENS)

    for iteration in range(1, max_iterations + 1):
        draft = str(await tool_module._refine_itinerary_tool.ainvoke({
            "trip_brief": trip_brief,
            "findings": "\n\n".join(findings),
            "draft_itinerary": draft,
        }))
        evaluation = await asyncio.to_thread(evaluate_itinerary_quality, trip_brief, draft)
        score = statistics.fmean([
            evaluation.feasibility_score,
            evaluation.budget_score,
            evaluation.experience_score,
        ])
        critique_update = await critic_node({
            "trip_brief": trip_brief,
            "draft_itinerary": draft,
            "critique_nums": iteration - 1,
        })
        new_critiques = [item.concern for item in critique_update.get("active_critiques", [])]
        critiques.extend(new_critiques)
        hard_issues = validate_itinerary_contract(trip_brief, draft)
        trajectory.append({
            "iteration": iteration,
            "score": score,
            "feedback": evaluation.reason,
            "hard_issues": hard_issues,
            "critique": new_critiques,
        })
        if score >= 6 and not hard_issues and not new_critiques:
            stop_reason = "quality_gate_passed"
            break
        if len(findings) >= scout_budget or iteration == max_iterations:
            stop_reason = "revision_budget_reached"
            break
        context = f"当前草稿：{draft[-5000:]}\n评估反馈：{evaluation.reason}\n体验官反馈：{' '.join(new_critiques)}"
        query = await _follow_up_query(trip_brief, findings, context)
        follow_up_queries.append(query)
        findings.extend(await _run_scouts([query]))
    else:  # pragma: no cover - the bounded loop always stops explicitly
        stop_reason = "revision_budget_reached"

    latest = trajectory[-1]
    passed = latest["score"] >= 6 and not latest["hard_issues"] and not latest["critique"]
    return {
        "report": draft,
        "initial_draft": initial_draft,
        "research_plan": plan,
        "follow_up_queries": follow_up_queries,
        "evidence": findings,
        "quality_trajectory": trajectory,
        "internal_quality_score": latest["score"],
        "revision_rounds": len(trajectory),
        "search_rounds": 1 + len(follow_up_queries),
        "scout_calls": len(findings),
        "completion_status": "passed" if passed else "degraded",
        "stop_reason": stop_reason,
        "unresolved_issues": list(dict.fromkeys([*latest["hard_issues"], *critiques[-1:]])),
        "latency_seconds": time.perf_counter() - started,
    }


async def run_task(task: dict, output_dir: Path, args) -> dict:
    path = output_dir / "cases" / f"{task['task_id']}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "task_id": task["task_id"], "query": task["query"], "variants": {}
    }
    if "initial" not in record:
        trip_brief, shared_seconds = await _prepare_brief(task)
        record["initial"] = {"trip_brief": trip_brief, "latency_seconds": shared_seconds}
        _write_json(path, record)

    if "openai_dr" not in record["variants"]:
        result = await _openai_dr(
            record["initial"]["trip_brief"], task["research_topics"],
            args.follow_up_rounds, args.scout_budget,
        )
        seconds = result.pop("latency_seconds")
        quality, defects, hard_issues = await _evaluate_report(
            record["initial"]["trip_brief"], result["report"], result["evidence"]
        )
        record["variants"]["openai_dr"] = {
            **result,
            "quality": quality,
            "defects": defects,
            "hard_issues": hard_issues,
            "gate_passed": quality["overall"] >= BENCHMARK_PASS_SCORE and not hard_issues,
            "revision_rounds": 0,
            "variant_latency_seconds": seconds,
            "end_to_end_latency_seconds": record["initial"]["latency_seconds"] + seconds,
            "completion_status": "passed" if quality["overall"] >= BENCHMARK_PASS_SCORE and not hard_issues else "degraded",
            "stop_reason": "plan_execute_synthesize",
            "sees_draft": False,
        }
        _write_json(path, record)

    if "ttd_dr" not in record["variants"]:
        result = await _ttd_dr(
            record["initial"]["trip_brief"], task["research_topics"],
            args.max_iterations, args.scout_budget,
        )
        seconds = result.pop("latency_seconds")
        initial_quality, _, _ = await _evaluate_report(
            record["initial"]["trip_brief"], result["initial_draft"], result["evidence"]
        )
        quality, defects, hard_issues = await _evaluate_report(
            record["initial"]["trip_brief"], result["report"], result["evidence"]
        )
        record["variants"]["ttd_dr"] = {
            **result,
            "initial_quality": initial_quality,
            "quality": quality,
            "defects": defects,
            "hard_issues": hard_issues,
            "gate_passed": quality["overall"] >= BENCHMARK_PASS_SCORE and not hard_issues,
            "max_iterations": args.max_iterations,
            "variant_latency_seconds": seconds,
            "end_to_end_latency_seconds": record["initial"]["latency_seconds"] + seconds,
            "sees_draft": True,
        }
        _write_json(path, record)

    if {"openai_dr", "ttd_dr"} <= set(record["variants"]) and args.pairwise_passes and not record.get("pairwise"):
        record["pairwise"] = await pairwise_judge(record, args.pairwise_passes)
        _write_json(path, record)
    return record


async def rejudge_record(record: dict, output_dir: Path, args) -> dict:
    brief = record["initial"]["trip_brief"]
    for name, variant in record.get("variants", {}).items():
        quality, defects, hard_issues = await _evaluate_report(
            brief, variant["report"], variant["evidence"]
        )
        variant.update(
            quality=quality,
            defects=defects,
            hard_issues=hard_issues,
            gate_passed=quality["overall"] >= BENCHMARK_PASS_SCORE and not hard_issues,
        )
        if name == "ttd_dr":
            variant["initial_quality"], _, _ = await _evaluate_report(
                brief, variant["initial_draft"], variant["evidence"]
            )
        if name == "openai_dr":
            variant["completion_status"] = "passed" if variant["gate_passed"] else "degraded"
    if {"openai_dr", "ttd_dr"} <= set(record.get("variants", {})):
        record["pairwise"] = await pairwise_judge(record, args.pairwise_passes)
    _write_json(output_dir / "cases" / f"{record['task_id']}.json", record)
    return record


async def async_main(args) -> None:
    output_dir = Path(args.output_dir)
    config = {
        "variants": list(VARIANTS),
        "rubric": RUBRIC,
        "ttd_max_revision_rounds": args.max_iterations,
        "scout_budget_per_variant": args.scout_budget,
        "scout_concurrency": FOLLOW_UP_CONCURRENCY,
        "search_results_per_query": EXPERIMENT_SEARCH_RESULTS,
        "openai_dr_follow_up_rounds": args.follow_up_rounds,
        "openai_dr_sees_draft": False,
        "ttd_dr_sees_draft": True,
        "pairwise_passes": args.pairwise_passes,
        "pass_score": BENCHMARK_PASS_SCORE,
    }
    tasks = load_tasks(Path(args.tasks), args.limit)
    if args.summarize_only or args.rejudge:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "cases").glob("*.json"))]
        if args.rejudge:
            records = [await rejudge_record(record, output_dir, args) for record in records]
    else:
        records = []
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task['task_id']}", flush=True)
            records.append(await run_task(task, output_dir, args))
    complete = [record for record in records if set(VARIANTS) <= set(record.get("variants", {}))]
    if complete:
        prepare_blind_review(complete, output_dir)
        print(json.dumps(write_summary(complete, output_dir, config), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/eval/itinerary_ab_tasks.jsonl")
    parser.add_argument("--output-dir", default="results/ab_openai_dr_vs_ttd")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=2, help="TTD-DR refinement-round ceiling")
    parser.add_argument("--scout-budget", type=int, default=DEFAULT_SCOUT_BUDGET, help="Maximum Scout calls per variant")
    parser.add_argument("--follow-up-rounds", type=int, default=1, help="OpenAI-DR style gap-driven extra search rounds")
    parser.add_argument("--pairwise-passes", type=int, default=2, help="Shuffled pairwise judge passes (order flipped)")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--rejudge", action="store_true", help="Re-score saved reports and re-run pairwise with the held-out judge")
    args = parser.parse_args()
    if args.max_iterations < 1 or args.scout_budget < 3 or args.follow_up_rounds < 0:
        parser.error("--max-iterations >= 1, --scout-budget >= 3, --follow-up-rounds >= 0")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
