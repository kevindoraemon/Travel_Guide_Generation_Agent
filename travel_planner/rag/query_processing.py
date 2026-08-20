"""Sequential query rewriting and metadata-condition extraction."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from travel_planner import logging as tp_logging
from travel_planner.llm import get_chat_model, safe_structured_output
from travel_planner.rag.config import RagConfig
from travel_planner.rag.schemas import MetadataConditions, QueryPlan, QueryRewriteResult


logger = tp_logging.get_logger(__name__)

KNOWN_CITIES = (
    "北京", "上海", "广州", "深圳", "杭州", "苏州", "南京", "成都", "重庆", "西安",
    "厦门", "青岛", "桂林", "阳朔", "昆明", "大理", "丽江", "拉萨", "乌鲁木齐",
    "三亚", "香港", "澳门", "台北", "东京", "京都", "大阪", "首尔", "清迈", "新加坡",
)
KNOWN_COUNTRIES = (
    "中国", "日本", "韩国", "泰国", "新加坡", "马来西亚", "越南", "法国", "意大利",
    "瑞士", "英国", "美国", "加拿大", "澳大利亚", "新西兰",
)
TOPIC_PATTERNS = {
    "family": ("亲子", "儿童", "带娃", "家庭"),
    "food": ("美食", "餐厅", "小吃", "餐饮"),
    "hotel": ("酒店", "住宿", "民宿"),
    "transport": ("交通", "地铁", "高铁", "公交", "租车"),
    "self_drive": ("自驾", "驾车", "环线"),
    "culture": ("文化", "历史", "博物馆", "古迹"),
    "nature": ("自然", "徒步", "山", "湖", "草原"),
    "shopping": ("购物", "商场", "买什么"),
    "attraction": ("景点", "游览", "打卡"),
}


def infer_topics(text: str) -> list[str]:
    """Return every controlled topic represented in a document or query."""

    return [
        name
        for name, patterns in TOPIC_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    ]


class QueryProcessor:
    """LLM-first processing with deterministic, offline-safe fallbacks."""

    def __init__(self, config: RagConfig, *, model: Any | None = None):
        self.config = config
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = get_chat_model(self.config.query_rewrite_role, max_tokens=800)
        return self._model

    @staticmethod
    def _fallback_rewrite(query: str) -> str:
        normalized = re.sub(r"\s+", " ", query).strip()
        # Dense models have finite context; keep the intent-rich beginning and end.
        if len(normalized) <= 700:
            return normalized
        return f"{normalized[:520]} {normalized[-160:]}"

    def rewrite(self, query: str) -> str:
        """Stage 1: turn conversational or verbose input into a standalone query."""

        if not self.config.query_rewrite_enabled:
            return self._fallback_rewrite(query)
        messages = [
            SystemMessage(
                content=(
                    "你是旅游知识库的查询改写器。将输入改写成一条独立、简洁、适合语义与关键词混合检索的查询。"
                    "保留目的地、天数、人群、预算、兴趣、交通方式和限制；不要回答问题，不要添加输入中没有的事实。"
                )
            ),
            HumanMessage(content=query),
        ]
        try:
            result = safe_structured_output(self._get_model(), QueryRewriteResult, messages)
            rewritten = re.sub(r"\s+", " ", result.rewritten_query).strip()
            return rewritten or self._fallback_rewrite(query)
        except Exception as exc:
            logger.warning("Query rewrite failed; using deterministic rewrite: %s", exc)
            return self._fallback_rewrite(query)

    @staticmethod
    def _rule_based_metadata(text: str) -> MetadataConditions:
        city = next((value for value in KNOWN_CITIES if value in text), None)
        country = next((value for value in KNOWN_COUNTRIES if value in text), None)
        topic = next(iter(infer_topics(text)), None)
        language = None
        if any(value in text for value in ("中文资料", "中文路书", "中文攻略")):
            language = "zh"
        elif any(value.lower() in text.lower() for value in ("English guide", "English itinerary")):
            language = "en"
        return MetadataConditions(city=city, country=country, topic=topic, language=language)

    def extract_metadata(self, original_query: str, rewritten_query: str) -> MetadataConditions:
        """Stage 2: extract only exact conditions safe for hard payload filtering."""

        fallback = self._rule_based_metadata(f"{original_query}\n{rewritten_query}")
        if not self.config.metadata_extraction_enabled:
            return fallback
        messages = [
            SystemMessage(
                content=(
                    "从旅游检索查询中提取可用于数据库精确过滤的元数据。只提取用户明确说出的条件；"
                    "不要根据城市推断国家，不确定就返回 null。topic 仅可使用 attraction、transport、food、"
                    "hotel、family、shopping、culture、nature、self_drive。language 仅可为 zh 或 en。"
                )
            ),
            HumanMessage(content=f"原始查询：{original_query}\n改写查询：{rewritten_query}"),
        ]
        try:
            result = safe_structured_output(self._get_model(), MetadataConditions, messages)
            # Rule extraction fills obvious values the model may omit, never overwriting it.
            merged = result.model_dump()
            for key, value in fallback.model_dump().items():
                if not merged.get(key) and value:
                    merged[key] = value
            return MetadataConditions.model_validate(merged)
        except Exception as exc:
            logger.warning("Metadata extraction failed; using rules: %s", exc)
            return fallback

    def plan(
        self,
        query: str,
        *,
        overrides: dict[str, str | None] | None = None,
    ) -> QueryPlan:
        rewritten = self.rewrite(query)
        filters = self.extract_metadata(query, rewritten)
        values = filters.model_dump()
        for key, value in (overrides or {}).items():
            if key in values and value:
                values[key] = value
        return QueryPlan(
            original_query=query,
            rewritten_query=rewritten,
            filters=MetadataConditions.model_validate(values),
        )
