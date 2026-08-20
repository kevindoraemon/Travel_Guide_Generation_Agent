#***********************************************
#      Filename: llm.py
#   Description: 大模型客户端
#***********************************************


from __future__ import annotations

import os
from typing import Any, Dict, Optional
from langchain.chat_models import init_chat_model

from travel_planner.utils import load_config
from travel_planner import logging as tp_logging


# 初始化 logger
logger = tp_logging.get_logger(__name__)

# 缓存 CONFIG，避免重复导入 (config_path, stage, loader_id)
_CONFIG_CACHE: Dict[tuple[str, str, int], Dict[str, Any]] = {}

# 默认的 stage
DEFAULT_STAGE = "prod"


class LLMConfigError(ValueError):
    """当 LLM 配置错误或者不合法时抛出该异常"""


def _resolve_stage(stage: str | None) -> str:
    return stage or os.environ.get("STAGE") or DEFAULT_STAGE


def _load_stage_config(stage_name: str | None, config_path: str | None) -> Dict[str, Any]:
    """加载 config.yml"""

    # Key 作为 config loader 的唯一标识
    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), stage_name, id(load_config))

    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    cfg = load_config(stage_name=stage_name, config_path=config_path)
    if cfg is None:
        raise LLMConfigError(f"No config found for stage '{stage_name}'")

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def _build_openai_kwargs(
    handle: str,
    api_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: Optional[int],
) -> Dict[str, Any]:
    """初始化 llm client 参数，例如 api_key, base_url"""

    model = handle or api_cfg.get("default_model")
    if not model:
        raise LLMConfigError("OpenAI config requires a model name!")

    # 显式指定 model_provider 为 openai，确保任何 OpenAI 兼容模型（如智谱 glm、deepseek、qwen）
    # 都能正确加载，而不是依赖 LangChain 按模型名推断 provider
    kwargs: Dict[str, Any] = {
        "model": model,
        "model_provider": "openai",
    }

    # api_key, base_url
    for key in ("api_key", "base_url", "organization"):
        if api_cfg.get(key):
            kwargs[key] = api_cfg[key]

    # 温度系数
    if api_cfg.get("temperature") is not None:
        kwargs["temperature"] = api_cfg["temperature"]

    # 最大 token 数
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # 请求超时
    if timeout_seconds is not None:
        # OpenAI 兼容接口可用接收 timeout/request_timeout
        kwargs["timeout"] = timeout_seconds
        kwargs["request_timeout"] = timeout_seconds

    # 模型参数（可选）
    model_kwargs: Dict[str, Any] = {}
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    # 429/5xx 自动重试（指数退避），应对智谱等网关的瞬时限流
    kwargs["max_retries"] = 8

    return kwargs


def _resolve_config_max_tokens(api_cfg: Dict[str, Any], handle: str) -> int | None:
    """解析最大 token 数"""
    models_cfg = api_cfg.get("models") or {}
    model_cfg = models_cfg.get(handle) or {}
    return model_cfg.get("max_tokens")


def _resolve_timeout_seconds(api_cfg: Dict[str, Any], role_cfg: Dict[str, Any]) -> Optional[int]:
    """解析 timeout/request_timeout/timeout_seconds 参数"""
    for cfg in (role_cfg, api_cfg):
        for key in ("timeout", "request_timeout", "timeout_seconds"):
            if cfg.get(key) is not None:
                return cfg.get(key)
    return None


def _build_kwargs(
    backend: str,
    handle: str,
    api_cfg: Dict[str, Any],
    role_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: int | None,
) -> Dict[str, Any]:

    if backend == "openai":
        return _build_openai_kwargs(handle, api_cfg, max_tokens, timeout_seconds)
    else:
        raise LLMConfigError(f"Unsupported backend '{backend}'")


def get_chat_model(role: str, *, stage: str | None = None, max_tokens: int | None = None):
    """根据 config 和 role 返回 LLM client.

    Args:
        role: 角色名，例如 coordinator, scout_main, briefing, writer, critic, evaluator 等
        stage: stage name
        max_tokens: 最大 tokens
    """

    # 获取 config 路径
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    resolved_stage = _resolve_stage(stage)

    # 加载 config.yml
    cfg = _load_stage_config(resolved_stage, config_path)

    # 获取 role 配置
    roles_cfg = cfg.get("roles", {})
    if role not in roles_cfg:
        # 清除 cache 重新加载一次
        _CONFIG_CACHE.clear()
        cfg = _load_stage_config(resolved_stage, config_path)
        roles_cfg = cfg.get("roles", {})

    # 如果 role 配置错误
    if role not in roles_cfg:
        available = ", ".join(sorted(roles_cfg.keys())) or "<none>"
        raise LLMConfigError(
            f"Role '{role}' not found for stage '{resolved_stage}' using config '{config_path}'. Available: {available}"
        )

    # 解析 backend 和 handle
    role_cfg = roles_cfg[role]
    backend = role_cfg.get("backend")
    handle = role_cfg.get("handle")
    if not backend or not handle:
        raise LLMConfigError(f"Role '{role}' is missing backend or handle")

    # 解析 llm api config
    api_cfg = cfg.get("cognition", {}).get(backend)
    if api_cfg is None:
        raise LLMConfigError(f"No cognition config for backend '{backend}'")

    # 获取超时时间
    resolved_timeout = _resolve_timeout_seconds(api_cfg, role_cfg)
    logger.info(
        "Selected cognition backend '%s' for role '%s' with handle '%s' (timeout=%s)",
        backend,
        role,
        handle,
        resolved_timeout,
    )

    # 获取输出最大 token 数
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = role_cfg.get("max_tokens")
    if resolved_max_tokens is None:
        resolved_max_tokens = _resolve_config_max_tokens(api_cfg, handle)

    # 新建 llm client
    kwargs = _build_kwargs(
        backend=backend,
        handle=handle,
        api_cfg=api_cfg,
        role_cfg=role_cfg,
        max_tokens=resolved_max_tokens,
        timeout_seconds=resolved_timeout
    )
    return init_chat_model(**kwargs)


def safe_structured_output(model, schema, messages):
    """带 fallback 的结构化输出调用，兼容各类 OpenAI 兼容接口（火山引擎/智谱/DeepSeek 等）。

    依次尝试三层 fallback，确保各种模型都能稳定返回结构化结果：
        1. function_calling：OpenAI tools API（部分模型不发起 tool call）
        2. json_mode：response_format json_object（部分模型不支持）
        3. raw_json：普通 invoke + 手动解析 JSON（终极保底，兼容所有模型）

    Args:
        model: LangChain ChatModel 实例
        schema: Pydantic BaseModel 类（如 TripRequirement）
        messages: 传给模型的 messages 列表

    Returns:
        schema 的实例

    Raises:
        ValueError: 所有结构化输出方式均失败
    """
    import json
    import re
    from langchain_core.messages import HumanMessage, SystemMessage

    # 从 schema 提取 Pydantic 字段 schema（用于提示模型）
    try:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = "{}"

    # Layer 1 & 2: function_calling → json_mode
    for method in ("function_calling", "json_mode"):
        try:
            structured = model.with_structured_output(schema, method=method)
            result = structured.invoke(messages)
            if result is not None:
                logger.info("结构化输出成功 (schema=%s, method=%s)", schema.__name__, method)
                return result
            logger.debug("结构化输出 method='%s' 返回 None，尝试下一种方式", method)
        except Exception as exc:
            logger.debug("结构化输出 method='%s' 失败: %s，尝试下一种方式", method, exc)
            continue

    # Layer 3: 终极保底 - 普通 invoke + 手动 JSON 解析
    logger.info("前两层 fallback 失败，启用终极保底方案：raw_json 解析")
    try:
        # 在原 messages 末尾追加一条"请输出 JSON"的指令
        raw_instruction = (
            "请严格按照以下 JSON Schema 输出 JSON，不要输出任何其他内容、解释或代码块标记：\n"
            f"{schema_json}"
        )
        raw_messages = list(messages) + [HumanMessage(content=raw_instruction)]
        response = model.invoke(raw_messages)
        content = getattr(response, "content", str(response))

        # 提取 JSON（兼容带 ```json ``` 代码块或纯 JSON 两种情况）
        json_str = _extract_json(content)
        if json_str:
            data = json.loads(json_str)
            result = schema.model_validate(data)
            logger.info("结构化输出成功 (schema=%s, method=raw_json)", schema.__name__)
            return result
    except Exception as exc:
        logger.error("终极保底方案失败：%s", exc)

    raise ValueError(
        f"模型无法生成 {schema.__name__} 结构化输出。"
        "请确认模型支持 function_calling、json_mode 或能输出合法 JSON。"
    )


def _extract_json(text: str) -> str | None:
    """从模型输出文本中提取 JSON 字符串（兼容代码块包裹或纯 JSON）"""
    import re

    # 1) 优先匹配 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) 匹配 { ... } 对象（贪婪最外层）
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()

    return None
