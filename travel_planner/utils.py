#***********************************************
#      Filename: utils.py
#   Description: 旅游路书工具函数库
#***********************************************

import asyncio
import os
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Awaitable, Callable, Iterable, TypeVar


T = TypeVar("T")
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """获取今天的日期并返回格式化的字符串（跨平台兼容：Windows 不支持 %-d）"""
    now = datetime.now()
    return f"{now.strftime('%a %b')} {now.day}, {now.year}"

def get_current_dir() -> Path:
    """获取当前的目录"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


async def gather_with_concurrency(
    limit: int,
    factories: Iterable[Callable[[], Awaitable[T]]],
) -> list[T]:
    """Run async factories in input order with a hard concurrency ceiling."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    semaphore = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return await asyncio.gather(*(run(factory) for factory in factories))


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 5,
    base_delay: float = 8.0,
    max_delay: float = 120.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential-backoff retry for transient provider errors (e.g. 429 rate limits)."""

    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except retry_on:
            if attempt == attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
    raise RuntimeError("unreachable")


# ===== CONFIG LOADER =====

def get_config_yml(path, section_name, subsection_name=None):
    """读取 yaml 文件"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, encoding="utf8") as f:
        data = yaml.safe_load(f)
        try:
            return (
                data[section_name]
                if subsection_name is None
                else data[section_name][subsection_name]
            )
        except KeyError as e:
            raise KeyError(
                f"No such section or subsection in config file: {section_name}, {subsection_name}. Config file: {path}"
            ) from e


def load_config(stage_name=None, config_path=None):
    """加载配置，并解析值为 ``${NAME}`` 的环境变量引用。"""
    config = get_config_yml(
        path=config_path, section_name="stages", subsection_name=stage_name
    )
    return _resolve_environment_references(config)


def _resolve_environment_references(value):
    """递归解析环境变量引用；变量缺失时返回空字符串。"""

    if isinstance(value, dict):
        return {key: _resolve_environment_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment_references(item) for item in value]
    if isinstance(value, str):
        match = _ENV_REFERENCE.fullmatch(value.strip())
        if match:
            return os.environ.get(match.group(1), "")
    return value
