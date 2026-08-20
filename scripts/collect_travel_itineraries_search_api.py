#!/usr/bin/env python3
"""Collect diverse travel itineraries through the configured search API.

Unlike HTML scraping of Google/Baidu result pages, this uses the project's
configured search provider for discovery and raw page extraction. Every record
keeps its original public URL and auditable search provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.tools.search_factory import (
    get_search_client,
    get_search_defaults,
    get_search_provider,
)


QUERIES = [
    ("北京", "北京 5日 自由行 路书 每日行程 交通 住宿"),
    ("上海", "上海 3日 自由行 路书 每日行程 交通 美食"),
    ("成都", "成都 4日 旅游路书 每日行程 景点 美食"),
    ("西安", "西安 4日 自由行 路书 每日路线 交通"),
    ("云南", "云南 昆明 大理 丽江 7日 路书 每日行程"),
    ("新疆", "新疆 北疆 10日 自驾 路书 每日路线"),
    ("川西", "川西 7日 自驾 路书 每日行程 住宿"),
    ("青甘大环线", "青甘大环线 8日 自驾 路书 每日路线"),
    ("桂林", "桂林 阳朔 4日 自由行 路书 每日行程"),
    ("厦门", "厦门 鼓浪屿 4日 路书 每日行程 美食"),
    ("重庆", "重庆 4日 自由行 路书 每日路线 美食"),
    ("杭州苏州", "杭州 苏州 5日 自由行 路书 每日行程"),
    ("贵州", "贵州 7日 自驾 路书 每日行程"),
    ("呼伦贝尔", "呼伦贝尔 7日 自驾 路书 每日路线"),
    ("海南", "海南 环岛 7日 自驾 路书 每日行程"),
    ("南京", "南京 3日 自由行 路书 每日行程"),
    ("青岛", "青岛 4日 自由行 路书 每日路线"),
    ("长沙张家界", "长沙 张家界 5日 路书 每日行程"),
    ("洛阳开封", "洛阳 开封 4日 文化旅行 路书 行程"),
    ("山西", "山西 古建 7日 自驾 路书 每日路线"),
    ("东京", "东京 5日 自由行 itinerary 每日行程"),
    ("关西", "京都 大阪 奈良 6日 itinerary 行程攻略"),
    ("首尔", "首尔 4日 itinerary travel guide daily"),
    ("清迈", "清迈 5日 itinerary travel guide daily"),
    ("新加坡", "新加坡 4日 itinerary travel guide daily"),
    ("意大利", "Italy 10 day itinerary Rome Florence Venice daily"),
    ("瑞士", "Switzerland 7 day itinerary daily route"),
    ("新西兰南岛", "New Zealand South Island 10 day road trip itinerary"),
    ("冰岛", "Iceland ring road 8 day itinerary daily"),
    ("土耳其", "Turkey 10 day itinerary Istanbul Cappadocia daily"),
]

# Extra destinations keep the corpus diverse after strict quality filtering.
QUERIES.extend(
    [
        ("武汉", "武汉 3天 自由行 路书 第一天 第二天 第三天 交通 美食"),
        ("大同平遥", "大同 平遥 5天 自驾 路书 每日行程 住宿"),
        ("敦煌", "敦煌 嘉峪关 5天 自驾 路书 每日路线"),
        ("宁夏", "宁夏 银川 中卫 5天 自驾 路书 每日行程"),
        ("珠海澳门", "珠海 澳门 4天 自由行 路书 每日行程"),
        ("香港", "香港 4天 自由行 itinerary day 1 day 2 day 3 day 4"),
        ("三亚", "三亚 5天 自由行 路书 每日行程 住宿 美食"),
        ("北海涠洲岛", "北海 涠洲岛 5天 自由行 路书 每日行程"),
        ("西双版纳", "西双版纳 5天 自由行 路书 每日路线"),
        ("香格里拉", "香格里拉 梅里雪山 6天 自驾 路书 每日行程"),
        ("九寨沟", "九寨沟 黄龙 4天 自驾 路书 每日行程"),
        ("婺源", "婺源 景德镇 4天 自驾 路书 每日路线"),
        ("黄山", "黄山 宏村 4天 自由行 路书 每日行程"),
        ("恩施", "恩施 5天 自驾 路书 每日路线 住宿"),
        ("武夷山", "武夷山 3天 自由行 路书 第一天 第二天 第三天"),
        ("泉州", "泉州 3天 自由行 路书 每日行程 美食"),
        ("扬州", "扬州 3天 自由行 路书 每日行程"),
        ("哈尔滨雪乡", "哈尔滨 雪乡 5天 自由行 路书 每日行程"),
        ("台北", "Taipei 5 day itinerary day 1 day 2 day 3 day 4 day 5"),
        ("成都重庆", "Chengdu Chongqing 7 day itinerary daily route"),
    ]
)

SKIP_DOMAIN_PARTS = (
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "scribd.com",
    "tripadvisor.com",
    "tiktok.com",
    "pinterest.com",
)

DAY_PATTERNS = (
    r"第(?:[一二三四五六七八九十]+|\d+)[天日]",
    r"\bday\s*[-:]?\s*\d+\b",
    r"(?:^|\s)d\s*\d+(?:\s|[:：])",
)

SIGNALS = [
    r"第[一二三四五六七八九十\d]+天",
    r"day\s*[-:]?\s*\d+",
    r"行程|itinerary",
    r"路线|route",
    r"交通|transport|drive|train",
    r"住宿|hotel|stay",
    r"景点|attraction|visit",
    r"美食|餐饮|food|restaurant",
]


def canonical_url(url: str) -> str | None:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    tracking = ("utm_", "spm", "from", "ref", "source", "share_")
    query = "&".join(
        part for part in parts.query.split("&")
        if part and not part.split("=", 1)[0].lower().startswith(tracking)
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", query, ""))


def normalize(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u3000", " ")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def quality_ok(
    title: str,
    content: str,
    min_chars: int,
    domain: str,
    city: str,
) -> tuple[bool, str]:
    if any(part in domain.lower() for part in SKIP_DOMAIN_PARTS):
        return False, "blocked_domain"
    if normalize(title).casefold() == normalize(city).casefold():
        return False, "generic_title"
    if len(content) < min_chars:
        return False, "short_content"
    sample = f"{title}\n{content[:30000]}"
    day_markers = {
        match.group(0).strip().lower()
        for pattern in DAY_PATTERNS
        for match in re.finditer(pattern, sample, re.I | re.M)
    }
    if len(day_markers) < 2:
        return False, "insufficient_daily_structure"
    signal_count = sum(bool(re.search(pattern, sample, re.I)) for pattern in SIGNALS)
    if signal_count < 4:
        return False, "weak_itinerary_signals"
    if len(set(content)) < 100:
        return False, "low_information"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--max-per-city", type=int, default=3)
    parser.add_argument("--max-per-domain", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}; pass --overwrite to replace it")

    provider = get_search_provider()
    client = get_search_client()
    defaults = get_search_defaults()
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    city_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    records: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for city, query in QUERIES:
            if len(records) >= args.target:
                break
            if city_counts[city] >= args.max_per_city:
                continue
            try:
                response = provider.search(
                    client,
                    query,
                    max_results=args.max_results,
                    include_raw_content=True,
                    topic="general",
                    timeout_seconds=defaults.get("timeout_seconds"),
                )
            except Exception as exc:
                stats["search_error"] += 1
                print(f"[search-error] {query}: {type(exc).__name__}: {exc}", flush=True)
                continue
            results = response.get("results", []) if isinstance(response, dict) else []
            stats["search_results"] += len(results)
            for rank, result in enumerate(results, 1):
                if len(records) >= args.target or city_counts[city] >= args.max_per_city:
                    break
                url = canonical_url(str(result.get("url") or ""))
                if not url or url in seen_urls:
                    stats["duplicate_or_invalid_url"] += 1
                    continue
                domain = urlsplit(url).hostname or ""
                if domain_counts[domain] >= args.max_per_domain:
                    stats["domain_cap"] += 1
                    continue
                title = normalize(str(result.get("title") or ""))[:500]
                content = normalize(str(result.get("raw_content") or result.get("content") or ""))
                ok, reason = quality_ok(title, content, args.min_chars, domain, city)
                if not ok:
                    stats[reason] += 1
                    continue
                content_hash = hashlib.sha256(re.sub(r"\s+", "", content).lower().encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    stats["duplicate_content"] += 1
                    continue
                seen_urls.add(url)
                seen_hashes.add(content_hash)
                city_counts[city] += 1
                domain_counts[domain] += 1
                cjk = len(re.findall(r"[\u4e00-\u9fff]", content[:5000]))
                record = {
                    "title": title or f"{city}旅游路书",
                    "content": content,
                    "source_url": url,
                    "search_engine": "tavily",
                    "source_type": "travel_itinerary",
                    "language": "zh" if cjk > 100 else "en",
                    "country": None,
                    "city": city,
                    "tags": ["旅游", "路书", "行程", city],
                    "metadata": {
                        "search_query": query,
                        "search_rank": rank,
                        "source_domain": domain,
                        "content_hash": content_hash,
                        "content_length": len(content),
                        "collector": "travel_search_api_collector_v1",
                    },
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                stats["accepted"] += 1
                print(f"[{len(records)}/{args.target}] {city} | {title[:60]} | {url}", flush=True)

    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "requested": args.target,
        "collected": len(records),
        "cities": dict(sorted(city_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
        "unique_domains": len(domain_counts),
        "stats": dict(sorted(stats.items())),
        "output": str(args.output),
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(records) == args.target else 2


if __name__ == "__main__":
    sys.exit(main())
