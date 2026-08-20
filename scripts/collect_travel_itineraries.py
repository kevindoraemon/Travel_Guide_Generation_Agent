#!/usr/bin/env python3
"""Collect travel itineraries discovered through Google and Baidu.

The output is JSONL compatible with ``scripts/ingest_travel_knowledge.py``.
Only publicly accessible HTML pages are fetched. Search provenance, extraction
statistics and content hashes are retained for auditing and de-duplication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 TravelRAGCollector/1.0"
)

GOOGLE_QUERIES = [
    "北京 5日 旅游 路书 行程",
    "上海 3日 旅行 行程 攻略",
    "成都 4日 旅游 路书",
    "西安 4日 行程 攻略",
    "云南 7日 旅行 路线 路书",
    "新疆 10日 自驾 路书",
    "川西 7日 自驾 路书",
    "青甘大环线 8日 路书",
    "桂林 阳朔 4日 行程 攻略",
    "厦门 4日 旅行 行程",
    "重庆 4日 旅游 行程 攻略",
    "杭州 苏州 5日 行程 路书",
    "东京 5日 itinerary travel guide",
    "京都 大阪 6日 行程 攻略",
    "首尔 4日 itinerary travel guide",
    "清迈 5日 itinerary travel guide",
    "新加坡 4日 itinerary travel guide",
    "意大利 10 day itinerary travel guide",
    "瑞士 7 day itinerary travel guide",
    "New Zealand South Island 10 day road trip itinerary",
]

BAIDU_QUERIES = [
    "北京五日游 路书 行程安排",
    "上海三日游 路书 行程安排",
    "成都四日游 路书 攻略",
    "西安四日游 行程 路线",
    "云南七日游 路书 行程",
    "新疆十日自驾 路书",
    "川西自驾游 路书 行程",
    "青甘大环线 路书 每日行程",
    "桂林阳朔四日游 行程攻略",
    "厦门四日游 行程攻略",
    "重庆四日游 行程攻略",
    "杭州苏州五日游 行程攻略",
    "贵州七日游 路书 行程",
    "呼伦贝尔自驾 路书 行程",
    "海南环岛自驾 路书 行程",
    "日本东京五日游 行程攻略",
    "京都大阪六日游 行程攻略",
    "泰国清迈五日游 行程攻略",
    "新加坡四日游 行程攻略",
    "意大利十日游 行程攻略",
]

ITINERARY_SIGNALS = [
    r"路书",
    r"行程",
    r"路线",
    r"攻略",
    r"第[一二三四五六七八九十\d]+天",
    r"day\s*[-:]?\s*\d+",
    r"住宿",
    r"交通",
    r"景点",
    r"餐饮|美食",
    r"预算|费用",
]

CITY_NAMES = [
    "北京", "上海", "成都", "西安", "云南", "新疆", "川西", "青海", "甘肃",
    "桂林", "阳朔", "厦门", "重庆", "杭州", "苏州", "贵州", "呼伦贝尔",
    "海南", "东京", "京都", "大阪", "首尔", "清迈", "新加坡", "意大利",
    "瑞士", "New Zealand", "South Island",
]

SKIP_HOST_PARTS = (
    "google.", "gstatic.com", "googleusercontent.com", "baidu.com", "bdstatic.com",
    "bing.com", "youtube.com", "facebook.com", "instagram.com", "x.com",
)

SKIP_SUFFIXES = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip",
    ".rar", ".jpg", ".jpeg", ".png", ".gif", ".mp4",
)


@dataclass(frozen=True)
class Candidate:
    engine: str
    query: str
    rank: int
    url: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonicalize_url(url: str) -> str | None:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    host = parts.hostname.lower() if parts.hostname else ""
    if any(part in host for part in SKIP_HOST_PARTS):
        return None
    path_lower = parts.path.lower()
    if path_lower.endswith(SKIP_SUFFIXES):
        return None
    # Remove common tracking parameters while retaining page-identifying params.
    tracking_prefixes = ("utm_", "spm", "from", "source", "ref", "share_")
    query_parts = []
    for item in parts.query.split("&") if parts.query else []:
        key = item.split("=", 1)[0].lower()
        if not any(key.startswith(prefix) for prefix in tracking_prefixes):
            query_parts.append(item)
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, "&".join(query_parts), ""))


class Collector:
    def __init__(self, delay: float, timeout: float, min_chars: int) -> None:
        self.delay = delay
        self.timeout = timeout
        self.min_chars = min_chars
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        self.robot_cache: dict[str, RobotFileParser | None] = {}
        self.stats: Counter[str] = Counter()
        self.seen_urls: set[str] = set()
        self.seen_hashes: set[str] = set()

    def pause(self, factor: float = 1.0) -> None:
        time.sleep(max(0.0, self.delay * factor * random.uniform(0.8, 1.25)))

    def robots_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robot_cache:
            parser = RobotFileParser()
            robots_url = urljoin(origin, "/robots.txt")
            parser.set_url(robots_url)
            try:
                response = self.session.get(
                    robots_url,
                    timeout=min(self.timeout, 3.0),
                    allow_redirects=True,
                )
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                    self.robot_cache[origin] = parser
                else:
                    self.robot_cache[origin] = None
            except requests.RequestException:
                # An unavailable robots file is not interpreted as a prohibition.
                self.robot_cache[origin] = None
        parser = self.robot_cache[origin]
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def get_html(self, url: str, check_robots: bool = True) -> tuple[str, str, int] | None:
        if check_robots and not self.robots_allowed(url):
            self.stats["robots_denied"] += 1
            return None
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException:
            self.stats["fetch_error"] += 1
            return None
        self.pause()
        status = response.status_code
        content_type = response.headers.get("content-type", "").lower()
        if status != 200:
            self.stats[f"http_{status}"] += 1
            return None
        if "html" not in content_type and not response.text.lstrip().startswith(("<!DOCTYPE", "<html")):
            self.stats["non_html"] += 1
            return None
        # Keep memory bounded on accidentally large pages.
        if len(response.content) > 4_000_000:
            self.stats["too_large"] += 1
            return None
        return response.url, response.text, status

    def google_candidates(self, query: str, page: int) -> list[Candidate]:
        start = page * 10
        url = f"https://www.google.com/search?q={quote_plus(query)}&num=10&hl=zh-CN&start={start}"
        result = self.get_html(url, check_robots=False)
        if not result:
            self.stats["google_search_error"] += 1
            return []
        _, html, _ = result
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for heading in soup.select("h3"):
            anchor = heading.find_parent("a", href=True)
            if anchor:
                links.append(anchor["href"])
        if not links:
            links = [anchor["href"] for anchor in soup.select("a[href]")]
        output: list[Candidate] = []
        for href in links:
            if href.startswith("/url?"):
                href = parse_qs(urlsplit(href).query).get("q", [""])[0]
            clean = canonicalize_url(href)
            if clean and clean not in {candidate.url for candidate in output}:
                output.append(Candidate("google", query, start + len(output) + 1, clean))
        self.stats["google_search_results"] += len(output)
        return output

    def baidu_candidates(self, query: str, page: int) -> list[Candidate]:
        pn = page * 10
        url = f"https://www.baidu.com/s?wd={quote_plus(query)}&ie=utf-8&rn=10&pn={pn}"
        result = self.get_html(url, check_robots=False)
        if not result:
            self.stats["baidu_search_error"] += 1
            return []
        _, html, _ = result
        soup = BeautifulSoup(html, "html.parser")
        output: list[Candidate] = []
        for heading in soup.select("h3"):
            anchor = heading.find("a", href=True) or heading.find_parent("a", href=True)
            if not anchor:
                continue
            href = urljoin("https://www.baidu.com", anchor["href"])
            # Preserve Baidu redirect links here; get_html follows the redirect and
            # canonicalization is performed on the final URL.
            if href not in {candidate.url for candidate in output}:
                output.append(Candidate("baidu", query, pn + len(output) + 1, href))
        self.stats["baidu_search_results"] += len(output)
        return output

    @staticmethod
    def extract_text(html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        for node in soup.select("script,style,noscript,svg,canvas,form,nav,footer,header,aside"):
            node.decompose()

        title = ""
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.title and soup.title.string:
            title = soup.title.string
        elif soup.find("h1"):
            title = soup.find("h1").get_text(" ", strip=True)

        selectors = [
            "article", "main", "[role=main]", ".article", ".article-content",
            ".post", ".post-content", ".entry-content", ".content", "#content",
        ]
        candidates = []
        for selector in selectors:
            for node in soup.select(selector):
                text = normalize_space(node.get_text("\n", strip=True))
                if text:
                    candidates.append(text)
        if candidates:
            content = max(candidates, key=len)
        elif soup.body:
            content = normalize_space(soup.body.get_text("\n", strip=True))
        else:
            content = normalize_space(soup.get_text("\n", strip=True))
        return normalize_space(title)[:500], content

    def quality_ok(self, title: str, content: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if len(content) < self.min_chars:
            reasons.append("short_content")
        combined = f"{title}\n{content[:20000]}"
        signals = sum(bool(re.search(pattern, combined, re.I)) for pattern in ITINERARY_SIGNALS)
        if signals < 3:
            reasons.append("weak_itinerary_signals")
        if len(set(content)) < 80:
            reasons.append("low_information")
        return not reasons, reasons

    @staticmethod
    def infer_metadata(query: str, content: str) -> tuple[str | None, str]:
        city = next((name for name in CITY_NAMES if name.lower() in query.lower()), None)
        sample = content[:5000]
        cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
        language = "zh" if cjk > max(20, len(sample) * 0.08) else "en"
        return city, language

    def collect_candidate(self, candidate: Candidate) -> dict | None:
        if candidate.url in self.seen_urls:
            self.stats["duplicate_candidate_url"] += 1
            return None
        self.seen_urls.add(candidate.url)

        # Search-engine redirect URLs need to be resolved before robots checks.
        check_robots = "baidu.com/link" not in candidate.url
        result = self.get_html(candidate.url, check_robots=check_robots)
        if not result:
            return None
        final_url, html, status = result
        clean_url = canonicalize_url(final_url)
        if not clean_url:
            self.stats["invalid_final_url"] += 1
            return None
        if clean_url in self.seen_urls and clean_url != candidate.url:
            self.stats["duplicate_final_url"] += 1
            return None
        self.seen_urls.add(clean_url)

        # If a redirect resolved to a new origin, enforce that site's robots rule
        # before accepting the fetched page.
        if not check_robots and not self.robots_allowed(clean_url):
            self.stats["robots_denied_after_redirect"] += 1
            return None

        title, content = self.extract_text(html)
        ok, reasons = self.quality_ok(title, content)
        if not ok:
            for reason in reasons:
                self.stats[reason] += 1
            return None

        normalized_for_hash = re.sub(r"\s+", "", content).lower()
        content_hash = hashlib.sha256(normalized_for_hash.encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            self.stats["duplicate_content"] += 1
            return None
        self.seen_hashes.add(content_hash)
        city, language = self.infer_metadata(candidate.query, content)
        host = urlsplit(clean_url).hostname or ""
        return {
            "title": title or f"{city or '旅游'}行程路书",
            "content": content,
            "source_url": clean_url,
            "search_engine": candidate.engine,
            "source_type": "travel_itinerary",
            "language": language,
            "country": None,
            "city": city,
            "tags": ["旅游", "路书", "行程", candidate.engine],
            "metadata": {
                "search_query": candidate.query,
                "search_rank": candidate.rank,
                "source_domain": host,
                "content_hash": content_hash,
                "http_status": status,
                "content_length": len(content),
                "collector": "travel_rag_collector_v1",
            },
            "fetched_at": utc_now(),
        }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--google-target", type=int, default=25)
    parser.add_argument("--baidu-target", type=int, default=25)
    parser.add_argument("--pages-per-query", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=18.0)
    parser.add_argument("--min-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collector = Collector(args.delay, args.timeout, args.min_chars)
    targets = {"google": args.google_target, "baidu": args.baidu_target}
    queries = {"google": GOOGLE_QUERIES, "baidu": BAIDU_QUERIES}
    collected: dict[str, list[dict]] = {"google": [], "baidu": []}

    for engine in ("google", "baidu"):
        search = collector.google_candidates if engine == "google" else collector.baidu_candidates
        print(f"[{utc_now()}] collecting {engine}, target={targets[engine]}", flush=True)
        for page in range(args.pages_per_query):
            for query in queries[engine]:
                if len(collected[engine]) >= targets[engine]:
                    break
                candidates = search(query, page)
                print(
                    f"[{engine}] query={query!r}, page={page + 1}, candidates={len(candidates)}, "
                    f"accepted={len(collected[engine])}",
                    flush=True,
                )
                for candidate in candidates:
                    if len(collected[engine]) >= targets[engine]:
                        break
                    collector.stats[f"{engine}_candidates_attempted"] += 1
                    record = collector.collect_candidate(candidate)
                    if record:
                        collected[engine].append(record)
                        collector.stats[f"{engine}_accepted"] += 1
                        print(
                            f"[{engine} {len(collected[engine])}/{targets[engine]}] "
                            f"{record['title'][:80]} | {record['source_url']}",
                            flush=True,
                        )
                    elif collector.stats[f"{engine}_candidates_attempted"] % 5 == 0:
                        recent_stats = {
                            key: value for key, value in collector.stats.items()
                            if key not in {"google_search_results", "baidu_search_results"}
                        }
                        print(f"[{engine}] rejection stats={recent_stats}", flush=True)
            if len(collected[engine]) >= targets[engine]:
                break

    records = collected["google"] + collected["baidu"]
    write_jsonl(args.output, records)
    report_path = args.report or args.output.with_suffix(".report.json")
    report = {
        "started_and_completed_at": utc_now(),
        "requested": targets,
        "collected": {engine: len(items) for engine, items in collected.items()},
        "total": len(records),
        "unique_domains": len({urlsplit(item["source_url"]).hostname for item in records}),
        "stats": dict(sorted(collector.stats.items())),
        "output": str(args.output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(records) == args.google_target + args.baidu_target else 2


if __name__ == "__main__":
    sys.exit(main())
