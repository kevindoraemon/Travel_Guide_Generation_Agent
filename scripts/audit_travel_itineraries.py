#!/usr/bin/env python3
"""Audit a travel-itinerary JSONL file and create a readable source catalog."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold()[:40000])
    return {" ".join(words[index:index + 7]) for index in range(0, max(0, len(words) - 6), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=50)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.30)
    args = parser.parse_args()

    rows = load_rows(args.input)
    urls = [row.get("source_url") for row in rows]
    hashes = [(row.get("metadata") or {}).get("content_hash") for row in rows]
    city_counts = Counter(str(row.get("city") or "unknown") for row in rows)
    domain_counts = Counter(urlsplit(url).hostname or "unknown" for url in urls if url)
    document_shingles = [shingles(str(row.get("content") or "")) for row in rows]
    near_duplicates: list[dict] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            union = document_shingles[left] | document_shingles[right]
            score = len(document_shingles[left] & document_shingles[right]) / len(union) if union else 0
            if score >= args.near_duplicate_threshold:
                near_duplicates.append(
                    {
                        "left": left + 1,
                        "right": right + 1,
                        "jaccard": round(score, 4),
                        "left_url": urls[left],
                        "right_url": urls[right],
                    }
                )

    errors: list[str] = []
    if len(rows) != args.expected:
        errors.append(f"expected {args.expected} documents, got {len(rows)}")
    if len(set(urls)) != len(rows):
        errors.append("duplicate source URLs found")
    if None in hashes or len(set(hashes)) != len(rows):
        errors.append("missing or duplicate content hashes found")
    if near_duplicates:
        errors.append(f"{len(near_duplicates)} near-duplicate pairs found")

    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "status": "pass" if not errors else "fail",
        "documents": len(rows),
        "unique_urls": len(set(urls)),
        "unique_content_hashes": len(set(hashes)),
        "unique_domains": len(domain_counts),
        "unique_city_groups": len(city_counts),
        "cities": dict(sorted(city_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "near_duplicates": near_duplicates,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 50 份旅游路书来源目录",
        "",
        f"> 生成时间：{report['audited_at']}；状态：**{report['status']}**；"
        f"目的地区域：{len(city_counts)}；来源域名：{len(domain_counts)}。",
        "",
        "| # | 目的地 | 标题 | 来源域名 | 正文字数 |",
        "|---:|---|---|---|---:|",
    ]
    for index, row in enumerate(rows, 1):
        title = str(row.get("title") or "未命名").replace("|", "\\|").replace("\n", " ")
        url = str(row.get("source_url") or "")
        domain = urlsplit(url).hostname or "unknown"
        content_length = len(str(row.get("content") or ""))
        lines.append(f"| {index} | {row.get('city') or '-'} | [{title}]({url}) | {domain} | {content_length} |")
    args.catalog.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
