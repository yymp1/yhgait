#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gait_demo_core import analyze_cache, load_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 gait demo cache 构建固定测试集检索分析结果。")
    parser.add_argument("--cache-path", required=True, help="embedding cache 路径。")
    parser.add_argument("--analysis-path", required=True, help="输出分析 JSON 路径。")
    parser.add_argument("--topk", type=int, default=5, help="分析时统计的 Top-K。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache_path).resolve()
    analysis_path = Path(args.analysis_path).resolve()
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cache_path)
    analysis = analyze_cache(cache, topk=args.topk)

    with analysis_path.open("w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=2)

    summary = {
        "analysis_path": str(analysis_path),
        "checkpoint_path": analysis["checkpoint_path"],
        "probe_count": analysis["probe_count"],
        "topk": analysis["topk"],
        "overall": analysis["overall"],
        "groups": analysis["groups"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
