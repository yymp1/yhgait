#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gait_demo_core import (  # noqa: E402
    PROBE_TYPES,
    find_probe_index,
    item_meta,
    load_cache,
    probe_markdown,
    random_probe_index,
    retrieve_topk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="命令行方式验证步态检索 demo。")
    parser.add_argument(
        "--cache-path",
        default=str(ROOT / "reports" / "gait_demo_cache_formal_conservative_real_iter01000.npz"),
        help="embedding 缓存路径。",
    )
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"),
        help="构建缓存时使用的评估配置。",
    )
    parser.add_argument("--checkpoint-iter", type=int, default=1000, help="构建缓存时使用的 checkpoint iter。")
    parser.add_argument("--topk", type=int, default=5, help="返回前 K 个结果。")
    parser.add_argument("--subject", help="手动指定 probe subject id。")
    parser.add_argument("--sequence-type", choices=PROBE_TYPES, help="手动指定 probe type。")
    parser.add_argument("--view", help="手动指定 probe view。")
    parser.add_argument("--seed", type=int, default=7, help="随机 probe 种子。")
    return parser.parse_args()


def ensure_cache(cache_path: Path, cfg_path: Path, checkpoint_iter: int) -> None:
    if cache_path.exists():
        return
    cmd = [
        str(ROOT / ".venvs" / "opengait-gpu" / "bin" / "python"),
        str(ROOT / "scripts" / "build_gait_gallery.py"),
        "--cfg-path",
        str(cfg_path),
        "--checkpoint-iter",
        str(checkpoint_iter),
        "--cache-path",
        str(cache_path),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache_path).resolve()
    ensure_cache(cache_path, Path(args.cfg_path), args.checkpoint_iter)
    cache = load_cache(cache_path)

    if args.subject and args.sequence_type and args.view:
        probe_index_value = find_probe_index(cache, args.subject, args.sequence_type, args.view)
    else:
        probe_index_value = random_probe_index(cache, seed=args.seed)

    result = retrieve_topk(cache, probe_index_value, topk=args.topk)
    probe = item_meta(cache, probe_index_value)
    print("=== Probe ===")
    print(probe_markdown(cache, probe, result["gallery_count"]))
    print("")
    print("=== Top-K ===")
    print("rank\tsubject\ttype\tview\tdistance\thit")
    for row in result["rows"]:
        print(
            f"{row['rank']}\t{row['subject_id']}\t{row['sequence_type']}\t{row['view']}\t"
            f"{row['distance']:.4f}\t{'hit' if row['hit'] else 'miss'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
