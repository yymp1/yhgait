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

from gait_demo_core import load_cache  # noqa: E402
from video_gait_core import process_real_video_demo, shutdown_distributed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实视频 -> silhouette -> gait retrieval 的最小 CLI 验证。")
    parser.add_argument("--video-path", required=True, help="输入视频路径。")
    parser.add_argument(
        "--gallery-cache-path",
        default=str(ROOT / "reports" / "gait_demo_cache_formal_conservative_real_iter01000.npz"),
        help="gallery cache 路径。",
    )
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"),
        help="OpenGait 评估配置。",
    )
    parser.add_argument("--checkpoint-iter", type=int, default=1000, help="用于 probe embedding 的 checkpoint iter。")
    parser.add_argument("--run-root", default=str(ROOT / "reports" / "real_video_demo_runs"), help="输出目录根。")
    parser.add_argument("--topk", type=int, default=5, help="Top-K。")
    parser.add_argument("--gallery-view", default="all", help="gallery 视角过滤，默认 all。")
    parser.add_argument("--expected-subject", default="", help="可选：如果你知道视频对应的 CASIA-B subject，可填入用于 hit/miss 标注。")
    parser.add_argument("--device", default="", help="视频检测设备，例如 cuda:0 或 cpu。默认自动。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gallery_cache = load_cache(Path(args.gallery_cache_path).resolve())
    try:
        session = process_real_video_demo(
            video_path=args.video_path,
            gallery_cache=gallery_cache,
            cfg_path=args.cfg_path,
            checkpoint_iter=args.checkpoint_iter,
            run_root=args.run_root,
            topk=args.topk,
            gallery_view=None if args.gallery_view.lower() == "all" else args.gallery_view,
            expected_subject=args.expected_subject or None,
            detection_device=args.device or None,
        )
    finally:
        shutdown_distributed()
    best_track = session["selected_track_id"]
    best_rows = []
    for track in session["tracks"]:
        if track["track_id"] == best_track:
            best_rows = track.get("rows", [])
            break
    summary = {
        "run_dir": session["run_dir"],
        "annotated_video": session["annotated_video"],
        "selected_track_id": best_track,
        "track_count": len(session["tracks"]),
        "best_rows": best_rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
