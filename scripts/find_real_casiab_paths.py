from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_ROOTS = [
    Path.cwd(),
    Path.home(),
    Path("/mnt"),
    Path("/media"),
    Path("/data"),
    Path("/srv"),
    Path("/opt"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="搜索 Linux 上的 CASIA-B 真实数据线索")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="要搜索的根目录列表",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="最大搜索深度，默认 8",
    )
    return parser


def walk_limited(root: Path, max_depth: int):
    root = root.resolve()
    if not root.exists():
        return
    base_depth = len(root.parts)
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        depth = len(current.parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        yield current, filenames


def summarize_pkl_root(root: Path) -> dict:
    subject_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    type_names = set()
    view_names = set()
    pkl_count = 0
    sample_paths: list[str] = []
    for subject_dir in subject_dirs:
        for type_dir in subject_dir.iterdir():
            if not type_dir.is_dir():
                continue
            type_names.add(type_dir.name)
            for view_dir in type_dir.iterdir():
                if not view_dir.is_dir():
                    continue
                view_names.add(view_dir.name)
                for pkl_path in sorted(view_dir.glob("*.pkl")):
                    pkl_count += 1
                    if len(sample_paths) < 5:
                        sample_paths.append(str(pkl_path))
    synthetic_like = len(subject_dirs) < 100 or pkl_count < 1000
    return {
        "path": str(root),
        "subject_count": len(subject_dirs),
        "type_count": len(type_names),
        "view_count": len(view_names),
        "pkl_count": pkl_count,
        "sample_paths": sample_paths,
        "classification": "synthetic_or_incomplete" if synthetic_like else "likely_real",
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = [Path(item).expanduser() for item in args.roots]
    found_pkl_roots: dict[str, dict] = {}
    found_raw_roots: set[str] = set()
    found_archives: set[str] = set()

    for root in roots:
        for current, filenames in walk_limited(root, args.max_depth):
            if current.name == "CASIA-B-pkl" and current.is_dir():
                key = str(current.resolve())
                if key not in found_pkl_roots:
                    try:
                        found_pkl_roots[key] = summarize_pkl_root(current)
                    except Exception as exc:
                        found_pkl_roots[key] = {
                            "path": key,
                            "classification": "error",
                            "error": repr(exc),
                        }
            if current.name == "casia_b" and current.is_dir():
                found_raw_roots.add(str(current.resolve()))
            for filename in filenames:
                if filename == "GaitDatasetB-silh.zip":
                    found_archives.add(str((current / filename).resolve()))

    result = {
        "searched_roots": [str(path.resolve()) for path in roots if path.exists()],
        "candidate_pkl_roots": sorted(found_pkl_roots.values(), key=lambda item: item["path"]),
        "candidate_raw_roots": sorted(found_raw_roots),
        "candidate_archives": sorted(found_archives),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
