from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

GALLERY_TYPES = ("nm-01", "nm-02", "nm-03", "nm-04")
PROBE_TYPES = ("nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02")
PROBE_GROUPS = {
    "NM": ("nm-05", "nm-06"),
    "BG": ("bg-01", "bg-02"),
    "CL": ("cl-01", "cl-02"),
}


def load_cache(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar_str(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def string_array(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def build_probe_key(label: str, seq_type: str, view: str) -> str:
    return f"{label} | {seq_type} | {view}"


def group_name_for_probe_type(seq_type: str) -> str:
    for group_name, group_types in PROBE_GROUPS.items():
        if seq_type in group_types:
            return group_name
    return "OTHER"


def probe_indices(cache: dict[str, Any]) -> np.ndarray:
    types = string_array(cache["types"])
    return np.where(np.isin(types, PROBE_TYPES))[0]


def gallery_indices_for_view(cache: dict[str, Any], view: str) -> np.ndarray:
    types = string_array(cache["types"])
    views = string_array(cache["views"])
    return np.where(np.isin(types, GALLERY_TYPES) & (views == view))[0]


def gallery_indices(cache: dict[str, Any], view: str | None = None) -> np.ndarray:
    types = string_array(cache["types"])
    mask = np.isin(types, GALLERY_TYPES)
    if view is not None and str(view).lower() != "all":
        views = string_array(cache["views"])
        mask = mask & (views == str(view))
    return np.where(mask)[0]


def list_gallery_views(cache: dict[str, Any]) -> list[str]:
    views = string_array(cache["views"])
    return sorted(set(views[gallery_indices(cache)]))


def list_probe_labels(cache: dict[str, Any]) -> list[str]:
    labels = string_array(cache["labels"])
    return sorted(set(labels[probe_indices(cache)]))


def list_probe_views(cache: dict[str, Any]) -> list[str]:
    views = string_array(cache["views"])
    return sorted(set(views[probe_indices(cache)]))


def item_meta(cache: dict[str, Any], index: int) -> dict[str, Any]:
    labels = string_array(cache["labels"])
    types = string_array(cache["types"])
    views = string_array(cache["views"])
    pkl_paths = string_array(cache["pkl_paths"])
    return {
        "index": int(index),
        "label": labels[index],
        "type": types[index],
        "view": views[index],
        "pkl_path": pkl_paths[index],
        "key": build_probe_key(labels[index], types[index], views[index]),
    }


def find_probe_index(cache: dict[str, Any], label: str, seq_type: str, view: str) -> int:
    labels = string_array(cache["labels"])
    types = string_array(cache["types"])
    views = string_array(cache["views"])
    indices = np.where((labels == label) & (types == seq_type) & (views == view))[0]
    valid = [idx for idx in indices if types[idx] in PROBE_TYPES]
    if not valid:
        raise KeyError(f"找不到 probe: {label} / {seq_type} / {view}")
    return int(valid[0])


def random_probe_index(cache: dict[str, Any], seed: int | None = None) -> int:
    rng = random.Random(seed)
    candidates = probe_indices(cache).tolist()
    if not candidates:
        raise RuntimeError("缓存中没有可用的 probe 样本。")
    return int(rng.choice(candidates))


def compute_average_bin_euclidean(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    if query.ndim != 2 or gallery.ndim != 3:
        raise ValueError(
            f"期望 query 为 [C, P]、gallery 为 [N, C, P]，实际得到 {query.shape} 和 {gallery.shape}"
        )
    diff = gallery - query[None, :, :]
    dist_per_bin = np.sqrt(np.maximum((diff ** 2).sum(axis=1), 0.0))
    return dist_per_bin.mean(axis=1)


def retrieve_topk(
    cache: dict[str, Any],
    probe_index_value: int,
    topk: int = 5,
) -> dict[str, Any]:
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    probe_meta = item_meta(cache, probe_index_value)
    gallery_idxs = gallery_indices_for_view(cache, probe_meta["view"])
    if len(gallery_idxs) == 0:
        raise RuntimeError(f"在视角 {probe_meta['view']} 下没有可用 gallery。")

    distances = compute_average_bin_euclidean(
        embeddings[probe_index_value], embeddings[gallery_idxs]
    )
    order = np.argsort(distances)[:topk]
    rows = []
    for rank, idx_in_gallery in enumerate(order, start=1):
        sample_index = int(gallery_idxs[idx_in_gallery])
        meta = item_meta(cache, sample_index)
        distance = float(distances[idx_in_gallery])
        rows.append(
            {
                "rank": rank,
                "subject_id": meta["label"],
                "sequence_type": meta["type"],
                "view": meta["view"],
                "distance": distance,
                "hit": meta["label"] == probe_meta["label"],
                "pkl_path": meta["pkl_path"],
                "key": meta["key"],
            }
        )
    return {
        "probe": probe_meta,
        "rows": rows,
        "gallery_count": int(len(gallery_idxs)),
    }


def retrieve_embedding_topk(
    cache: dict[str, Any],
    query_embedding: np.ndarray,
    *,
    topk: int = 5,
    gallery_view: str | None = None,
    expected_subject: str | None = None,
    probe_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    gallery_idxs = gallery_indices(cache, gallery_view)
    if len(gallery_idxs) == 0:
        raise RuntimeError(f"当前 gallery 过滤条件下没有可用候选: view={gallery_view!r}")

    distances = compute_average_bin_euclidean(query_embedding, embeddings[gallery_idxs])
    order = np.argsort(distances)[:topk]
    rows = []
    for rank, idx_in_gallery in enumerate(order, start=1):
        sample_index = int(gallery_idxs[idx_in_gallery])
        meta = item_meta(cache, sample_index)
        distance = float(distances[idx_in_gallery])
        hit = None if not expected_subject else (meta["label"] == str(expected_subject))
        rows.append(
            {
                "rank": rank,
                "subject_id": meta["label"],
                "sequence_type": meta["type"],
                "view": meta["view"],
                "distance": distance,
                "hit": hit,
                "pkl_path": meta["pkl_path"],
                "key": meta["key"],
            }
        )
    return {
        "probe": probe_meta or {},
        "rows": rows,
        "gallery_count": int(len(gallery_idxs)),
        "gallery_view": "all" if gallery_view is None else str(gallery_view),
        "expected_subject": "" if expected_subject is None else str(expected_subject),
    }


def hit_rank(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if row["hit"]:
            return int(row["rank"])
    return None


def analysis_case_from_result(result: dict[str, Any]) -> dict[str, Any]:
    probe = result["probe"]
    rows = result["rows"]
    best_hit_rank = hit_rank(rows)
    normalized_rows = [
        {
            "rank": int(row["rank"]),
            "subject_id": str(row["subject_id"]),
            "sequence_type": str(row["sequence_type"]),
            "view": str(row["view"]),
            "distance": float(row["distance"]),
            "hit": bool(row["hit"]),
            "pkl_path": str(row["pkl_path"]),
            "key": str(row["key"]),
        }
        for row in rows
    ]
    top1_distance = normalized_rows[0]["distance"] if normalized_rows else None
    return {
        "probe_key": str(probe["key"]),
        "probe": {
            "index": int(probe["index"]),
            "label": str(probe["label"]),
            "type": str(probe["type"]),
            "view": str(probe["view"]),
            "pkl_path": str(probe["pkl_path"]),
            "key": str(probe["key"]),
        },
        "probe_group": group_name_for_probe_type(str(probe["type"])),
        "gallery_count": int(result["gallery_count"]),
        "top1_hit": bool(normalized_rows and normalized_rows[0]["hit"]),
        "top5_hit": any(row["hit"] for row in normalized_rows),
        "best_hit_rank": best_hit_rank,
        "top1_subject": normalized_rows[0]["subject_id"] if normalized_rows else None,
        "top1_distance": top1_distance,
        "rows": normalized_rows,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    top1_hits = sum(1 for case in cases if case["top1_hit"])
    top5_hits = sum(1 for case in cases if case["top5_hit"])
    return {
        "count": count,
        "top1_hits": top1_hits,
        "top5_hits": top5_hits,
        "top1_rate": float(top1_hits / count) if count else 0.0,
        "top5_rate": float(top5_hits / count) if count else 0.0,
    }


def sort_case_keys(cases: list[dict[str, Any]], mode: str) -> list[str]:
    if mode == "success":
        filtered = [case for case in cases if case["top1_hit"]]
        filtered.sort(
            key=lambda case: (
                case["top1_distance"] if case["top1_distance"] is not None else 999.0,
                case["probe_key"],
            )
        )
    elif mode == "failure":
        filtered = [case for case in cases if not case["top1_hit"]]
        filtered.sort(
            key=lambda case: (
                case["top5_hit"],
                case["best_hit_rank"] if case["best_hit_rank"] is not None else 999,
                case["top1_distance"] if case["top1_distance"] is not None else 999.0,
                case["probe_key"],
            )
        )
    else:
        raise ValueError(f"不支持的 case 排序模式: {mode}")
    return [case["probe_key"] for case in filtered]


def analyze_cache(cache: dict[str, Any], topk: int = 5) -> dict[str, Any]:
    cases = [
        analysis_case_from_result(retrieve_topk(cache, int(probe_idx), topk=topk))
        for probe_idx in probe_indices(cache).tolist()
    ]
    overall = summarize_cases(cases)
    group_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        group_cases[case["probe_group"]].append(case)
    groups = {
        group_name: summarize_cases(group_cases.get(group_name, []))
        for group_name in PROBE_GROUPS
    }
    return {
        "checkpoint_path": scalar_str(cache["checkpoint_path"]),
        "dataset_root": scalar_str(cache["dataset_root"]),
        "dataset_partition": scalar_str(cache["dataset_partition"]),
        "save_name": scalar_str(cache["save_name"]),
        "restore_hint": scalar_str(cache["restore_hint"]),
        "metric": scalar_str(cache["metric"]),
        "topk": int(topk),
        "gallery_types": list(GALLERY_TYPES),
        "probe_types": list(PROBE_TYPES),
        "probe_groups": {name: list(values) for name, values in PROBE_GROUPS.items()},
        "protocol": "CASIA-B 单视角 gallery：gallery 使用 nm-01~04，probe 使用 nm-05/06、bg-01/02、cl-01/02，并要求与 probe 同视角。",
        "probe_count": int(len(cases)),
        "overall": overall,
        "groups": groups,
        "case_lists": {
            "success_top1": sort_case_keys(cases, "success"),
            "failure_top1": sort_case_keys(cases, "failure"),
        },
        "cases": cases,
    }


def case_map(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["probe_key"]: case for case in analysis["cases"]}


def comparison_summary(
    baseline_analysis: dict[str, Any],
    target_analysis: dict[str, Any],
) -> dict[str, Any]:
    baseline_lookup = case_map(baseline_analysis)
    target_lookup = case_map(target_analysis)
    shared_keys = sorted(set(baseline_lookup) & set(target_lookup))
    pairs = []
    improved_top1 = 0
    regressed_top1 = 0
    improved_top5 = 0
    regressed_top5 = 0
    unchanged_top1 = 0
    for probe_key in shared_keys:
        base_case = baseline_lookup[probe_key]
        target_case = target_lookup[probe_key]
        if (not base_case["top1_hit"]) and target_case["top1_hit"]:
            improved_top1 += 1
        elif base_case["top1_hit"] and (not target_case["top1_hit"]):
            regressed_top1 += 1
        else:
            unchanged_top1 += 1
        if (not base_case["top5_hit"]) and target_case["top5_hit"]:
            improved_top5 += 1
        elif base_case["top5_hit"] and (not target_case["top5_hit"]):
            regressed_top5 += 1
        pairs.append(
            {
                "probe_key": probe_key,
                "probe": target_case["probe"],
                "probe_group": target_case["probe_group"],
                "baseline_top1_hit": bool(base_case["top1_hit"]),
                "baseline_top5_hit": bool(base_case["top5_hit"]),
                "baseline_top1_distance": base_case["top1_distance"],
                "target_top1_hit": bool(target_case["top1_hit"]),
                "target_top5_hit": bool(target_case["top5_hit"]),
                "target_top1_distance": target_case["top1_distance"],
            }
        )
    return {
        "probe_count": len(shared_keys),
        "improved_top1": improved_top1,
        "regressed_top1": regressed_top1,
        "unchanged_top1": unchanged_top1,
        "improved_top5": improved_top5,
        "regressed_top5": regressed_top5,
        "pairs": pairs,
    }


def load_sequence_array(pkl_path: str | Path) -> np.ndarray:
    with Path(pkl_path).open("rb") as fh:
        seq = pickle.load(fh)
    arr = np.asarray(seq)
    if arr.ndim != 3:
        raise ValueError(f"期望 silhouette 序列为 [T, H, W]，实际得到 {arr.shape}")
    return arr.astype(np.uint8)


def make_sequence_strip(
    pkl_path: str | Path,
    num_frames: int = 6,
    scale: int = 3,
    border: int = 2,
) -> Image.Image:
    sequence = load_sequence_array(pkl_path)
    return make_sequence_strip_from_array(sequence, num_frames=num_frames, scale=scale, border=border)


def make_sequence_strip_from_array(
    sequence: np.ndarray,
    num_frames: int = 6,
    scale: int = 3,
    border: int = 2,
) -> Image.Image:
    frame_count = sequence.shape[0]
    sample_count = min(num_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int)
    tiles = []
    for idx in indices:
        frame = Image.fromarray(sequence[idx]).convert("L")
        frame = frame.resize(
            (frame.width * scale, frame.height * scale),
            resample=Image.Resampling.NEAREST,
        )
        frame = ImageOps.expand(frame, border=border, fill=48)
        tiles.append(frame)

    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    canvas = Image.new("L", (width, height), color=0)
    offset = 0
    for tile in tiles:
        canvas.paste(tile, (offset, 0))
        offset += tile.width
    return canvas.convert("RGB")


def probe_markdown(cache: dict[str, Any], probe_meta: dict[str, Any], gallery_count: int) -> str:
    checkpoint = scalar_str(cache["checkpoint_path"])
    dataset_root = scalar_str(cache["dataset_root"])
    return "\n".join(
        [
            f"**Checkpoint**: `{checkpoint}`",
            f"**数据路径**: `{dataset_root}`",
            f"**Probe**: `{probe_meta['label']} / {probe_meta['type']} / {probe_meta['view']}`",
            f"**检索协议**: 官方 CASIA-B 单视角 gallery",
            f"gallery 条目数: `{gallery_count}`",
            "score 为平均欧式距离，越小越相似。",
        ]
    )
