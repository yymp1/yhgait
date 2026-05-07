from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gait_demo_core import compute_average_bin_euclidean


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def empty_gallery_index() -> dict[str, Any]:
    timestamp = now_text()
    return {
        "version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "identities": {},
        "samples": {},
    }


def ensure_private_gallery_root(gallery_root: str | Path) -> Path:
    root = Path(gallery_root).resolve()
    (root / "identities").mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    if not index_path.exists():
        index_path.write_text(
            json.dumps(empty_gallery_index(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return root


def load_private_gallery_index(gallery_root: str | Path) -> dict[str, Any]:
    root = ensure_private_gallery_root(gallery_root)
    index_path = root / "index.json"
    with index_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data.get("identities"), list):
        legacy_identities = data.get("identities", [])
        identities: dict[str, Any] = {}
        samples: dict[str, Any] = {}
        for entry in legacy_identities:
            identity = str(entry.get("identity", "")).strip()
            if not identity:
                continue
            identity_samples = entry.get("samples", []) or []
            sample_ids = []
            for sample in identity_samples:
                sample_id = str(sample.get("sample_id", "")).strip()
                if not sample_id:
                    continue
                sample_record = {
                    "sample_id": sample_id,
                    "identity": identity,
                    "created_at": sample.get("created_at", entry.get("created_at", now_text())),
                    "input_video": sample.get("source_video", ""),
                    "run_dir": sample.get("run_dir", ""),
                    "track_id": sample.get("track_id", ""),
                    "frame_count": int(sample.get("frame_count", 0)),
                    "valid_frames": int(sample.get("valid_frames", 0)),
                    "used_frames": int(sample.get("used_frames", 0)),
                    "avg_bbox_area": float(sample.get("avg_bbox_area", 0.0)),
                    "avg_confidence": float(sample.get("avg_confidence", 0.0)),
                    "embedding_path": sample.get("embedding_path", ""),
                    "silhouette_pkl_path": sample.get("silhouette_pkl_path", ""),
                    "silhouette_strip_path": sample.get("silhouette_strip_path", ""),
                    "track_clip_path": sample.get("track_clip_path", ""),
                    "note": sample.get("note", ""),
                }
                samples[sample_id] = sample_record
                sample_ids.append(sample_id)
            identities[identity] = {
                "identity": identity,
                "dir_name": entry.get("dir_name", identity_dir_name(identity)),
                "created_at": entry.get("created_at", now_text()),
                "updated_at": entry.get("updated_at", now_text()),
                "prototype_method": "mean",
                "prototype_path": entry.get("aggregate_embedding_path", ""),
                "representative_sample_id": entry.get("representative_sample_id", ""),
                "sample_count": int(entry.get("sample_count", len(sample_ids))),
                "sample_ids": sample_ids,
            }
        data["identities"] = identities
        data["samples"] = samples
    else:
        data.setdefault("identities", {})
        data.setdefault("samples", {})
    return data


def save_private_gallery_index(gallery_root: str | Path, index: dict[str, Any]) -> None:
    root = ensure_private_gallery_root(gallery_root)
    index["updated_at"] = now_text()
    (root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def path_to_rel(gallery_root: Path, path_text: str | Path | None) -> str:
    if not path_text:
        return ""
    path = Path(path_text).resolve()
    try:
        return str(path.relative_to(gallery_root))
    except ValueError:
        return str(path)


def rel_to_path(gallery_root: Path, path_text: str | Path | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (gallery_root / path).resolve()


def identity_dir_name(identity: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in identity).strip("_")
    prefix = cleaned[:32] or "identity"
    suffix = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}__{suffix}"


def sample_id_for_track(identity: str, track_id: str, input_video: str) -> str:
    suffix = hashlib.sha1(f"{identity}|{track_id}|{input_video}|{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:8]
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}_{suffix}"


def get_track(session: dict[str, Any], track_id: str) -> dict[str, Any]:
    track = next((item for item in session.get("tracks", []) if item.get("track_id") == track_id), None)
    if track is None:
        raise KeyError(f"找不到 track: {track_id}")
    return track


def ensure_identity(index: dict[str, Any], identity: str) -> dict[str, Any]:
    identities = index.setdefault("identities", {})
    if identity not in identities:
        timestamp = now_text()
        identities[identity] = {
            "identity": identity,
            "dir_name": identity_dir_name(identity),
            "created_at": timestamp,
            "updated_at": timestamp,
            "prototype_method": "mean",
            "prototype_path": "",
            "representative_sample_id": "",
            "sample_count": 0,
            "sample_ids": [],
        }
    return identities[identity]


def copy_if_exists(src_text: str | Path | None, dst: Path) -> str:
    if not src_text:
        return ""
    src = Path(src_text)
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def load_sample_embedding(gallery_root: Path, sample_meta: dict[str, Any]) -> np.ndarray | None:
    embedding_path = rel_to_path(gallery_root, sample_meta.get("embedding_path"))
    if embedding_path is None or not embedding_path.exists():
        return None
    return np.asarray(np.load(embedding_path), dtype=np.float32)


def recompute_identity(index: dict[str, Any], gallery_root: str | Path, identity: str) -> dict[str, Any] | None:
    root = ensure_private_gallery_root(gallery_root)
    identity_meta = index.get("identities", {}).get(identity)
    if identity_meta is None:
        return None

    valid_sample_ids: list[str] = []
    sample_embeddings: list[np.ndarray] = []
    sample_metas: list[dict[str, Any]] = []
    for sample_id in identity_meta.get("sample_ids", []):
        sample_meta = index.get("samples", {}).get(sample_id)
        if sample_meta is None:
            continue
        embedding = load_sample_embedding(root, sample_meta)
        if embedding is None:
            continue
        valid_sample_ids.append(sample_id)
        sample_embeddings.append(embedding)
        sample_metas.append(sample_meta)

    if not valid_sample_ids:
        identity_dir = root / "identities" / identity_meta["dir_name"]
        for sample_id in identity_meta.get("sample_ids", []):
            index.get("samples", {}).pop(sample_id, None)
        if identity_dir.exists():
            shutil.rmtree(identity_dir)
        index["identities"].pop(identity, None)
        return None

    prototype = np.mean(np.stack(sample_embeddings, axis=0), axis=0).astype(np.float32)
    identity_dir = root / "identities" / identity_meta["dir_name"]
    identity_dir.mkdir(parents=True, exist_ok=True)
    prototype_path = identity_dir / "prototype.npy"
    np.save(prototype_path, prototype)

    distances = compute_average_bin_euclidean(prototype, np.stack(sample_embeddings, axis=0))
    rep_index = int(np.argmin(distances))
    identity_meta.update(
        {
            "updated_at": now_text(),
            "prototype_path": path_to_rel(root, prototype_path),
            "representative_sample_id": valid_sample_ids[rep_index],
            "sample_count": len(valid_sample_ids),
            "sample_ids": valid_sample_ids,
        }
    )
    return identity_meta


def rebuild_private_gallery(gallery_root: str | Path) -> dict[str, Any]:
    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    for identity in list(index.get("identities", {}).keys()):
        recompute_identity(index, root, identity)
    save_private_gallery_index(root, index)
    return index


def enroll_track_to_private_gallery(
    session: dict[str, Any],
    track_id: str,
    identity: str,
    gallery_root: str | Path,
    note: str = "",
) -> dict[str, Any]:
    identity = identity.strip()
    if not identity:
        raise ValueError("identity 不能为空。")

    track = get_track(session, track_id)
    if track.get("status") != "ok":
        raise ValueError(f"track {track_id} 当前不可录入，status={track.get('status')}")
    if not track.get("embedding_path"):
        raise ValueError(f"track {track_id} 缺少 embedding 产物。")

    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    identity_meta = ensure_identity(index, identity)
    sample_id = sample_id_for_track(identity, track_id, session.get("input_video", "video"))

    sample_dir = root / "identities" / identity_meta["dir_name"] / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    embedding_dst = sample_dir / "embedding.npy"
    silhouette_pkl_dst = sample_dir / "silhouettes.pkl"
    silhouette_strip_dst = sample_dir / "silhouette_strip.png"
    track_clip_dst = sample_dir / "track_clip.mp4"

    copy_if_exists(track.get("embedding_path"), embedding_dst)
    copy_if_exists(track.get("silhouette_pkl"), silhouette_pkl_dst)
    copy_if_exists(track.get("silhouette_strip"), silhouette_strip_dst)
    copy_if_exists(track.get("clip_path"), track_clip_dst)

    sample_meta = {
        "sample_id": sample_id,
        "identity": identity,
        "created_at": now_text(),
        "input_video": str(session.get("input_video", "")),
        "run_dir": str(session.get("run_dir", "")),
        "track_id": track_id,
        "frame_count": int(track.get("frame_count", 0)),
        "valid_frames": int(track.get("valid_frames", 0)),
        "used_frames": int(track.get("used_frames", 0)),
        "avg_bbox_area": float(track.get("avg_bbox_area", 0.0)),
        "avg_confidence": float(track.get("avg_confidence", 0.0)),
        "embedding_path": path_to_rel(root, embedding_dst),
        "silhouette_pkl_path": path_to_rel(root, silhouette_pkl_dst),
        "silhouette_strip_path": path_to_rel(root, silhouette_strip_dst),
        "track_clip_path": path_to_rel(root, track_clip_dst),
        "note": note.strip(),
    }
    (sample_dir / "sample_meta.json").write_text(
        json.dumps(sample_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index["samples"][sample_id] = sample_meta
    if sample_id not in identity_meta["sample_ids"]:
        identity_meta["sample_ids"].append(sample_id)
    recompute_identity(index, root, identity)
    save_private_gallery_index(root, index)
    return index["samples"][sample_id]


def delete_identity_from_private_gallery(gallery_root: str | Path, identity: str) -> dict[str, Any]:
    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    identity_meta = index.get("identities", {}).get(identity)
    if identity_meta is None:
        raise KeyError(f"找不到 identity: {identity}")

    for sample_id in identity_meta.get("sample_ids", []):
        index.get("samples", {}).pop(sample_id, None)
    identity_dir = root / "identities" / identity_meta["dir_name"]
    if identity_dir.exists():
        shutil.rmtree(identity_dir)
    index["identities"].pop(identity, None)
    save_private_gallery_index(root, index)
    return index


def list_identity_rows(gallery_root: str | Path) -> list[dict[str, Any]]:
    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    rows = []
    for identity, meta in sorted(index.get("identities", {}).items()):
        rep_sample = index.get("samples", {}).get(meta.get("representative_sample_id", ""), {})
        rows.append(
            {
                "identity": identity,
                "sample_count": int(meta.get("sample_count", 0)),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "prototype_method": meta.get("prototype_method", "mean"),
                "representative_sample_id": meta.get("representative_sample_id", ""),
                "representative_strip_path": str(rel_to_path(root, rep_sample.get("silhouette_strip_path")) or ""),
            }
        )
    return rows


def list_sample_rows(gallery_root: str | Path, identity: str | None = None) -> list[dict[str, Any]]:
    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    rows = []
    for sample_id, meta in sorted(index.get("samples", {}).items()):
        if identity and meta.get("identity") != identity:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "identity": meta.get("identity", ""),
                "created_at": meta.get("created_at", ""),
                "track_id": meta.get("track_id", ""),
                "frame_count": int(meta.get("frame_count", 0)),
                "used_frames": int(meta.get("used_frames", 0)),
                "input_video": Path(meta.get("input_video", "")).name,
                "track_clip_path": str(rel_to_path(root, meta.get("track_clip_path")) or ""),
                "silhouette_strip_path": str(rel_to_path(root, meta.get("silhouette_strip_path")) or ""),
                "note": meta.get("note", ""),
            }
        )
    return rows


def private_gallery_identity_choices(gallery_root: str | Path) -> list[str]:
    return [row["identity"] for row in list_identity_rows(gallery_root)]


def assess_review(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not rows:
        return True, ["私有步态库为空。"]

    top1 = rows[0]
    if len(rows) < 2:
        reasons.append("当前私有库身份数不足 2 个。")
    else:
        gap = float(rows[1]["distance"]) - float(top1["distance"])
        if gap < 0.03:
            reasons.append("Top-1 与 Top-2 的距离差小于 0.03。")

    if int(top1.get("sample_count", 0)) < 2:
        reasons.append("Top-1 身份当前样本数少于 2。")
    if float(top1.get("distance", 1.0)) > 0.65:
        reasons.append("Top-1 距离偏大。")
    return bool(reasons), reasons


def match_private_gallery(
    query_embedding: np.ndarray,
    gallery_root: str | Path,
    *,
    topk: int = 5,
) -> dict[str, Any]:
    root = ensure_private_gallery_root(gallery_root)
    index = load_private_gallery_index(root)
    identity_metas = sorted(index.get("identities", {}).items())
    if not identity_metas:
        raise RuntimeError("当前私有步态库为空，请先录入至少一个 identity。")

    rows = []
    for identity, meta in identity_metas:
        prototype_path = rel_to_path(root, meta.get("prototype_path"))
        if prototype_path is None or not prototype_path.exists():
            continue
        prototype = np.asarray(np.load(prototype_path), dtype=np.float32)
        prototype_distance = float(
            compute_average_bin_euclidean(
                np.asarray(query_embedding, dtype=np.float32),
                prototype[None, :, :],
            )[0]
        )

        sample_metas = [
            index["samples"][sample_id]
            for sample_id in meta.get("sample_ids", [])
            if sample_id in index.get("samples", {})
        ]
        sample_embeddings = []
        valid_sample_metas = []
        for sample_meta in sample_metas:
            embedding = load_sample_embedding(root, sample_meta)
            if embedding is None:
                continue
            sample_embeddings.append(embedding)
            valid_sample_metas.append(sample_meta)

        best_sample_distance = prototype_distance
        best_sample_meta = {}
        if sample_embeddings:
            sample_distances = compute_average_bin_euclidean(
                np.asarray(query_embedding, dtype=np.float32),
                np.stack(sample_embeddings, axis=0),
            )
            best_index = int(np.argmin(sample_distances))
            best_sample_distance = float(sample_distances[best_index])
            best_sample_meta = valid_sample_metas[best_index]

        representative_sample = index.get("samples", {}).get(meta.get("representative_sample_id", ""), {})
        preview_path = rel_to_path(root, representative_sample.get("silhouette_strip_path"))
        track_clip_path = rel_to_path(root, representative_sample.get("track_clip_path"))
        rows.append(
            {
                "identity": identity,
                "distance": prototype_distance,
                "score": float(1.0 / (1.0 + prototype_distance)),
                "best_sample_distance": best_sample_distance,
                "sample_count": int(meta.get("sample_count", 0)),
                "representative_sample_id": meta.get("representative_sample_id", ""),
                "best_sample_id": best_sample_meta.get("sample_id", ""),
                "preview_path": str(preview_path) if preview_path else "",
                "track_clip_path": str(track_clip_path) if track_clip_path else "",
            }
        )

    rows.sort(key=lambda row: (row["distance"], row["best_sample_distance"], row["identity"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    top_rows = rows[:topk]
    review_required, review_reasons = assess_review(top_rows)
    return {
        "rows": top_rows,
        "identity_count": len(rows),
        "top1_identity": top_rows[0]["identity"] if top_rows else "",
        "top1_distance": top_rows[0]["distance"] if top_rows else None,
        "review_required": review_required,
        "review_reasons": review_reasons,
    }


def attach_private_gallery_matches(
    session: dict[str, Any],
    gallery_root: str | Path,
    *,
    topk: int = 5,
) -> dict[str, Any]:
    for track in session.get("tracks", []):
        if track.get("status") != "ok" or not track.get("embedding_path"):
            continue
        embedding = np.asarray(np.load(track["embedding_path"]), dtype=np.float32)
        result = match_private_gallery(embedding, gallery_root, topk=topk)
        track.update(
            {
                "rows": result["rows"],
                "gallery_count": int(result["identity_count"]),
                "top1_identity": result["top1_identity"],
                "top1_distance": result["top1_distance"],
                "review_required": result["review_required"],
                "review_reasons": result["review_reasons"],
            }
        )
    session["mode"] = "private_gallery_recognition"
    return session
