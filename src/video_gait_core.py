from __future__ import annotations

import json
import os
import pickle
import socket
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
OPENGAIT_PY_ROOT = OPENGAIT_ROOT / "opengait"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OPENGAIT_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENGAIT_PY_ROOT))

import app as video_app  # noqa: E402
from data.collate_fn import CollateFn  # noqa: E402
from modeling import models  # noqa: E402
from utils import config_loader, get_ddp_module, get_msg_mgr, init_seeds, params_count, ts2np  # noqa: E402

from gait_demo_core import make_sequence_strip_from_array, retrieve_embedding_topk
from private_gallery_core import load_private_gallery_index, match_private_gallery


@dataclass(slots=True)
class VideoSilhouetteConfig:
    threshold: float = 0.55
    output_size: int = 64
    min_mask_ratio: float = 0.03
    min_valid_frames: int = 16


def ensure_single_process_env(master_port: int) -> None:
    def pick_master_port(preferred_port: int) -> int:
        if os.environ.get("MASTER_PORT"):
            return int(os.environ["MASTER_PORT"])
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", int(preferred_port)))
            return int(preferred_port)
        except OSError:
            pass
        finally:
            probe.close()

        fallback = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            fallback.bind(("127.0.0.1", 0))
            return int(fallback.getsockname()[1])
        finally:
            fallback.close()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(pick_master_port(master_port)))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")


def preferred_distributed_backend() -> str:
    if not torch.cuda.is_available():
        return "gloo"
    if sys.platform.startswith("win"):
        return "gloo"
    if not torch.distributed.is_nccl_available():
        return "gloo"
    return "nccl"


def resolve_from_opengait_root(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((OPENGAIT_ROOT / path).resolve())


def load_cfg(cfg_path: Path, checkpoint_iter: int) -> dict:
    previous_cwd = Path.cwd()
    os.chdir(OPENGAIT_ROOT)
    try:
        cfgs = config_loader(str(cfg_path.resolve()))
    finally:
        os.chdir(previous_cwd)
    cfgs["data_cfg"]["dataset_root"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_root"])
    cfgs["data_cfg"]["dataset_partition"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_partition"])
    cfgs["evaluator_cfg"]["restore_hint"] = checkpoint_iter
    cfgs["trainer_cfg"]["restore_hint"] = checkpoint_iter
    return cfgs


def init_logger(cfgs: dict) -> None:
    msg_mgr = get_msg_mgr()
    output_path = (
        OPENGAIT_ROOT
        / "output"
        / cfgs["data_cfg"]["dataset_name"]
        / cfgs["model_cfg"]["model"]
        / cfgs["evaluator_cfg"]["save_name"]
    )
    msg_mgr.init_logger(str(output_path), False)


class OpenGaitFeatureExtractor:
    def __init__(self, cfg_path: str | Path, checkpoint_iter: int, master_port: int = 29741) -> None:
        self.cfg_path = Path(cfg_path).resolve()
        self.checkpoint_iter = int(checkpoint_iter)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "当前私有步态库 demo 的 OpenGait 特征提取需要可用 CUDA GPU。"
                " 如果你只想验证基础检测/跟踪流程，请改用 app.py；"
                " 如果要跑完整录入/识别 demo，请先安装支持 CUDA 的 PyTorch。"
            )
        ensure_single_process_env(master_port)
        backend = preferred_distributed_backend()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend, init_method="env://")

        previous_cwd = Path.cwd()
        os.chdir(OPENGAIT_ROOT)
        try:
            self.cfgs = load_cfg(self.cfg_path, self.checkpoint_iter)
            init_logger(self.cfgs)
            init_seeds(torch.distributed.get_rank())
            model_cfg = self.cfgs["model_cfg"]
            msg_mgr = get_msg_mgr()
            msg_mgr.log_info(model_cfg)
            model_class = getattr(models, model_cfg["model"])
            model = model_class(self.cfgs, training=False)
            self.model = get_ddp_module(model, self.cfgs["trainer_cfg"]["find_unused_parameters"])
            msg_mgr.log_info(params_count(self.model))
            self.collate_fn = CollateFn(["video_probe"], self.cfgs["evaluator_cfg"]["sampler"])
        finally:
            os.chdir(previous_cwd)

    def embed_sequence(self, sequence: np.ndarray, *, seq_type: str = "video-probe", view: str = "video") -> np.ndarray:
        batch = self.collate_fn([([np.asarray(sequence, dtype=np.uint8)], ["video_probe", seq_type, view])])
        with torch.no_grad():
            ipts = self.model.inputs_pretreament(batch)
            with autocast(enabled=self.cfgs["evaluator_cfg"]["enable_float16"]):
                retval = self.model(ipts)
            embedding = ts2np(retval["inference_feat"]["embeddings"])[0]
        return np.asarray(embedding, dtype=np.float32)


@lru_cache(maxsize=2)
def get_feature_extractor(cfg_path_text: str, checkpoint_iter: int) -> OpenGaitFeatureExtractor:
    return OpenGaitFeatureExtractor(cfg_path_text, checkpoint_iter)


def shutdown_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class VideoSilhouetteExtractor:
    def __init__(self, config: VideoSilhouetteConfig | None = None) -> None:
        self.config = config or VideoSilhouetteConfig()
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def close(self) -> None:
        self.segmenter.close()

    def build_track_sequence(
        self,
        track_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        frame_files = sorted(track_dir.glob("frame_*.jpg"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if not frame_files:
            return {
                "status": "no_frames",
                "quality_note": "no_crop_frames",
                "frame_count": 0,
                "valid_frames": 0,
                "used_frames": 0,
                "silhouette_pkl": "",
                "silhouette_strip": "",
            }

        processed_frames = []
        valid_entries = []
        for frame_path in frame_files:
            frame_index = parse_frame_index(frame_path)
            image = cv2.imread(str(frame_path))
            if image is None:
                processed_frames.append({"frame_index": frame_index, "valid": False})
                continue
            mask = self.extract_silhouette(image)
            if mask is None:
                processed_frames.append({"frame_index": frame_index, "valid": False})
                continue
            processed_frames.append({"frame_index": frame_index, "valid": True, "mask": mask})
            valid_entries.append({"frame_index": frame_index, "mask": mask})

        longest_run = longest_valid_run(processed_frames)
        if not longest_run:
            return {
                "status": "no_valid_silhouette",
                "quality_note": "segmentation_failed",
                "frame_count": len(frame_files),
                "valid_frames": len(valid_entries),
                "used_frames": 0,
                "silhouette_pkl": "",
                "silhouette_strip": "",
            }

        sequence = np.asarray([entry["mask"] for entry in longest_run], dtype=np.uint8)
        if len(sequence) < self.config.min_valid_frames:
            return {
                "status": "too_short",
                "quality_note": f"valid_run_too_short<{self.config.min_valid_frames}",
                "frame_count": len(frame_files),
                "valid_frames": len(valid_entries),
                "used_frames": int(len(sequence)),
                "silhouette_pkl": "",
                "silhouette_strip": "",
            }

        pkl_path = output_dir / "silhouettes.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(sequence, fh)
        strip_path = output_dir / "silhouette_strip.png"
        make_sequence_strip_from_array(sequence).save(strip_path)
        preview_dir = output_dir / "preview_frames"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for offset, entry in enumerate(longest_run[: min(12, len(longest_run))], start=1):
            Image.fromarray(entry["mask"]).save(preview_dir / f"{offset:03d}.png")

        return {
            "status": "ok",
            "quality_note": "ok",
            "frame_count": len(frame_files),
            "valid_frames": len(valid_entries),
            "used_frames": int(len(sequence)),
            "silhouette_pkl": str(pkl_path),
            "silhouette_strip": str(strip_path),
            "sequence": sequence,
        }

    def extract_silhouette(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        mask_prob = getattr(result, "segmentation_mask", None)
        if mask_prob is None:
            return None
        mask = (mask_prob >= self.config.threshold).astype(np.uint8) * 255
        mask = largest_connected_component(mask)
        if mask is None:
            return None
        area_ratio = float(mask.sum() / 255.0) / float(mask.shape[0] * mask.shape[1])
        if area_ratio < self.config.min_mask_ratio:
            return None
        return normalize_silhouette(mask, self.config.output_size)


def largest_connected_component(mask: np.ndarray) -> np.ndarray | None:
    if mask is None or mask.size == 0 or int(mask.max()) == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    component_ids = list(range(1, num_labels))
    component_ids.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    best_id = component_ids[0]
    filtered = np.where(labels == best_id, 255, 0).astype(np.uint8)
    return filtered if int(filtered.max()) > 0 else None


def normalize_silhouette(mask: np.ndarray, output_size: int = 64) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    cropped = mask[y1:y2, x1:x2]
    h, w = cropped.shape
    scale = min(output_size / max(w, 1), output_size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    x_offset = (output_size - new_w) // 2
    y_offset = output_size - new_h
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def parse_frame_index(path: Path) -> int:
    stem = path.stem
    if "_" in stem:
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0
    return 0


def longest_valid_run(processed_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_index: int | None = None
    for entry in processed_frames:
        if not entry.get("valid"):
            if len(current) > len(best):
                best = current
            current = []
            last_index = None
            continue
        frame_index = int(entry["frame_index"])
        if last_index is None or frame_index == last_index + 1:
            current.append(entry)
        else:
            if len(current) > len(best):
                best = current
            current = [entry]
        last_index = frame_index
    if len(current) > len(best):
        best = current
    return best


def safe_stem(path: str | Path) -> str:
    source = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source)
    return cleaned.strip("_") or "video"


def default_detection_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_run_dir(base_dir: Path, source_video: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{safe_stem(source_video)}_{timestamp}"


def build_track_record(summary: Any) -> dict[str, Any]:
    track_id = f"{summary.track_id:04d}"
    return {
        "track_id": track_id,
        "status": "skipped",
        "quality_note": "export_filtered",
        "frame_count": int(summary.frame_count),
        "avg_bbox_area": float(summary.avg_bbox_area),
        "avg_confidence": float(summary.avg_confidence),
        "first_frame_idx": summary.first_frame_idx,
        "last_frame_idx": summary.last_frame_idx,
        "crop_dir": str(summary.crop_dir) if summary.crop_dir else "",
        "clip_path": str(summary.clip_path) if summary.clip_path else "",
        "silhouette_pkl": "",
        "silhouette_strip": "",
        "embedding_path": "",
        "used_frames": 0,
        "valid_frames": 0,
        "rows": [],
        "gallery_count": 0,
        "top1_subject": "",
        "top1_identity": "",
        "top1_distance": None,
        "review_required": False,
        "review_reasons": [],
    }


def choose_best_track(tracks: list[dict[str, Any]]) -> str | None:
    candidates = [track for track in tracks if track["status"] == "ok"]
    if not candidates:
        candidates = tracks
    if not candidates:
        return None
    candidates.sort(
        key=lambda track: (
            int(track.get("used_frames", 0)),
            int(track.get("frame_count", 0)),
            float(track.get("avg_bbox_area", 0.0)),
            float(track.get("avg_confidence", 0.0)),
        ),
        reverse=True,
    )
    return str(candidates[0]["track_id"])


def process_video_to_tracks(
    *,
    video_path: str | Path,
    cfg_path: str | Path,
    checkpoint_iter: int,
    run_root: str | Path,
    detection_device: str | None = None,
    yolo_model: str = "yolov8n.pt",
    det_conf: float = 0.35,
    pose_conf: float = 0.4,
    min_track_frames: int = 20,
    min_bbox_size: int = 64,
    min_track_confidence: float = 0.30,
    mask_threshold: float = 0.55,
    min_mask_ratio: float = 0.03,
    min_valid_frames: int = 16,
) -> dict[str, Any]:
    source_video = Path(video_path).resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"视频不存在: {source_video}")

    run_dir = build_run_dir(Path(run_root).resolve(), source_video)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_copy = run_dir / source_video.name
    if input_copy != source_video:
        shutil.copy2(source_video, input_copy)
    annotated_video = run_dir / "annotated_result.mp4"
    artifacts_dir = run_dir / "artifacts"

    device_name = detection_device or default_detection_device()
    summaries = video_app.process_video(
        input_path=input_copy,
        output_path=annotated_video,
        model_path=yolo_model,
        det_conf=det_conf,
        pose_conf=pose_conf,
        device=device_name,
        enable_tracking=True,
        export_tracks=True,
        export_format="both",
        artifacts_dir=artifacts_dir,
        track_iou_threshold=0.3,
        track_max_age=15,
        min_track_frames=min_track_frames,
        min_bbox_size=min_bbox_size,
        min_track_confidence=min_track_confidence,
    )

    segmenter = VideoSilhouetteExtractor(
        VideoSilhouetteConfig(
            threshold=mask_threshold,
            min_mask_ratio=min_mask_ratio,
            min_valid_frames=min_valid_frames,
        )
    )
    feature_extractor = get_feature_extractor(str(cfg_path), int(checkpoint_iter))
    tracks: list[dict[str, Any]] = []
    try:
        for summary in summaries:
            track_record = build_track_record(summary)
            if not summary.kept or summary.crop_dir is None:
                tracks.append(track_record)
                continue

            track_output_dir = run_dir / "probe_tracks" / track_record["track_id"]
            sil_result = segmenter.build_track_sequence(summary.crop_dir, track_output_dir)
            track_record.update(
                {
                    "status": sil_result["status"],
                    "quality_note": sil_result["quality_note"],
                    "valid_frames": int(sil_result["valid_frames"]),
                    "used_frames": int(sil_result["used_frames"]),
                    "silhouette_pkl": sil_result["silhouette_pkl"],
                    "silhouette_strip": sil_result["silhouette_strip"],
                }
            )
            if sil_result["status"] != "ok":
                tracks.append(track_record)
                continue

            embedding = feature_extractor.embed_sequence(
                sil_result["sequence"],
                seq_type=f"video_track_{track_record['track_id']}",
                view="video",
            )
            embedding_path = track_output_dir / "embedding.npy"
            np.save(embedding_path, embedding)
            track_record.update(
                {
                    "status": "ok",
                    "quality_note": "ok",
                    "embedding_path": str(embedding_path),
                }
            )
            tracks.append(track_record)
    finally:
        segmenter.close()

    session = {
        "run_dir": str(run_dir),
        "input_video": str(input_copy),
        "annotated_video": str(annotated_video),
        "checkpoint_iter": int(checkpoint_iter),
        "detection_device": str(device_name),
        "tracks": tracks,
        "selected_track_id": choose_best_track(tracks),
        "mode": "tracks_only",
    }
    (Path(session["run_dir"]) / "session_summary.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session


def process_real_video_demo(
    *,
    video_path: str | Path,
    gallery_cache: dict[str, Any],
    cfg_path: str | Path,
    checkpoint_iter: int,
    run_root: str | Path,
    topk: int = 5,
    gallery_view: str | None = None,
    expected_subject: str | None = None,
    detection_device: str | None = None,
    yolo_model: str = "yolov8n.pt",
    det_conf: float = 0.35,
    pose_conf: float = 0.4,
    min_track_frames: int = 20,
    min_bbox_size: int = 64,
    min_track_confidence: float = 0.30,
    mask_threshold: float = 0.55,
    min_mask_ratio: float = 0.03,
    min_valid_frames: int = 16,
) -> dict[str, Any]:
    session = process_video_to_tracks(
        video_path=video_path,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root,
        detection_device=detection_device,
        yolo_model=yolo_model,
        det_conf=det_conf,
        pose_conf=pose_conf,
        min_track_frames=min_track_frames,
        min_bbox_size=min_bbox_size,
        min_track_confidence=min_track_confidence,
        mask_threshold=mask_threshold,
        min_mask_ratio=min_mask_ratio,
        min_valid_frames=min_valid_frames,
    )

    for track_record in session["tracks"]:
        if track_record["status"] != "ok" or not track_record.get("embedding_path"):
            continue
        embedding = np.load(track_record["embedding_path"])
        result = retrieve_embedding_topk(
            gallery_cache,
            embedding,
            topk=topk,
            gallery_view=gallery_view,
            expected_subject=expected_subject,
            probe_meta={
                "key": f"video_track_{track_record['track_id']}",
                "pkl_path": track_record["silhouette_pkl"],
                "type": "video_probe",
                "view": "video",
            },
        )
        track_record.update(
            {
                "rows": result["rows"],
                "gallery_count": int(result["gallery_count"]),
                "top1_subject": result["rows"][0]["subject_id"] if result["rows"] else "",
                "top1_distance": result["rows"][0]["distance"] if result["rows"] else None,
            }
        )

    session.update(
        {
            "gallery_view": "all" if gallery_view is None else str(gallery_view),
            "expected_subject": "" if expected_subject is None else str(expected_subject),
            "mode": "dataset_retrieval",
        }
    )
    (Path(session["run_dir"]) / "session_summary.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session


def process_private_gallery_enrollment(
    *,
    video_path: str | Path,
    cfg_path: str | Path,
    checkpoint_iter: int,
    run_root: str | Path,
    detection_device: str | None = None,
    yolo_model: str = "yolov8n.pt",
    det_conf: float = 0.35,
    pose_conf: float = 0.4,
    min_track_frames: int = 20,
    min_bbox_size: int = 64,
    min_track_confidence: float = 0.30,
    mask_threshold: float = 0.55,
    min_mask_ratio: float = 0.03,
    min_valid_frames: int = 16,
) -> dict[str, Any]:
    session = process_video_to_tracks(
        video_path=video_path,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root,
        detection_device=detection_device,
        yolo_model=yolo_model,
        det_conf=det_conf,
        pose_conf=pose_conf,
        min_track_frames=min_track_frames,
        min_bbox_size=min_bbox_size,
        min_track_confidence=min_track_confidence,
        mask_threshold=mask_threshold,
        min_mask_ratio=min_mask_ratio,
        min_valid_frames=min_valid_frames,
    )
    session["mode"] = "private_gallery_enrollment"
    (Path(session["run_dir"]) / "session_summary.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session


def process_private_gallery_recognition(
    *,
    video_path: str | Path,
    gallery_root: str | Path,
    cfg_path: str | Path,
    checkpoint_iter: int,
    run_root: str | Path,
    topk: int = 5,
    low_conf_distance: float = 0.70,
    low_conf_margin: float = 0.02,
    detection_device: str | None = None,
    yolo_model: str = "yolov8n.pt",
    det_conf: float = 0.35,
    pose_conf: float = 0.4,
    min_track_frames: int = 20,
    min_bbox_size: int = 64,
    min_track_confidence: float = 0.30,
    mask_threshold: float = 0.55,
    min_mask_ratio: float = 0.03,
    min_valid_frames: int = 16,
) -> dict[str, Any]:
    session = process_video_to_tracks(
        video_path=video_path,
        cfg_path=cfg_path,
        checkpoint_iter=checkpoint_iter,
        run_root=run_root,
        detection_device=detection_device,
        yolo_model=yolo_model,
        det_conf=det_conf,
        pose_conf=pose_conf,
        min_track_frames=min_track_frames,
        min_bbox_size=min_bbox_size,
        min_track_confidence=min_track_confidence,
        mask_threshold=mask_threshold,
        min_mask_ratio=min_mask_ratio,
        min_valid_frames=min_valid_frames,
    )

    gallery_index = load_private_gallery_index(gallery_root)
    for track_record in session["tracks"]:
        if track_record["status"] != "ok" or not track_record.get("embedding_path"):
            continue
        embedding = np.asarray(np.load(track_record["embedding_path"]), dtype=np.float32)
        result = match_private_gallery(embedding, gallery_root, topk=topk)
        rows = result["rows"]
        top1_score = rows[0]["score"] if rows else None
        top2_distance = rows[1]["distance"] if len(rows) > 1 else None
        distance_margin = None
        if rows and len(rows) > 1:
            distance_margin = float(rows[1]["distance"]) - float(rows[0]["distance"])
        track_record.update(
            {
                "rows": rows,
                "gallery_count": int(result["identity_count"]),
                "top1_identity": result["top1_identity"],
                "top1_distance": result["top1_distance"],
                "top1_score": top1_score,
                "top2_distance": top2_distance,
                "distance_margin": distance_margin,
                "review_required": bool(result["review_required"]),
                "review_reasons": list(result["review_reasons"]),
            }
        )

    session.update(
        {
            "private_gallery_root": str(Path(gallery_root).resolve()),
            "private_gallery_identity_count": int(len(gallery_index.get("identities", {}))),
            "low_conf_distance": float(low_conf_distance),
            "low_conf_margin": float(low_conf_margin),
            "mode": "private_gallery_recognition",
        }
    )
    (Path(session["run_dir"]) / "session_summary.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session
