from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PLACEHOLDER_IDENTITIES = {"", "unknown", "unknown_id", "unknown-id", "unlabeled", "unlabelled"}


@dataclass(slots=True)
class SequenceSource:
    source_collection: str
    source_identity: str
    source_sequence: str
    source_video: str
    seq_dir: Path
    clip_path: Path | None
    metadata_path: Path | None


@dataclass(slots=True)
class PreparedSequence:
    subject_id: str
    subject_key: str
    sequence_id: str
    view_name: str
    source_collection: str
    source_identity: str
    source_sequence: str
    source_video: str
    frame_count: int
    status: str
    output_dir: str
    source_seq_dir: str
    metadata_path: str
    clip_path: str
    frame_source: str
    clip_copied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将当前项目导出的 gait 序列整理成更接近 OpenGait 的结构")
    parser.add_argument("--input-dir", default="data", help="输入目录，通常是当前项目的 data/")
    parser.add_argument("--output-dir", default="opengait_ready", help="OpenGait 准备目录")
    parser.add_argument("--subject-prefix", default="subject_", help="输出身份目录前缀")
    parser.add_argument("--sequence-prefix", default="seq_", help="输出序列目录前缀")
    parser.add_argument("--view-name", default="000", help="视角占位目录名，默认使用 OpenGait 常见的数字视角占位")
    parser.add_argument("--frame-prefix", default="", help="输出帧文件名前缀，默认空")
    parser.add_argument("--frame-padding", type=int, default=6, help="输出帧编号宽度")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy", help="帧文件使用复制还是软链接")
    parser.add_argument(
        "--subject-key-mode",
        choices=["auto", "collection_identity", "identity"],
        default="auto",
        help="subject 合并策略：auto 在 unknown_id 时按视频分开，已命名 identity 时可跨视频合并",
    )
    parser.add_argument(
        "--exclude-collections",
        nargs="*",
        default=[],
        help="可选：排除 input-dir 下指定 collection 目录，例如历史 smoke 数据目录",
    )
    parser.add_argument("--copy-clips", action="store_true", help="把原始 clip.mp4 也复制到输出序列目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_collections = {name.strip() for name in args.exclude_collections if name.strip()}
    sequences = discover_sequences(input_dir, excluded_collections)
    if not sequences:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到可转换的序列")

    subject_map: dict[str, str] = {}
    prepared_rows: list[PreparedSequence] = []

    for source in sequences:
        subject_key = build_subject_key(source, args.subject_key_mode)
        subject_id = subject_map.get(subject_key)
        if subject_id is None:
            subject_id = f"{args.subject_prefix}{len(subject_map) + 1:03d}"
            subject_map[subject_key] = subject_id

        frame_source, frame_files = locate_frame_source(source.seq_dir)
        sequence_root = ""
        clip_copied = False
        copied_frames = 0
        status = "prepared"

        if not frame_files and (source.clip_path is None or not source.clip_path.exists()):
            frame_source = "missing"
            status = "skipped_no_frames"
            sequence_id = ""
        else:
            subject_root = output_dir / subject_id
            subject_root.mkdir(parents=True, exist_ok=True)
            subject_seq_index = _next_sequence_index(subject_root)
            sequence_id = f"{args.sequence_prefix}{subject_seq_index:03d}"
            sequence_root_path = subject_root / sequence_id
            view_root = sequence_root_path / args.view_name
            view_root.mkdir(parents=True, exist_ok=True)

            if frame_files:
                copied_frames = copy_frames(
                    frame_files=frame_files,
                    destination_dir=view_root,
                    mode=args.copy_mode,
                    frame_prefix=args.frame_prefix,
                    frame_padding=args.frame_padding,
                )
            else:
                copied_frames = extract_frames_from_clip(
                    clip_path=source.clip_path,
                    destination_dir=view_root,
                    frame_prefix=args.frame_prefix,
                    frame_padding=args.frame_padding,
                )
                frame_source = "clip"

            if args.copy_clips and source.clip_path is not None and source.clip_path.exists():
                shutil.copy2(source.clip_path, sequence_root_path / "clip.mp4")
                clip_copied = True

            metadata = load_metadata(source.metadata_path) if source.metadata_path else {}
            metadata.update(
                {
                    "subject_id": subject_id,
                    "subject_key": subject_key,
                    "sequence_id": sequence_id,
                    "view_name": args.view_name,
                    "source_collection": source.source_collection,
                    "source_identity": source.source_identity,
                    "source_sequence": source.source_sequence,
                    "source_video": source.source_video,
                    "frame_source": frame_source,
                    "frame_count": copied_frames,
                    "clip_copied": clip_copied,
                    "status": status,
                }
            )
            write_metadata(sequence_root_path / "metadata.json", metadata)
            sequence_root = str(sequence_root_path)

        prepared_rows.append(
            PreparedSequence(
                subject_id=subject_id,
                subject_key=subject_key,
                sequence_id=sequence_id,
                view_name=args.view_name,
                source_collection=source.source_collection,
                source_identity=source.source_identity,
                source_sequence=source.source_sequence,
                source_video=source.source_video,
                frame_count=copied_frames,
                status=status,
                output_dir=sequence_root,
                source_seq_dir=str(source.seq_dir),
                metadata_path=str(source.metadata_path) if source.metadata_path else "",
                clip_path=str(source.clip_path) if source.clip_path else "",
                frame_source=frame_source,
                clip_copied=clip_copied,
            )
        )

    write_subject_map(output_dir / "subject_map.tsv", subject_map)
    write_manifest(output_dir / "manifest.tsv", prepared_rows)
    cleanup_appledouble_files(output_dir)
    print(f"完成: {len(prepared_rows)} 个序列已整理到 {output_dir}")


def discover_sequences(input_dir: Path, excluded_collections: set[str] | None = None) -> list[SequenceSource]:
    sources: list[SequenceSource] = []
    excluded = excluded_collections or set()
    for collection_dir in sorted(input_dir.iterdir()):
        if not collection_dir.is_dir() or collection_dir.name.startswith("."):
            continue
        if collection_dir.name in excluded:
            continue
        sources.extend(_discover_collection_sequences(collection_dir))
    return sources


def _discover_collection_sequences(collection_dir: Path) -> list[SequenceSource]:
    sequences: list[SequenceSource] = []
    direct_seq_dirs = [path for path in collection_dir.iterdir() if path.is_dir() and path.name.startswith("seq_")]
    if direct_seq_dirs:
        for seq_dir in sorted(direct_seq_dirs):
            sequences.append(
                SequenceSource(
                    source_collection=collection_dir.name,
                    source_identity=collection_dir.name,
                    source_sequence=seq_dir.name,
                    source_video=_read_source_video(seq_dir),
                    seq_dir=seq_dir,
                    clip_path=_find_clip(seq_dir),
                    metadata_path=seq_dir / "metadata.json",
                )
            )
        return sequences

    for identity_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        seq_dirs = [path for path in identity_dir.iterdir() if path.is_dir() and path.name.startswith("seq_")]
        if not seq_dirs:
            continue
        for seq_dir in sorted(seq_dirs):
            sequences.append(
                SequenceSource(
                    source_collection=collection_dir.name,
                    source_identity=identity_dir.name,
                    source_sequence=seq_dir.name,
                    source_video=_read_source_video(seq_dir),
                    seq_dir=seq_dir,
                    clip_path=_find_clip(seq_dir),
                    metadata_path=seq_dir / "metadata.json",
                )
            )
    return sequences


def locate_frame_source(seq_dir: Path) -> tuple[str, list[Path]]:
    direct_frames = sorted(
        path for path in seq_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if direct_frames:
        return "direct", direct_frames

    for child in sorted(seq_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        child_frames = sorted(
            path for path in child.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if child_frames:
            return child.name, child_frames
    return "missing", []


def copy_frames(
    *,
    frame_files: Iterable[Path],
    destination_dir: Path,
    mode: str,
    frame_prefix: str,
    frame_padding: int,
) -> int:
    count = 0
    for index, src in enumerate(frame_files, start=1):
        suffix = src.suffix.lower() or ".jpg"
        dst = destination_dir / f"{frame_prefix}{index:0{frame_padding}d}{suffix}"
        if mode == "symlink":
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        count += 1
    return count


def extract_frames_from_clip(
    *,
    clip_path: Path,
    destination_dir: Path,
    frame_prefix: str,
    frame_padding: int,
) -> int:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开 clip: {clip_path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            frame_path = destination_dir / f"{frame_prefix}{count:0{frame_padding}d}.jpg"
            cv2.imwrite(str(frame_path), frame)
    finally:
        capture.release()
    return count


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return csv_load_json_like(fh)


def csv_load_json_like(handle) -> dict:
    try:
        return json.load(handle)
    except json.JSONDecodeError:
        return {}


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_subject_map(path: Path, subject_map: dict[str, str]) -> None:
    lines = ["subject_id\tsubject_key\tsource_collection\tsource_identity"]
    for source_key, subject_id in sorted(subject_map.items(), key=lambda item: item[1]):
        source_collection, source_identity = split_subject_key(source_key)
        lines.append(f"{subject_id}\t{source_key}\t{source_collection}\t{source_identity}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, rows: list[PreparedSequence]) -> None:
    lines = [
        "status\tsubject_id\tsubject_key\tsequence_id\tview_name\tsource_collection\tsource_identity\tsource_sequence\tsource_video\tframe_count\tframe_source\tclip_copied\toutput_dir\tsource_seq_dir\tmetadata_path\tclip_path"
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.status,
                    row.subject_id,
                    row.subject_key,
                    row.sequence_id,
                    row.view_name,
                    row.source_collection,
                    row.source_identity,
                    row.source_sequence,
                    row.source_video,
                    str(row.frame_count),
                    row.frame_source,
                    "1" if row.clip_copied else "0",
                    row.output_dir,
                    row.source_seq_dir,
                    row.metadata_path,
                    row.clip_path,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_sequence_index(subject_root: Path) -> int:
    existing = []
    for child in subject_root.iterdir():
        if not child.is_dir() or not child.name.startswith("seq_"):
            continue
        try:
            existing.append(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(existing, default=0) + 1


def _find_clip(seq_dir: Path) -> Path | None:
    direct = seq_dir / "clip.mp4"
    if direct.exists():
        return direct
    for child in seq_dir.iterdir():
        if child.is_file() and child.suffix.lower() == ".mp4":
            return child
    return None


def _read_source_video(seq_dir: Path) -> str:
    meta = seq_dir / "metadata.json"
    if not meta.exists():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return str(data.get("source_video", ""))
    except Exception:
        return ""


def build_subject_key(source: SequenceSource, mode: str) -> str:
    identity = source.source_identity.strip()
    if mode == "collection_identity":
        return f"{source.source_collection}::{identity}"
    if mode == "identity":
        return identity or source.source_collection
    if is_placeholder_identity(identity):
        return f"{source.source_collection}::{identity or 'unknown_id'}"
    return identity


def split_subject_key(subject_key: str) -> tuple[str, str]:
    if "::" in subject_key:
        collection, identity = subject_key.split("::", 1)
        return collection, identity
    return "", subject_key


def is_placeholder_identity(identity: str) -> bool:
    normalized = identity.strip().lower()
    return normalized in PLACEHOLDER_IDENTITIES or normalized.startswith("unknown")


def cleanup_appledouble_files(root: Path) -> None:
    sibling = root.parent / f"._{root.name}"
    if sibling.exists():
        sibling.unlink(missing_ok=True)
    if not root.exists():
        return
    for path in root.rglob("._*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
