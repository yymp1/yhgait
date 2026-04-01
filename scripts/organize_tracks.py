from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将导出的 gait track 组织成可人工整理的数据结构")
    parser.add_argument("--source-root", "--source", dest="source_root", default="artifacts", help="导出根目录或 tracks 目录")
    parser.add_argument("--target-root", "--output", dest="target_root", default="data", help="整理后的数据根目录")
    parser.add_argument("--identity", default="unknown_id", help="默认身份目录名")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy", help="复制或创建软链接")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = organize_tracks(
        source_root=Path(args.source_root),
        target_root=Path(args.target_root),
        identity=args.identity,
        copy_mode=args.copy_mode,
    )
    print(f"完成: {result['track_count']} 个轨迹已整理到 {result['identity_dir']}")


def organize_tracks(
    source_root: Path,
    target_root: Path,
    identity: str = "unknown_id",
    copy_mode: str = "copy",
) -> dict[str, object]:
    if (source_root / "tracks").exists():
        tracks_root = source_root / "tracks"
        clips_root = source_root / "clips"
    else:
        tracks_root = source_root
        clips_root = source_root.parent / "clips"

    track_entries = _discover_track_entries(tracks_root, clips_root)
    if not track_entries:
        raise FileNotFoundError(f"找不到可整理的轨迹数据: {tracks_root} 或 {clips_root}")

    identity_dir = target_root / identity
    identity_dir.mkdir(parents=True, exist_ok=True)
    seq_start = _next_sequence_index(identity_dir)
    manifest_rows: list[dict[str, str]] = []

    for offset, track_entry in enumerate(track_entries, start=0):
        seq_name = f"seq_{seq_start + offset:03d}"
        seq_dir = identity_dir / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        track_id = track_entry["track_id"]
        frame_dir = track_entry.get("frame_dir")
        clip_src = track_entry.get("clip_src")
        track_meta = _load_metadata(frame_dir / "metadata.json") if frame_dir is not None else None

        frame_files = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir is not None else []
        if frame_files:
            frame_target = seq_dir / "frames"
            frame_target.mkdir(parents=True, exist_ok=True)
            for frame_file in frame_files:
                _transfer(frame_file, frame_target / frame_file.name, copy_mode)

        if clip_src is not None and clip_src.exists():
            _transfer(clip_src, seq_dir / "clip.mp4", copy_mode)

        if track_meta is not None:
            with open(seq_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(track_meta, f, ensure_ascii=False, indent=2)
        else:
            track_meta = {}

        manifest_rows.append(
            {
                "identity": identity,
                "sequence": seq_name,
                "track_id": track_id,
                "frames": str(len(frame_files)),
                "clip": "clip.mp4" if clip_src is not None and clip_src.exists() else "",
                "source_video": str(track_meta.get("source_video", "")),
            }
        )

    manifest_path = identity_dir / "manifest.tsv"
    _write_manifest(manifest_path, manifest_rows)
    _cleanup_appledouble_files(target_root)
    return {
        "identity_dir": identity_dir,
        "manifest_path": manifest_path,
        "track_count": len(track_entries),
    }


def _discover_track_entries(tracks_root: Path, clips_root: Path) -> list[dict[str, Path | str | None]]:
    entries: list[dict[str, Path | str | None]] = []

    if tracks_root.exists():
        track_dirs = [p for p in tracks_root.iterdir() if p.is_dir()]
        track_dirs.sort(key=_track_sort_key)
        for track_dir in track_dirs:
            track_id = track_dir.name
            entries.append(
                {
                    "track_id": track_id,
                    "frame_dir": track_dir,
                    "clip_src": clips_root / f"track_{track_id}.mp4",
                }
            )
        if entries:
            return entries

    if clips_root.exists():
        clip_files = sorted(clips_root.glob("track_*.mp4"), key=_track_sort_key)
        for clip_file in clip_files:
            track_id = clip_file.stem.replace("track_", "", 1)
            entries.append(
                {
                    "track_id": track_id,
                    "frame_dir": None,
                    "clip_src": clip_file,
                }
            )

    return entries


def _track_sort_key(path: Path) -> tuple[int, str]:
    name = path.stem if path.suffix else path.name
    if name.startswith("track_"):
        name = name.replace("track_", "", 1)
    try:
        return (int(name), path.name)
    except ValueError:
        return (10**9, path.name)


def _next_sequence_index(identity_dir: Path) -> int:
    existing = []
    for child in identity_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("seq_"):
            continue
        try:
            existing.append(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(existing, default=0) + 1


def _load_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _transfer(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copyfile(src, dst)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    headers = ["identity", "sequence", "track_id", "frames", "clip", "source_video"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row.get(header, "") for header in headers) + "\n")


def _cleanup_appledouble_files(root: Path) -> None:
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
