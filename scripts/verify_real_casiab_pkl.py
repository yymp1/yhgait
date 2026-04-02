from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class SampleInfo:
    path: str
    shape: list[int]
    dtype: str


@dataclass(slots=True)
class VerificationReport:
    dataset_root: str
    subject_count: int
    type_count: int
    view_count: int
    pkl_count: int
    sample_paths: list[str]
    samples: list[SampleInfo]
    classification: str
    real_data_status: str
    notes: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证当前 CASIA-B-pkl 是否更像真实 pretreatment 产物")
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "datasets" / "processed" / "CASIA-B-pkl"),
        help="待验证的 CASIA-B-pkl 根目录",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports"),
        help="报告输出目录",
    )
    parser.add_argument("--json-report", default="real_data_validation.json", help="JSON 报告文件名")
    parser.add_argument("--md-report", default="real_data_validation.md", help="Markdown 报告文件名")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = inspect_dataset_root(dataset_root)
    json_path = reports_dir / args.json_report
    md_path = reports_dir / args.md_report
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"dataset_root: {report.dataset_root}")
    print(f"subject_count: {report.subject_count}")
    print(f"pkl_count: {report.pkl_count}")
    print(f"classification: {report.classification}")
    print(f"real_data_status: {report.real_data_status}")
    print(f"json_report: {json_path}")
    print(f"md_report: {md_path}")

    if report.real_data_status != "ready":
        raise SystemExit(1)


def inspect_dataset_root(dataset_root: Path) -> VerificationReport:
    if not dataset_root.exists():
        return VerificationReport(
            dataset_root=str(dataset_root),
            subject_count=0,
            type_count=0,
            view_count=0,
            pkl_count=0,
            sample_paths=[],
            samples=[],
            classification="missing",
            real_data_status="blocked",
            notes=["数据目录不存在"],
        )

    subject_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    type_names: set[str] = set()
    view_names: set[str] = set()
    pkl_paths = sorted(
        path for path in dataset_root.rglob("*.pkl")
        if path.is_file() and not path.name.startswith("._")
    )

    for subject_dir in subject_dirs:
        for type_dir in subject_dir.iterdir():
            if not type_dir.is_dir() or type_dir.name.startswith("."):
                continue
            type_names.add(type_dir.name)
            for view_dir in type_dir.iterdir():
                if not view_dir.is_dir() or view_dir.name.startswith("."):
                    continue
                view_names.add(view_dir.name)

    sample_paths = pick_sample_paths(pkl_paths)
    samples: list[SampleInfo] = []
    for sample_path in sample_paths:
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        samples.append(
            SampleInfo(
                path=str(sample_path),
                shape=list(getattr(sample, "shape", [])),
                dtype=str(getattr(sample, "dtype", "unknown")),
            )
        )

    notes = []
    if len(subject_dirs) < 100:
        notes.append("subject 数明显低于真实 CASIA-B pretreatment 常见量级")
    if len(pkl_paths) < 1000:
        notes.append("pkl 数明显低于真实 CASIA-B pretreatment 常见量级")
    if len(subject_dirs) >= 100 and len(pkl_paths) >= 1000:
        classification = "likely_real"
        real_data_status = "ready"
        notes.append("数量级已经与真实 pretreatment 产物明显区分开")
    else:
        classification = "synthetic_or_incomplete"
        real_data_status = "blocked"

    return VerificationReport(
        dataset_root=str(dataset_root),
        subject_count=len(subject_dirs),
        type_count=len(type_names),
        view_count=len(view_names),
        pkl_count=len(pkl_paths),
        sample_paths=[str(path) for path in sample_paths],
        samples=samples,
        classification=classification,
        real_data_status=real_data_status,
        notes=notes,
    )


def pick_sample_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    return [paths[index] for index in indices]


def render_markdown(report: VerificationReport) -> str:
    lines = [
        "# 真实 CASIA-B-pkl 校验报告",
        "",
        f"- 数据目录：`{report.dataset_root}`",
        f"- subject 数：`{report.subject_count}`",
        f"- type 数：`{report.type_count}`",
        f"- view 数：`{report.view_count}`",
        f"- pkl 数：`{report.pkl_count}`",
        f"- 分类：`{report.classification}`",
        f"- real_data_status：`{report.real_data_status}`",
        "",
        "## 抽样 pkl",
    ]
    if not report.samples:
        lines.append("")
        lines.append("- none")
    else:
        lines.append("")
        for sample in report.samples:
            lines.append(f"- `{sample.path}` shape={sample.shape} dtype={sample.dtype}")
    lines.extend(
        [
            "",
            "## 备注",
            "",
        ]
    )
    if not report.notes:
        lines.append("- none")
    else:
        for note in report.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
