from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBJECT_RE = re.compile(r"^\d{3}$")
TYPE_RE = re.compile(r"^(bg|cl|nm)-\d{2}$", re.IGNORECASE)
VIEW_RE = re.compile(r"^\d{3}$")


@dataclass
class LayoutIssue:
    level: str
    path: str
    message: str


@dataclass
class LayoutReport:
    input_dir: Path
    subject_count: int = 0
    type_count: int = 0
    view_count: int = 0
    png_count: int = 0
    subjects: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    views: set[str] = field(default_factory=set)
    issues: list[LayoutIssue] = field(default_factory=list)

    @property
    def readiness(self) -> str:
        if self.issues:
            return "needs_attention"
        if self.subject_count == 0 or self.type_count == 0 or self.view_count == 0 or self.png_count == 0:
            return "needs_attention"
        return "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 CASIA-B 原始目录是否接近 OpenGait pretreatment 期望的层级")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "datasets" / "external" / "casia_b"),
        help="CASIA-B 原始数据目录",
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "tsv"],
        default="text",
        help="输出报告格式",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：只要存在 warning 或 error 就返回非零退出码",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = inspect_layout(Path(args.input_dir).expanduser().resolve())
    if args.report_format == "tsv":
        print_tsv(report)
    else:
        print_text(report)

    if report.error_count > 0:
        raise SystemExit(2)
    if args.strict and report.warning_count > 0:
        raise SystemExit(1)


def inspect_layout(input_dir: Path) -> LayoutReport:
    report = LayoutReport(input_dir=input_dir)
    if not input_dir.exists():
        report.issues.append(LayoutIssue("error", str(input_dir), "输入目录不存在"))
        return report
    if not input_dir.is_dir():
        report.issues.append(LayoutIssue("error", str(input_dir), "输入路径不是目录"))
        return report

    subject_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not subject_dirs:
        report.issues.append(LayoutIssue("warning", str(input_dir), "没有发现 subject 目录"))
        return report

    for subject_dir in subject_dirs:
        report.subjects.add(subject_dir.name)
        if not SUBJECT_RE.match(subject_dir.name):
            report.issues.append(LayoutIssue("warning", str(subject_dir), "subject 目录名不是常见的三位编号"))

        type_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
        if not type_dirs:
            report.issues.append(LayoutIssue("warning", str(subject_dir), "subject 下没有 sequence/type 目录"))
            continue

        for type_dir in type_dirs:
            report.types.add(type_dir.name)
            if not TYPE_RE.match(type_dir.name):
                report.issues.append(LayoutIssue("warning", str(type_dir), "type 目录名不符合常见的 bg-01 / nm-01 / cl-01 形式"))

            view_dirs = sorted(path for path in type_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
            if not view_dirs:
                report.issues.append(LayoutIssue("warning", str(type_dir), "type 下没有 view 目录"))
                continue

            for view_dir in view_dirs:
                report.views.add(view_dir.name)
                if not VIEW_RE.match(view_dir.name):
                    report.issues.append(LayoutIssue("warning", str(view_dir), "view 目录名不是常见的三位角度编号"))

                png_files = sorted(path for path in view_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
                if not png_files:
                    report.issues.append(LayoutIssue("warning", str(view_dir), "view 目录下没有 png 帧文件"))
                    continue
                report.png_count += len(png_files)

    report.subject_count = len(report.subjects)
    report.type_count = len(report.types)
    report.view_count = len(report.views)
    if report.png_count == 0:
        report.issues.append(LayoutIssue("warning", str(input_dir), "没有发现任何 png 帧文件，当前不适合直接进入 pretreatment"))
    return report


def print_text(report: LayoutReport) -> None:
    print(f"input_dir: {report.input_dir}")
    print(f"subject_count: {report.subject_count}")
    print(f"type_count: {report.type_count}")
    print(f"view_count: {report.view_count}")
    print(f"png_count: {report.png_count}")
    print(f"readiness: {report.readiness}")
    print(f"sample_subjects: {', '.join(sorted(report.subjects)[:10]) or '-'}")
    print(f"sample_types: {', '.join(sorted(report.types)[:10]) or '-'}")
    print(f"sample_views: {', '.join(sorted(report.views)[:10]) or '-'}")
    print("issues:")
    if not report.issues:
        print("- none")
        return
    for issue in report.issues:
        print(f"- [{issue.level}] {issue.path}: {issue.message}")


def print_tsv(report: LayoutReport) -> None:
    summary_rows = [
        ("summary", "input_dir", str(report.input_dir)),
        ("summary", "subject_count", str(report.subject_count)),
        ("summary", "type_count", str(report.type_count)),
        ("summary", "view_count", str(report.view_count)),
        ("summary", "png_count", str(report.png_count)),
        ("summary", "readiness", report.readiness),
    ]
    print("row_type\tkey\tvalue")
    for row_type, key, value in summary_rows:
        print(f"{row_type}\t{key}\t{value}")
    for issue in report.issues:
        print(f"issue\t{issue.level}\t{issue.path}: {issue.message}")


if __name__ == "__main__":
    main()
