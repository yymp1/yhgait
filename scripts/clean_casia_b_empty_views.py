from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class ViewEntry:
    subject: str
    type_name: str
    view: str
    path: Path
    state: str
    detail: str = ""


@dataclass
class ScanReport:
    input_dir: Path
    empty_views: list[ViewEntry] = field(default_factory=list)
    suspicious_views: list[ViewEntry] = field(default_factory=list)
    scanned_subjects: set[str] = field(default_factory=set)
    affected_subjects: set[str] = field(default_factory=set)
    affected_types: set[str] = field(default_factory=set)

    @property
    def empty_view_count(self) -> int:
        return len(self.empty_views)

    @property
    def suspicious_view_count(self) -> int:
        return len(self.suspicious_views)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="筛查并可选清理 CASIA-B 中真正为空的 view 目录")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "datasets" / "external" / "casia_b"),
        help="CASIA-B 原始数据目录",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports"),
        help="报告输出目录",
    )
    parser.add_argument(
        "--tsv-report",
        default="casia_b_empty_views.tsv",
        help="TSV 报告文件名",
    )
    parser.add_argument(
        "--md-report",
        default="casia_b_empty_views.md",
        help="Markdown 报告文件名",
    )
    parser.add_argument("--dry-run", action="store_true", help="显示将要删除的目录，但不实际删除")
    parser.add_argument(
        "--delete-empty-views",
        action="store_true",
        help="显式允许删除真正为空的 view 目录",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=50,
        help="终端里最多展示多少条空目录记录",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    scan_report = scan_empty_views(input_dir)
    tsv_path = reports_dir / args.tsv_report
    md_path = reports_dir / args.md_report
    write_tsv_report(tsv_path, scan_report)
    write_markdown_report(md_path, scan_report)

    if args.delete_empty_views:
        deleted_count = delete_empty_views(scan_report.empty_views)
        print_scan_report(
            scan_report,
            deleted_count=deleted_count,
            dry_run=False,
            print_limit=args.print_limit,
        )
        print(f"tsv_report: {tsv_path}")
        print(f"md_report: {md_path}")
        return

    print_scan_report(
        scan_report,
        deleted_count=0,
        dry_run=args.dry_run,
        print_limit=args.print_limit,
    )
    print(f"tsv_report: {tsv_path}")
    print(f"md_report: {md_path}")


def scan_empty_views(input_dir: Path) -> ScanReport:
    report = ScanReport(input_dir=input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        return report

    for subject_dir in sorted(path for path in input_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        report.scanned_subjects.add(subject_dir.name)
        for type_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            for view_dir in sorted(path for path in type_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
                entry = classify_view_dir(subject_dir.name, type_dir.name, view_dir)
                if entry is None:
                    continue
                if entry.state == "empty":
                    report.empty_views.append(entry)
                    report.affected_subjects.add(entry.subject)
                    report.affected_types.add(entry.type_name)
                elif entry.state == "suspicious":
                    report.suspicious_views.append(entry)
                    report.affected_subjects.add(entry.subject)
                    report.affected_types.add(entry.type_name)
    return report


def classify_view_dir(subject: str, type_name: str, view_dir: Path) -> ViewEntry | None:
    entries = list(view_dir.iterdir())
    if not entries:
        return ViewEntry(subject=subject, type_name=type_name, view=view_dir.name, path=view_dir, state="empty")

    visible_entries = [entry for entry in entries if not entry.name.startswith(".")]
    if not visible_entries:
        return ViewEntry(
            subject=subject,
            type_name=type_name,
            view=view_dir.name,
            path=view_dir,
            state="suspicious",
            detail="仅包含隐藏文件或隐藏目录，保守起见不自动删除",
        )

    png_files = [entry for entry in visible_entries if entry.is_file() and entry.suffix.lower() == ".png"]
    if png_files:
        return None

    detail_parts = []
    other_files = [entry.name for entry in visible_entries if entry.is_file()]
    nested_dirs = [entry.name for entry in visible_entries if entry.is_dir()]
    if other_files:
        detail_parts.append(f"包含非 png 文件: {', '.join(other_files[:5])}")
    if nested_dirs:
        detail_parts.append(f"包含子目录: {', '.join(nested_dirs[:5])}")
    if not detail_parts:
        detail_parts.append("目录非空，但未检测到 png 文件")

    return ViewEntry(
        subject=subject,
        type_name=type_name,
        view=view_dir.name,
        path=view_dir,
        state="suspicious",
        detail="；".join(detail_parts),
    )


def delete_empty_views(entries: list[ViewEntry]) -> int:
    deleted = 0
    for entry in entries:
        if not entry.path.exists() or not entry.path.is_dir():
            continue
        if any(entry.path.iterdir()):
            continue
        entry.path.rmdir()
        deleted += 1
    return deleted


def print_scan_report(
    report: ScanReport,
    *,
    deleted_count: int,
    dry_run: bool,
    print_limit: int,
) -> None:
    mode = "delete" if deleted_count else ("dry-run" if dry_run else "scan")
    print(f"mode: {mode}")
    print(f"input_dir: {report.input_dir}")
    print(f"scanned_subject_count: {len(report.scanned_subjects)}")
    print(f"empty_view_count: {report.empty_view_count}")
    print(f"suspicious_view_count: {report.suspicious_view_count}")
    print(f"affected_subject_count: {len(report.affected_subjects)}")
    print(f"affected_type_count: {len(report.affected_types)}")
    print(f"deleted_count: {deleted_count}")
    print("empty_views:")
    if not report.empty_views:
        print("- none")
    else:
        for entry in report.empty_views[:print_limit]:
            print(f"- {entry.path}")
        if len(report.empty_views) > print_limit:
            print(f"- ... 其余 {len(report.empty_views) - print_limit} 条已写入报告文件")
    print("suspicious_views:")
    if not report.suspicious_views:
        print("- none")
    else:
        for entry in report.suspicious_views[:print_limit]:
            print(f"- {entry.path}: {entry.detail}")
        if len(report.suspicious_views) > print_limit:
            print(f"- ... 其余 {len(report.suspicious_views) - print_limit} 条已写入报告文件")


def write_tsv_report(path: Path, report: ScanReport) -> None:
    lines = [
        "row_type\tstate\tsubject\ttype\tview\tpath\tdetail",
        f"summary\tscan\t\t\t\t{report.input_dir}\tempty={report.empty_view_count};suspicious={report.suspicious_view_count}",
    ]
    for entry in report.empty_views:
        lines.append(
            "\t".join(
                [
                    "view",
                    "empty",
                    entry.subject,
                    entry.type_name,
                    entry.view,
                    str(entry.path),
                    entry.detail,
                ]
            )
        )
    for entry in report.suspicious_views:
        lines.append(
            "\t".join(
                [
                    "view",
                    "suspicious",
                    entry.subject,
                    entry.type_name,
                    entry.view,
                    str(entry.path),
                    entry.detail,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: ScanReport) -> None:
    lines = [
        "# CASIA-B 空目录扫描报告",
        "",
        f"- 输入目录：`{report.input_dir}`",
        f"- 扫描到的 subject 数：`{len(report.scanned_subjects)}`",
        f"- 空 view 目录数：`{report.empty_view_count}`",
        f"- suspicious 目录数：`{report.suspicious_view_count}`",
        "",
        "## 空 view 目录",
    ]
    if report.empty_views:
        for entry in report.empty_views:
            lines.append(f"- `{entry.path}`")
    else:
        lines.append("- 无")
    lines.extend(["", "## Suspicious 目录"])
    if report.suspicious_views:
        for entry in report.suspicious_views:
            lines.append(f"- `{entry.path}`：{entry.detail}")
    else:
        lines.append("- 无")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
