from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_casia_b_layout import LayoutReport, inspect_layout  # noqa: E402
from scripts.clean_casia_b_empty_views import ScanReport, scan_empty_views  # noqa: E402
from scripts.run_opengait_pretreatment import build_plan, validate_plan  # noqa: E402


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    detail: str = ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在真正执行 OpenGait pretreatment 前做一键预检")
    parser.add_argument(
        "--opengait-root",
        default=str(ROOT / "external" / "OpenGait"),
        help="外部 OpenGait 仓库根目录",
    )
    parser.add_argument(
        "--input-path",
        default=str(ROOT / "datasets" / "external" / "casia_b"),
        help="CASIA-B 原始数据目录",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "datasets" / "processed" / "CASIA-B-pkl"),
        help="pretreatment 输出目录",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports"),
        help="报告输出目录",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="执行 pretreatment 的 Python 可执行文件")
    parser.add_argument("--dataset-name", default="CASIAB", help="传给 OpenGait 的 dataset 名称")
    parser.add_argument("--n-workers", type=int, default=4, help="pretreatment worker 数")
    parser.add_argument("--img-size", type=int, default=64, help="pretreatment 输出尺寸")
    parser.add_argument("--verbose", action="store_true", help="透传给 pretreatment.py 的 verbose 标记")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    opengait_root = Path(args.opengait_root).expanduser().resolve()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    opengait_check = check_opengait_root(opengait_root)
    layout_report = inspect_layout(input_path)
    layout_check = summarize_layout(layout_report)
    empty_view_report = scan_empty_views(input_path)
    empty_view_check = summarize_empty_views(empty_view_report)
    pretreatment_plan, pretreatment_issues = build_pretreatment_plan(
        args=args,
        opengait_root=opengait_root,
        input_path=input_path,
        output_path=output_path,
    )
    pretreatment_check = summarize_pretreatment(pretreatment_plan, pretreatment_issues)

    checks = [opengait_check, layout_check, empty_view_check, pretreatment_check]
    preflight_status = "ready" if all(check.status == "passed" for check in checks) else "blocked"

    wrapper_dry_run_command = build_wrapper_command(args=args, dry_run=True)
    wrapper_execute_command = build_wrapper_command(args=args, dry_run=False)
    raw_pretreatment_command = shlex.join(pretreatment_plan.command())

    print_report(
        preflight_status=preflight_status,
        checks=checks,
        wrapper_dry_run_command=wrapper_dry_run_command,
        wrapper_execute_command=wrapper_execute_command,
        raw_pretreatment_command=raw_pretreatment_command,
    )
    write_tsv_report(
        reports_dir / "pretreatment_preflight.tsv",
        preflight_status=preflight_status,
        checks=checks,
        wrapper_dry_run_command=wrapper_dry_run_command,
        raw_pretreatment_command=raw_pretreatment_command,
    )
    write_markdown_report(
        reports_dir / "pretreatment_preflight.md",
        preflight_status=preflight_status,
        checks=checks,
        wrapper_dry_run_command=wrapper_dry_run_command,
        wrapper_execute_command=wrapper_execute_command,
        raw_pretreatment_command=raw_pretreatment_command,
    )


def check_opengait_root(opengait_root: Path) -> CheckResult:
    missing = []
    if not opengait_root.exists():
        missing.append(f"仓库目录不存在: {opengait_root}")
    pretreatment_script = opengait_root / "datasets" / "pretreatment.py"
    casia_readme = opengait_root / "datasets" / "CASIA-B" / "README.md"
    if not pretreatment_script.exists():
        missing.append(f"缺少 datasets/pretreatment.py: {pretreatment_script}")
    if not casia_readme.exists():
        missing.append(f"缺少 datasets/CASIA-B/README.md: {casia_readme}")
    if missing:
        return CheckResult(
            name="opengait_root",
            status="failed",
            summary="OpenGait 仓库或关键文件缺失",
            detail="；".join(missing),
        )
    return CheckResult(
        name="opengait_root",
        status="passed",
        summary="OpenGait 仓库和关键文件齐全",
        detail=f"repo={opengait_root}",
    )


def summarize_layout(layout_report: LayoutReport) -> CheckResult:
    detail = (
        f"subject_count={layout_report.subject_count}; "
        f"type_count={layout_report.type_count}; "
        f"view_count={layout_report.view_count}; "
        f"png_count={layout_report.png_count}; "
        f"readiness={layout_report.readiness}"
    )
    if layout_report.readiness != "ready":
        if layout_report.issues:
            issue_preview = "；".join(f"{issue.path}: {issue.message}" for issue in layout_report.issues[:5])
            detail = f"{detail}; issues={issue_preview}"
        return CheckResult(
            name="casia_b_layout",
            status="failed",
            summary="CASIA-B 原始目录尚未达到 ready",
            detail=detail,
        )
    return CheckResult(
        name="casia_b_layout",
        status="passed",
        summary="CASIA-B 原始目录检查通过",
        detail=detail,
    )


def summarize_empty_views(scan_report: ScanReport) -> CheckResult:
    detail = (
        f"empty_view_count={scan_report.empty_view_count}; "
        f"suspicious_view_count={scan_report.suspicious_view_count}; "
        f"affected_subject_count={len(scan_report.affected_subjects)}; "
        f"affected_type_count={len(scan_report.affected_types)}"
    )
    if scan_report.empty_view_count > 0 or scan_report.suspicious_view_count > 0:
        preview_parts = [str(entry.path) for entry in scan_report.empty_views[:5]]
        preview_parts.extend(f"{entry.path} ({entry.detail})" for entry in scan_report.suspicious_views[:5])
        if preview_parts:
            detail = f"{detail}; examples={'；'.join(preview_parts)}"
        return CheckResult(
            name="empty_view_scan",
            status="failed",
            summary="仍存在空目录或 suspicious 目录",
            detail=detail,
        )
    return CheckResult(
        name="empty_view_scan",
        status="passed",
        summary="空目录和 suspicious 目录检查通过",
        detail=detail,
    )


def build_pretreatment_plan(*, args: argparse.Namespace, opengait_root: Path, input_path: Path, output_path: Path):
    namespace = SimpleNamespace(
        opengait_root=str(opengait_root),
        pretreatment_script="",
        input_path=str(input_path),
        output_path=str(output_path),
        python_bin=args.python_bin,
        dataset_name=args.dataset_name,
        n_workers=args.n_workers,
        img_size=args.img_size,
        log_file="",
        verbose=args.verbose,
        dry_run=True,
        execute=False,
    )
    plan = build_plan(namespace)
    issues = validate_plan(plan)
    return plan, issues


def summarize_pretreatment(plan, issues: list[str]) -> CheckResult:
    detail = shlex.join(plan.command())
    if issues:
        return CheckResult(
            name="pretreatment_dry_run",
            status="failed",
            summary="pretreatment dry-run 规划存在阻塞项",
            detail=f"{detail}; issues={'；'.join(issues)}",
        )
    return CheckResult(
        name="pretreatment_dry_run",
        status="passed",
        summary="pretreatment dry-run 命令已生成",
        detail=detail,
    )


def build_wrapper_command(*, args: argparse.Namespace, dry_run: bool) -> str:
    command = [
        args.python_bin,
        str(ROOT / "scripts" / "run_opengait_pretreatment.py"),
        "--opengait-root",
        args.opengait_root,
        "--input-path",
        args.input_path,
        "--output-path",
        args.output_path,
        "--dataset-name",
        args.dataset_name,
        "--n-workers",
        str(args.n_workers),
        "--img-size",
        str(args.img_size),
    ]
    if args.verbose:
        command.append("--verbose")
    command.append("--dry-run" if dry_run else "--execute")
    return shlex.join(command)


def print_report(
    *,
    preflight_status: str,
    checks: list[CheckResult],
    wrapper_dry_run_command: str,
    wrapper_execute_command: str,
    raw_pretreatment_command: str,
) -> None:
    print(f"preflight_status: {preflight_status}")
    print("checks:")
    for check in checks:
        print(f"- {check.name}: {check.status} | {check.summary}")
        if check.detail:
            print(f"  detail: {check.detail}")
    print("next_dry_run_command:")
    print(wrapper_dry_run_command)
    print("raw_pretreatment_command:")
    print(raw_pretreatment_command)
    if preflight_status == "ready":
        print("next_execute_command:")
        print(wrapper_execute_command)


def write_tsv_report(
    path: Path,
    *,
    preflight_status: str,
    checks: list[CheckResult],
    wrapper_dry_run_command: str,
    raw_pretreatment_command: str,
) -> None:
    lines = [
        "row_type\tname\tstatus\tsummary\tdetail",
        f"summary\tpreflight_status\t{preflight_status}\t\t",
        f"command\tnext_dry_run\tready\t{wrapper_dry_run_command}\t",
        f"command\traw_pretreatment\tready\t{raw_pretreatment_command}\t",
    ]
    for check in checks:
        lines.append(
            "\t".join(
                [
                    "check",
                    check.name,
                    check.status,
                    check.summary,
                    check.detail,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_report(
    path: Path,
    *,
    preflight_status: str,
    checks: list[CheckResult],
    wrapper_dry_run_command: str,
    wrapper_execute_command: str,
    raw_pretreatment_command: str,
) -> None:
    lines = [
        "# OpenGait pretreatment 预检报告",
        "",
        f"- 预检结论：`{preflight_status}`",
        "",
        "## 检查项",
    ]
    for check in checks:
        lines.append(f"- `{check.name}`：`{check.status}`，{check.summary}")
        if check.detail:
            lines.append(f"  说明：`{check.detail}`")
    lines.extend(
        [
            "",
            "## 下一条 dry-run 命令",
            "",
            "```bash",
            wrapper_dry_run_command,
            "```",
            "",
            "## OpenGait 原始 pretreatment 命令",
            "",
            "```bash",
            raw_pretreatment_command,
            "```",
        ]
    )
    if preflight_status == "ready":
        lines.extend(
            [
                "",
                "## 通过后可执行命令",
                "",
                "```bash",
                wrapper_execute_command,
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
