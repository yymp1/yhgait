from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PretreatmentPlan:
    python_bin: Path
    opengait_root: Path
    pretreatment_script: Path
    input_path: Path
    output_path: Path
    log_file: Path
    dataset_name: str
    n_workers: int
    img_size: int
    verbose: bool

    def command(self) -> list[str]:
        command = [
            str(self.python_bin),
            str(self.pretreatment_script),
            "--input_path",
            str(self.input_path),
            "--output_path",
            str(self.output_path),
            "--dataset",
            self.dataset_name,
            "--n_workers",
            str(self.n_workers),
            "--img_size",
            str(self.img_size),
            "--log_file",
            str(self.log_file),
        ]
        if self.verbose:
            command.append("--verbose")
        return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成或执行 OpenGait 的 CASIA-B pretreatment 命令")
    parser.add_argument("--opengait-root", required=True, help="外部 OpenGait 仓库根目录")
    parser.add_argument(
        "--pretreatment-script",
        default="",
        help="可选：显式指定 pretreatment.py 路径；默认使用 <opengait-root>/datasets/pretreatment.py",
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
    parser.add_argument("--python-bin", default=sys.executable, help="执行 OpenGait pretreatment 的 Python 可执行文件")
    parser.add_argument("--dataset-name", default="CASIAB", help="传给 OpenGait 的 dataset 名称，默认 CASIAB")
    parser.add_argument("--n-workers", type=int, default=4, help="pretreatment 并行 worker 数")
    parser.add_argument("--img-size", type=int, default=64, help="pretreatment 输出尺寸")
    parser.add_argument(
        "--log-file",
        default="",
        help="可选：pretreatment 日志路径；默认写到 <output-path>/../casia_b_pretreatment.log",
    )
    parser.add_argument("--verbose", action="store_true", help="把 --verbose 透传给 OpenGait pretreatment.py")
    parser.add_argument("--dry-run", action="store_true", help="显式声明只生成命令；默认也是 dry-run")
    parser.add_argument("--execute", action="store_true", help="真正执行 pretreatment 命令")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dry_run and args.execute:
        raise SystemExit("`--dry-run` 和 `--execute` 不能同时使用")

    plan = build_plan(args)
    issues = validate_plan(plan)
    dry_run = not args.execute

    print_report(plan, issues, dry_run=dry_run)

    if dry_run:
        return

    if issues:
        raise SystemExit("路径检查未通过，已停止执行。请先修正上面的目录问题。")

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(plan.command(), cwd=plan.opengait_root, check=False)
    raise SystemExit(completed.returncode)


def build_plan(args: argparse.Namespace) -> PretreatmentPlan:
    opengait_root = Path(args.opengait_root).expanduser().resolve()
    pretreatment_script = (
        Path(args.pretreatment_script).expanduser().resolve()
        if args.pretreatment_script
        else opengait_root / "datasets" / "pretreatment.py"
    )
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    log_file = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else output_path.parent / "casia_b_pretreatment.log"
    )
    return PretreatmentPlan(
        python_bin=Path(args.python_bin).expanduser(),
        opengait_root=opengait_root,
        pretreatment_script=pretreatment_script,
        input_path=input_path,
        output_path=output_path,
        log_file=log_file,
        dataset_name=args.dataset_name,
        n_workers=args.n_workers,
        img_size=args.img_size,
        verbose=args.verbose,
    )


def validate_plan(plan: PretreatmentPlan) -> list[str]:
    issues: list[str] = []
    if not plan.opengait_root.exists():
        issues.append(f"OpenGait 根目录不存在: {plan.opengait_root}")
    if not plan.pretreatment_script.exists():
        issues.append(f"pretreatment.py 不存在: {plan.pretreatment_script}")
    if not plan.input_path.exists():
        issues.append(f"CASIA-B 输入目录不存在: {plan.input_path}")
    elif not plan.input_path.is_dir():
        issues.append(f"CASIA-B 输入路径不是目录: {plan.input_path}")
    if plan.dataset_name.upper() != "CASIAB":
        issues.append(f"dataset 名称通常应为 CASIAB，当前为: {plan.dataset_name}")
    if plan.n_workers < 1:
        issues.append("n_workers 必须大于等于 1")
    if plan.img_size < 1:
        issues.append("img_size 必须大于等于 1")
    return issues


def print_report(plan: PretreatmentPlan, issues: list[str], *, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "execute"
    readiness = "ready" if not issues else "needs_attention"
    print(f"mode: {mode}")
    print(f"readiness: {readiness}")
    print(f"opengait_root: {plan.opengait_root}")
    print(f"pretreatment_script: {plan.pretreatment_script}")
    print(f"input_path: {plan.input_path}")
    print(f"output_path: {plan.output_path}")
    print(f"log_file: {plan.log_file}")
    print(f"dataset_name: {plan.dataset_name}")
    print(f"n_workers: {plan.n_workers}")
    print(f"img_size: {plan.img_size}")
    print(f"python_bin: {plan.python_bin}")
    print("command:")
    print(shlex.join(plan.command()))
    print("recommended_precheck:")
    print(
        shlex.join(
            [
                str(plan.python_bin),
                str(ROOT / "scripts" / "check_casia_b_layout.py"),
                "--input-dir",
                str(plan.input_path),
            ]
        )
    )
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("issues:")
        print("- none")


if __name__ == "__main__":
    main()
