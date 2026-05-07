from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opengait_runtime import default_project_python  # noqa: E402

DEFAULT_PYTHON = default_project_python("win11-demo")
DEFAULT_OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
DEFAULT_CONFIG = ROOT / "configs" / "opengait_casiab_baseline_small.yaml"
DEFAULT_REPORTS_DIR = ROOT / "reports"
DEFAULT_ENTRY = ROOT / "scripts" / "run_opengait_main.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于真实 CASIA-B-pkl 执行 OpenGait GPU probe / short-run")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="运行 OpenGait 的 Python")
    parser.add_argument("--entry-script", default=str(DEFAULT_ENTRY), help="训练入口包装脚本")
    parser.add_argument("--opengait-root", default=str(DEFAULT_OPENGAIT_ROOT), help="OpenGait 仓库根目录")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG), help="基础配置文件")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="报告输出目录")
    parser.add_argument("--mode", choices=["probe", "short-run", "baseline"], default="probe", help="执行模式")
    parser.add_argument("--total-iter", type=int, default=0, help="显式覆盖迭代数；0 表示按 mode 使用默认值")
    parser.add_argument("--save-name", default="", help="输出目录名；默认按 mode 自动命名")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数，默认 0")
    parser.add_argument("--master-port", type=int, default=29541, help="单机运行端口")
    parser.add_argument("--report-stem", default="", help="报告文件名前缀；默认按 mode 自动命名")
    parser.add_argument("--restore-hint", default="", help="可选：恢复训练的 checkpoint iteration 或 checkpoint 路径")
    parser.add_argument("--save-iter", type=int, default=0, help="可选：覆盖 save_iter")
    parser.add_argument("--log-iter", type=int, default=0, help="可选：覆盖 log_iter")
    parser.add_argument("--trainer-batch-p", type=int, default=0, help="可选：覆盖 TripletSampler 的 P")
    parser.add_argument("--trainer-batch-k", type=int, default=0, help="可选：覆盖 TripletSampler 的 K")
    parser.add_argument("--frames-num-fixed", type=int, default=0, help="可选：覆盖固定帧数")
    parser.add_argument("--frames-num-min", type=int, default=0, help="可选：覆盖最小帧数")
    parser.add_argument("--frames-num-max", type=int, default=0, help="可选：覆盖最大帧数")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    python_bin = normalize_path(args.python_bin)
    entry_script = normalize_path(args.entry_script)
    opengait_root = normalize_path(args.opengait_root)
    config_path = normalize_path(args.config_path)
    reports_dir = normalize_path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    total_iter = args.total_iter or (2 if mode == "probe" else 5 if mode == "short-run" else 20)
    if args.save_name:
        save_name = args.save_name
    elif mode == "probe":
        save_name = f"real_gpu_probe_{total_iter}iter"
    elif mode == "short-run":
        save_name = f"real_gpu_short_run_{total_iter}iter"
    else:
        save_name = f"baseline_small_real_{total_iter}iter"
    report_stem = args.report_stem or save_name
    master_port = choose_master_port(args.master_port)

    runtime_config_path = reports_dir / f"{report_stem}.yaml"
    runtime_cfg = build_runtime_config(
        base_config_path=config_path,
        opengait_root=opengait_root,
        total_iter=total_iter,
        save_name=save_name,
        num_workers=args.num_workers,
        restore_hint=parse_restore_hint(args.restore_hint),
        save_iter=args.save_iter,
        log_iter=args.log_iter,
        trainer_batch_p=args.trainer_batch_p,
        trainer_batch_k=args.trainer_batch_k,
        frames_num_fixed=args.frames_num_fixed,
        frames_num_min=args.frames_num_min,
        frames_num_max=args.frames_num_max,
    )
    runtime_config_path.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    command = [
        str(python_bin),
        str(entry_script),
        "--cfg-path",
        str(runtime_config_path),
        "--phase",
        "train",
        "--log-to-file",
        "--opengait-root",
        str(opengait_root),
        "--master-port",
        str(master_port),
    ]
    start_time = time.time()
    completed = subprocess.run(
        command,
        cwd=opengait_root,
        check=False,
        capture_output=True,
        text=True,
    )
    duration_seconds = round(time.time() - start_time, 2)

    output_dir = opengait_root / "output" / runtime_cfg["data_cfg"]["dataset_name"] / runtime_cfg["model_cfg"]["model"] / save_name
    checkpoints = sorted(str(path) for path in (output_dir / "checkpoints").glob("*.pt") if path.is_file())
    log_files = sorted(str(path) for path in (output_dir / "logs").glob("*.txt") if path.is_file())
    summary_files = sorted(str(path) for path in (output_dir / "summary").glob("*") if path.is_file())
    latest_log_path = Path(log_files[-1]) if log_files else None
    report = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "mode": mode,
        "python_bin": str(python_bin),
        "opengait_root": str(opengait_root),
        "base_config_path": str(config_path),
        "runtime_config_path": str(runtime_config_path),
        "dataset_root": runtime_cfg["data_cfg"]["dataset_root"],
        "dataset_partition": runtime_cfg["data_cfg"]["dataset_partition"],
        "save_name": save_name,
        "total_iter": total_iter,
        "num_workers": runtime_cfg["data_cfg"].get("num_workers", 0),
        "num_workers_override": args.num_workers,
        "restore_hint": runtime_cfg["trainer_cfg"]["restore_hint"],
        "master_port": master_port,
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "output_dir": str(output_dir),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "log_files": log_files,
        "summary_files": summary_files,
        "latest_log_path": str(latest_log_path) if latest_log_path else "",
        "latest_iteration_lines": extract_iteration_lines(latest_log_path),
        "stdout_tail": tail_text(completed.stdout, max_chars=12000),
        "stderr_tail": tail_text(completed.stderr, max_chars=12000),
    }
    write_reports(reports_dir, report, stem=report_stem)
    print_summary(report, reports_dir, stem=report_stem)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def normalize_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_master_port(preferred_port: int) -> int:
    if preferred_port > 0 and port_available(preferred_port):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_runtime_config(
    *,
    base_config_path: Path,
    opengait_root: Path,
    total_iter: int,
    save_name: str,
    num_workers: int,
    restore_hint: int | str | None,
    save_iter: int,
    log_iter: int,
    trainer_batch_p: int,
    trainer_batch_k: int,
    frames_num_fixed: int,
    frames_num_min: int,
    frames_num_max: int,
) -> dict:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["data_cfg"]["dataset_root"])
    dataset_partition = Path(cfg["data_cfg"]["dataset_partition"])
    if not dataset_root.is_absolute():
        dataset_root = (opengait_root / dataset_root).resolve()
    if not dataset_partition.is_absolute():
        dataset_partition = (opengait_root / dataset_partition).resolve()

    cfg["data_cfg"]["dataset_root"] = str(dataset_root)
    cfg["data_cfg"]["dataset_partition"] = str(dataset_partition)
    if num_workers > 0:
        cfg["data_cfg"]["num_workers"] = num_workers
    cfg["trainer_cfg"]["enable_float16"] = False
    cfg["trainer_cfg"]["sync_BN"] = False
    cfg["trainer_cfg"]["with_test"] = False
    cfg["trainer_cfg"]["total_iter"] = total_iter
    cfg["trainer_cfg"]["save_name"] = save_name
    cfg["trainer_cfg"]["restore_hint"] = 0 if restore_hint in (None, "", 0) else restore_hint
    cfg["trainer_cfg"]["optimizer_reset"] = False
    cfg["trainer_cfg"]["scheduler_reset"] = False
    cfg["evaluator_cfg"]["restore_hint"] = cfg["trainer_cfg"]["restore_hint"]
    cfg["evaluator_cfg"]["save_name"] = save_name
    if save_iter > 0:
        cfg["trainer_cfg"]["save_iter"] = save_iter
    if log_iter > 0:
        cfg["trainer_cfg"]["log_iter"] = log_iter
    sampler = cfg["trainer_cfg"]["sampler"]
    if trainer_batch_p > 0 or trainer_batch_k > 0:
        current_p, current_k = sampler["batch_size"]
        sampler["batch_size"] = [
            trainer_batch_p if trainer_batch_p > 0 else current_p,
            trainer_batch_k if trainer_batch_k > 0 else current_k,
        ]
    if frames_num_fixed > 0:
        sampler["frames_num_fixed"] = frames_num_fixed
    if frames_num_min > 0:
        sampler["frames_num_min"] = frames_num_min
    if frames_num_max > 0:
        sampler["frames_num_max"] = frames_num_max
    return cfg


def parse_restore_hint(raw_value: str) -> int | str | None:
    value = raw_value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return str(normalize_path(value))


def tail_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def extract_iteration_lines(log_path: Path | None, *, limit: int = 20) -> list[str]:
    if log_path is None or not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    matched = [line for line in lines if "Iteration " in line]
    return matched[-limit:]


def write_reports(reports_dir: Path, report: dict, *, stem: str) -> None:
    (reports_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OpenGait GPU Probe 报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 模式：`{report['mode']}`",
        f"- Python：`{report['python_bin']}`",
        f"- OpenGait：`{report['opengait_root']}`",
        f"- 基础配置：`{report['base_config_path']}`",
        f"- 运行配置：`{report['runtime_config_path']}`",
        f"- dataset_root：`{report['dataset_root']}`",
        f"- dataset_partition：`{report['dataset_partition']}`",
        f"- save_name：`{report['save_name']}`",
        f"- total_iter：`{report['total_iter']}`",
        f"- num_workers：`{report['num_workers']}`",
        f"- restore_hint：`{report['restore_hint']}`",
        f"- master_port：`{report['master_port']}`",
        f"- returncode：`{report['returncode']}`",
        f"- 耗时：`{report['duration_seconds']}` 秒",
        f"- 输出目录：`{report['output_dir']}`",
        f"- checkpoint 数：`{report['checkpoint_count']}`",
        "",
        "## 实际命令",
        "",
        "```bash",
        report["command"],
        "```",
        "",
        "## 产物",
        "",
    ]
    if report["checkpoints"]:
        for path in report["checkpoints"]:
            lines.append(f"- checkpoint: `{path}`")
    else:
        lines.append("- checkpoint: none")
    if report["log_files"]:
        for path in report["log_files"]:
            lines.append(f"- log: `{path}`")
    else:
        lines.append("- log: none")
    if report["summary_files"]:
        for path in report["summary_files"]:
            lines.append(f"- summary: `{path}`")
    else:
        lines.append("- summary: none")
    lines.extend(
        [
            "",
            "## 最新迭代日志",
            "",
        ]
    )
    if report["latest_iteration_lines"]:
        for line in report["latest_iteration_lines"]:
            lines.append(f"- `{line}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## stdout tail",
            "",
            "```text",
            report["stdout_tail"] or "(empty)",
            "```",
            "",
            "## stderr tail",
            "",
            "```text",
            report["stderr_tail"] or "(empty)",
            "```",
        ]
    )
    (reports_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict, reports_dir: Path, *, stem: str) -> None:
    print(f"status: {report['status']}")
    print(f"mode: {report['mode']}")
    print(f"dataset_root: {report['dataset_root']}")
    print(f"output_dir: {report['output_dir']}")
    print(f"checkpoint_count: {report['checkpoint_count']}")
    print(f"json_report: {reports_dir / f'{stem}.json'}")
    print(f"md_report: {reports_dir / f'{stem}.md'}")


if __name__ == "__main__":
    main()
