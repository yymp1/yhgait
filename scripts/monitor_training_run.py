from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="轮询记录训练产物与 GPU 资源")
    parser.add_argument("--pid", type=int, required=True, help="训练进程 PID")
    parser.add_argument("--output-dir", required=True, help="OpenGait 输出目录")
    parser.add_argument("--log-path", required=True, help="CSV 日志输出路径")
    parser.add_argument("--interval", type=float, default=5.0, help="采样间隔秒数")
    return parser


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def latest_name(path: Path, pattern: str) -> str:
    if not path.exists():
        return ""
    items = sorted(item.name for item in path.glob(pattern) if item.is_file())
    return items[-1] if items else ""


def query_gpu() -> tuple[str, str, str] | tuple[str, str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return "", "", ""
    first = completed.stdout.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def pid_alive(pid: int) -> bool:
    completed = subprocess.run(["bash", "-lc", f"kill -0 {pid}"], check=False)
    return completed.returncode == 0


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    summary_dir = output_dir / "summary"
    log_path = Path(args.log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "alive",
                "checkpoint_count",
                "latest_checkpoint",
                "log_count",
                "latest_log",
                "summary_count",
                "latest_summary",
                "gpu_mem_mib",
                "gpu_util_percent",
                "gpu_temp_c",
            ]
        )
        while True:
            alive = pid_alive(args.pid)
            gpu_mem, gpu_util, gpu_temp = query_gpu()
            writer.writerow(
                [
                    int(time.time()),
                    int(alive),
                    count_files(checkpoints_dir, "*.pt"),
                    latest_name(checkpoints_dir, "*.pt"),
                    count_files(logs_dir, "*.txt"),
                    latest_name(logs_dir, "*.txt"),
                    count_files(summary_dir, "*"),
                    latest_name(summary_dir, "*"),
                    gpu_mem,
                    gpu_util,
                    gpu_temp,
                ]
            )
            handle.flush()
            if not alive:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
