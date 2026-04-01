from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path.home() / ".venvs" / "person_orientation_demo" / "opengait_pretreatment" / "bin" / "python"
DEFAULT_OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
DEFAULT_CONFIG = ROOT / "configs" / "opengait_casiab_smoke.yaml"
DEFAULT_REPORTS_DIR = ROOT / "reports"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 macOS/CPU 上执行 OpenGait 的低风险兼容训练探针")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="运行探针的 Python")
    parser.add_argument("--opengait-root", default=str(DEFAULT_OPENGAIT_ROOT), help="OpenGait 仓库根目录")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG), help="基础配置文件")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="报告输出目录")
    parser.add_argument("--mode", choices=["smoke", "short-run"], default="smoke", help="smoke=1步验证；short-run=受限小跑")
    parser.add_argument("--total-iter", type=int, default=0, help="显式覆盖迭代数；0 表示按 mode 取默认值")
    parser.add_argument("--save-name", default="", help="输出目录名；默认按 mode 自动命名")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数，默认 0")
    parser.add_argument("--cleanup-appledouble", action="store_true", help="清理输出目录中的 ._* sidecar")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    python_bin = Path(args.python_bin).expanduser()
    opengait_root = Path(args.opengait_root).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_iter = args.total_iter or (1 if args.mode == "smoke" else 2)
    save_name = args.save_name or ("cpu_probe_smoke" if args.mode == "smoke" else "cpu_probe_short_run")

    run_report = run_probe(
        python_bin=python_bin,
        opengait_root=opengait_root,
        config_path=config_path,
        mode=args.mode,
        total_iter=total_iter,
        save_name=save_name,
        num_workers=args.num_workers,
        cleanup_appledouble=args.cleanup_appledouble,
    )
    write_reports(reports_dir, run_report)
    print_summary(run_report, reports_dir)


def run_probe(
    *,
    python_bin: Path,
    opengait_root: Path,
    config_path: Path,
    mode: str,
    total_iter: int,
    save_name: str,
    num_workers: int,
    cleanup_appledouble: bool,
) -> dict:
    code = f"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

opengait_root = Path({str(opengait_root)!r})
config_path = Path({str(config_path)!r})
mode = {mode!r}
total_iter = {total_iter!r}
save_name = {save_name!r}
num_workers = {num_workers!r}
cleanup_appledouble = {cleanup_appledouble!r}

sys.path.insert(0, str(opengait_root / "opengait"))

orig_module_to = nn.Module.to
orig_tensor_cuda = torch.Tensor.cuda
orig_module_cuda = nn.Module.cuda
orig_cuda_set_device = torch.cuda.set_device
orig_cuda_current_device = torch.cuda.current_device
orig_cuda_is_available = torch.cuda.is_available

def patched_module_to(self, *args, **kwargs):
    if args:
        target = args[0]
        if isinstance(target, torch.device) and target.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]
        elif isinstance(target, str) and target.startswith("cuda"):
            args = ("cpu",) + args[1:]
    if "device" in kwargs:
        target = kwargs["device"]
        if isinstance(target, torch.device) and target.type == "cuda":
            kwargs["device"] = torch.device("cpu")
        elif isinstance(target, str) and target.startswith("cuda"):
            kwargs["device"] = "cpu"
    return orig_module_to(self, *args, **kwargs)

def cleanup_sidecars(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed

def tail_text(path: Path, max_lines: int = 20):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]

nn.Module.to = patched_module_to
torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.current_device = lambda: 0
torch.cuda.is_available = lambda: False

init_dir = Path(tempfile.mkdtemp(prefix="opengait_cpu_probe_"))
init_file = init_dir / "dist_init"
start_time = time.time()

try:
    import utils
    import utils.common as common

    common.get_ddp_module = lambda module, *a, **k: module
    utils.get_ddp_module = common.get_ddp_module
    common.ddp_all_gather = lambda features, dim=0, requires_grad=True: features
    utils.ddp_all_gather = common.ddp_all_gather
    common.ts2var = lambda x, **kwargs: torch.autograd.Variable(x, **kwargs)
    common.np2var = lambda x, **kwargs: common.ts2var(torch.from_numpy(x), **kwargs)
    common.list2var = lambda x, **kwargs: common.np2var(__import__("numpy").array(x), **kwargs)
    utils.ts2var = common.ts2var
    utils.np2var = common.np2var
    utils.list2var = common.list2var

    from utils import config_loader, get_msg_mgr
    from modeling import models

    dist.init_process_group(backend="gloo", init_method=f"file://{{init_file}}", rank=0, world_size=1)

    cfg = config_loader(str(config_path))
    dataset_root = Path(cfg["data_cfg"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (opengait_root / dataset_root).resolve()
    dataset_partition = Path(cfg["data_cfg"]["dataset_partition"])
    if not dataset_partition.is_absolute():
        dataset_partition = (opengait_root / dataset_partition).resolve()

    cfg["data_cfg"]["dataset_root"] = str(dataset_root)
    cfg["data_cfg"]["dataset_partition"] = str(dataset_partition)
    cfg["data_cfg"]["num_workers"] = num_workers
    cfg["trainer_cfg"]["enable_float16"] = False
    cfg["trainer_cfg"]["sync_BN"] = False
    cfg["trainer_cfg"]["with_test"] = False
    cfg["trainer_cfg"]["total_iter"] = total_iter
    cfg["trainer_cfg"]["save_iter"] = 1
    cfg["trainer_cfg"]["save_name"] = save_name
    cfg["trainer_cfg"]["restore_hint"] = 0
    cfg["trainer_cfg"]["optimizer_reset"] = False
    cfg["trainer_cfg"]["scheduler_reset"] = False

    output_path = Path("output") / cfg["data_cfg"]["dataset_name"] / cfg["model_cfg"]["model"] / save_name
    output_path_abs = (opengait_root / output_path).resolve()
    if output_path_abs.exists():
        shutil.rmtree(output_path_abs)
    get_msg_mgr().init_manager(str(output_path), True, cfg["trainer_cfg"]["log_iter"], 0)

    Model = getattr(models, cfg["model_cfg"]["model"])
    model = Model(cfg, training=True)
    Model.run_train(model)

    checkpoints_dir = output_path_abs / "checkpoints"
    logs_dir = output_path_abs / "logs"
    summary_dir = output_path_abs / "summary"
    checkpoint_files = sorted(
        str(path) for path in checkpoints_dir.glob("*.pt")
        if path.is_file() and not path.name.startswith("._")
    )
    log_files = sorted(
        str(path) for path in logs_dir.glob("*.txt")
        if path.is_file() and not path.name.startswith("._")
    )
    summary_files = sorted(
        str(path) for path in summary_dir.glob("*")
        if path.is_file() and not path.name.startswith("._")
    )

    removed_sidecars = cleanup_sidecars(output_path_abs) if cleanup_appledouble else 0

    latest_log = Path(log_files[-1]) if log_files else None
    result = {{
        "status": "passed",
        "mode": mode,
        "total_iter": total_iter,
        "save_name": save_name,
        "dataset_root": str(dataset_root),
        "dataset_partition": str(dataset_partition),
        "output_path": str(output_path_abs),
        "iteration": model.iteration,
        "checkpoint_count": len(checkpoint_files),
        "checkpoint_files": checkpoint_files,
        "log_files": log_files,
        "summary_files": summary_files,
        "removed_appledouble_files": removed_sidecars,
        "latest_log_tail": tail_text(latest_log) if latest_log else [],
        "duration_seconds": round(time.time() - start_time, 2),
        "notes": [
            "这是本机 CPU/gloo 兼容探针，不是官方支持的 CUDA/NCCL 正式训练入口。",
            "探针通过 monkeypatch 绕过了 OpenGait 当前版本里硬编码的 cuda/ddp 包装逻辑。",
        ],
    }}
    print(json.dumps(result, ensure_ascii=False))
finally:
    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.destroy_process_group()
    nn.Module.to = orig_module_to
    torch.Tensor.cuda = orig_tensor_cuda
    nn.Module.cuda = orig_module_cuda
    torch.cuda.set_device = orig_cuda_set_device
    torch.cuda.current_device = orig_cuda_current_device
    torch.cuda.is_available = orig_cuda_is_available
    shutil.rmtree(init_dir, ignore_errors=True)
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        cwd=opengait_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "mode": mode,
            "total_iter": total_iter,
            "save_name": save_name,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    result = json.loads(completed.stdout.splitlines()[-1])
    result["stdout_prefix"] = completed.stdout.splitlines()[:-1]
    return result


def write_reports(reports_dir: Path, report: dict) -> None:
    suffix = "cpu_probe_smoke" if report["mode"] == "smoke" else "cpu_probe_short_run"
    json_path = reports_dir / f"{suffix}.json"
    md_path = reports_dir / f"{suffix}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# OpenGait CPU 兼容探针报告",
        "",
        f"- 模式：`{report['mode']}`",
        f"- 状态：`{report['status']}`",
        f"- 迭代数：`{report['total_iter']}`",
        f"- save_name：`{report['save_name']}`",
    ]
    if report["status"] == "passed":
        lines.extend(
            [
                f"- 输出目录：`{report['output_path']}`",
                f"- 实际迭代：`{report['iteration']}`",
                f"- checkpoint 数：`{report['checkpoint_count']}`",
                f"- 运行耗时：`{report['duration_seconds']}` 秒",
                f"- 清理的 `._*` 数量：`{report['removed_appledouble_files']}`",
                "",
                "## 说明",
            ]
        )
        for note in report.get("notes", []):
            lines.append(f"- {note}")
        lines.extend(
            [
                "",
                "## 产物",
                f"- checkpoints：`{report['checkpoint_files']}`",
                f"- logs：`{report['log_files']}`",
                f"- summary：`{report['summary_files']}`",
            ]
        )
        if report.get("latest_log_tail"):
            lines.extend(["", "## 最新日志尾部", "```text", *report["latest_log_tail"], "```"])
    else:
        lines.extend(["", "## stdout", "```text", report.get("stdout", "").strip(), "```", "", "## stderr", "```text", report.get("stderr", "").strip(), "```"])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict, reports_dir: Path) -> None:
    suffix = "cpu_probe_smoke" if report["mode"] == "smoke" else "cpu_probe_short_run"
    print(f"mode: {report['mode']}")
    print(f"status: {report['status']}")
    print(f"total_iter: {report['total_iter']}")
    if report["status"] == "passed":
        print(f"output_path: {report['output_path']}")
        print(f"iteration: {report['iteration']}")
        print(f"checkpoint_count: {report['checkpoint_count']}")
        print(f"duration_seconds: {report['duration_seconds']}")
    print(f"json_report: {reports_dir / f'{suffix}.json'}")
    print(f"md_report: {reports_dir / f'{suffix}.md'}")


if __name__ == "__main__":
    main()
