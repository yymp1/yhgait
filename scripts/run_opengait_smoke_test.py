from __future__ import annotations

import argparse
import json
import os
import socket
import shlex
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path.home() / ".venvs" / "person_orientation_demo" / "opengait_pretreatment" / "bin" / "python"
DEFAULT_OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
DEFAULT_CONFIG = ROOT / "configs" / "opengait_casiab_smoke.yaml"
DEFAULT_REPORTS_DIR = ROOT / "reports"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 OpenGait 最小训练 smoke test")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="运行 OpenGait smoke test 的 Python")
    parser.add_argument("--opengait-root", default=str(DEFAULT_OPENGAIT_ROOT), help="OpenGait 仓库根目录")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG), help="smoke test 配置文件")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="报告输出目录")
    parser.add_argument("--master-port", type=int, default=29531, help="单机 smoke test 的分布式端口")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    python_bin = normalize_python_bin(args.python_bin)
    opengait_root = Path(args.opengait_root).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    master_port = choose_master_port(args.master_port)

    import_check = run_import_check(python_bin)
    artifact_check = run_artifact_check(python_bin, opengait_root, config_path)
    data_check = run_data_smoke(python_bin, opengait_root, config_path)
    entry_check = run_entry_smoke(python_bin, opengait_root, config_path, master_port=master_port)

    report = {
        "python_bin": str(python_bin),
        "opengait_root": str(opengait_root),
        "config_path": str(config_path),
        "master_port": master_port,
        "import_check": import_check,
        "artifact_check": artifact_check,
        "data_smoke": data_check,
        "entry_smoke": entry_check,
    }
    write_reports(reports_dir, report)
    print_summary(report, reports_dir)


def normalize_python_bin(raw_path: str) -> Path:
    python_bin = Path(raw_path).expanduser()
    if not python_bin.is_absolute():
        python_bin = Path.cwd() / python_bin
    return python_bin


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


def run_import_check(python_bin: Path) -> dict:
    code = """
import importlib
import json

targets = ["torch", "yaml", "tensorboard", "torchvision", "einops", "kornia", "matplotlib", "imageio", "sklearn", "cv2", "numpy", "tqdm"]
report = {}
for name in targets:
    try:
        module = importlib.import_module(name)
        report[name] = {"status": "ok", "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        report[name] = {"status": "fail", "error": repr(exc)}
print(json.dumps(report, ensure_ascii=False))
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    modules = json.loads(completed.stdout)
    overall = "passed" if all(info.get("status") == "ok" for info in modules.values()) else "failed"
    return {"status": overall, "modules": modules}


def run_artifact_check(python_bin: Path, opengait_root: Path, config_path: Path) -> dict:
    code = f"""
import json
import pickle
import sys
from pathlib import Path

opengait_root = Path({str(opengait_root)!r})
config_path = Path({str(config_path)!r})
sys.path.insert(0, str(opengait_root / "opengait"))

from utils import config_loader

cfgs = config_loader(str(config_path))
dataset_root = Path(cfgs["data_cfg"]["dataset_root"])
if not dataset_root.is_absolute():
    dataset_root = (opengait_root / dataset_root).resolve()

subject_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
type_names = set()
view_names = set()
for subject_dir in subject_dirs:
    for type_dir in subject_dir.iterdir():
        if type_dir.is_dir():
            type_names.add(type_dir.name)
            for view_dir in type_dir.iterdir():
                if view_dir.is_dir():
                    view_names.add(view_dir.name)

pkl_paths = sorted(
    path for path in dataset_root.rglob("*.pkl")
    if path.is_file() and not path.name.startswith("._")
)
sample_indices = []
if pkl_paths:
    sample_indices = sorted(set([0, len(pkl_paths) // 2, len(pkl_paths) - 1]))

samples = []
for idx in sample_indices:
    sample_path = pkl_paths[idx]
    with sample_path.open("rb") as handle:
        data = pickle.load(handle)
    samples.append({{
        "path": str(sample_path),
        "shape": list(getattr(data, "shape", [])),
        "dtype": str(getattr(data, "dtype", "unknown")),
    }})

result = {{
    "dataset_root": str(dataset_root),
    "subject_count": len(subject_dirs),
    "type_count": len(type_names),
    "view_count": len(view_names),
    "pkl_count": len(pkl_paths),
    "samples": samples,
}}
print(json.dumps(result, ensure_ascii=False))
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        cwd=opengait_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    result = json.loads(completed.stdout)
    status = "passed" if result["subject_count"] > 0 and result["pkl_count"] > 0 else "failed"
    return {"status": status, "result": result}


def run_data_smoke(python_bin: Path, opengait_root: Path, config_path: Path) -> dict:
    init_dir = Path(tempfile.mkdtemp(prefix="opengait_data_smoke_"))
    init_file = init_dir / "dist_init"
    code = f"""
import json
import sys
from pathlib import Path

opengait_root = Path({str(opengait_root)!r})
config_path = Path({str(config_path)!r})
sys.path.insert(0, str(opengait_root / "opengait"))

import torch.distributed as dist
from utils import config_loader, get_msg_mgr
from data.dataset import DataSet
from data.collate_fn import CollateFn

dist.init_process_group(
    backend="gloo",
    init_method={f"file://{init_file}"!r},
    rank=0,
    world_size=1,
)
get_msg_mgr().init_logger(str(opengait_root / "output" / "smoke_data_check"), False)
cfgs = config_loader(str(config_path))
train_set = DataSet(cfgs["data_cfg"], training=True)
test_set = DataSet(cfgs["data_cfg"], training=False)
first = train_set[0]
second = train_set[1]
collate = CollateFn(train_set.label_set, cfgs["trainer_cfg"]["sampler"])
batch = collate([first, second])

feature_shapes = []
for feature_group in batch[0]:
    group_shapes = []
    for item in feature_group:
        group_shapes.append(list(item.shape))
    feature_shapes.append(group_shapes)

result = {{
    "train_seq_count": len(train_set),
    "test_seq_count": len(test_set),
    "label_count": len(train_set.label_set),
    "sample_seq_info": first[1],
    "batch_feature_shapes": feature_shapes,
    "batch_label_count": len(batch[1]),
    "batch_seqL": None if batch[-1] is None else batch[-1].tolist(),
}}
print(json.dumps(result, ensure_ascii=False))
dist.destroy_process_group()
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        cwd=opengait_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"status": "failed", "stdout": completed.stdout, "stderr": completed.stderr}
    return {"status": "passed", "result": json.loads(completed.stdout)}


def run_entry_smoke(python_bin: Path, opengait_root: Path, config_path: Path, *, master_port: int) -> dict:
    command = [
        str(python_bin),
        "opengait/main.py",
        "--cfgs",
        str(config_path),
        "--phase",
        "train",
        "--log_to_file",
    ]
    env = os.environ.copy()
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }
    )
    completed = subprocess.run(
        command,
        cwd=opengait_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    merged = (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or "")
    status = "passed" if completed.returncode == 0 else "failed"
    return {
        "status": status,
        "returncode": completed.returncode,
        "command": shlex.join(command),
        "output": merged[-12000:],
    }


def write_reports(reports_dir: Path, report: dict) -> None:
    (reports_dir / "opengait_smoke_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# OpenGait Smoke Test 报告",
        "",
        f"- Python: `{report['python_bin']}`",
        f"- OpenGait: `{report['opengait_root']}`",
        f"- 配置: `{report['config_path']}`",
        f"- 训练入口端口: `{report['master_port']}`",
        "",
        "## 依赖检查",
        f"- 状态：`{report['import_check']['status']}`",
    ]
    modules = report["import_check"].get("modules", {})
    if modules:
        for name, info in modules.items():
            detail = info.get("version", info.get("error", ""))
            lines.append(f"- `{name}`: `{info['status']}` {detail}")

    lines.extend(
        [
            "",
            "## Pretreatment 产物检查",
            f"- 状态：`{report['artifact_check']['status']}`",
        ]
    )
    if report["artifact_check"]["status"] == "passed":
        result = report["artifact_check"]["result"]
        lines.extend(
            [
                f"- 数据根目录：`{result['dataset_root']}`",
                f"- subject 数：`{result['subject_count']}`",
                f"- type 数：`{result['type_count']}`",
                f"- view 数：`{result['view_count']}`",
                f"- pkl 数：`{result['pkl_count']}`",
            ]
        )
        if result["samples"]:
            lines.append("- 抽样：")
            for sample in result["samples"]:
                lines.append(f"  - `{sample['path']}` shape={sample['shape']} dtype={sample['dtype']}")
    else:
        lines.append("```text")
        lines.append((report["artifact_check"].get("stdout", "") + report["artifact_check"].get("stderr", "")).strip())
        lines.append("```")

    lines.extend(
        [
            "",
            "## 数据集 Smoke",
            f"- 状态：`{report['data_smoke']['status']}`",
        ]
    )
    if report["data_smoke"]["status"] in {"ok", "passed"}:
        result = report["data_smoke"]["result"]
        lines.extend(
            [
                f"- train 序列数：`{result['train_seq_count']}`",
                f"- test 序列数：`{result['test_seq_count']}`",
                f"- train 身份数：`{result['label_count']}`",
                f"- 样例序列：`{result['sample_seq_info']}`",
                f"- batch 形状：`{result['batch_feature_shapes']}`",
            ]
        )
    else:
        lines.append("```text")
        lines.append((report["data_smoke"].get("stdout", "") + report["data_smoke"].get("stderr", "")).strip())
        lines.append("```")

    lines.extend(
        [
            "",
            "## 官方训练入口 Smoke",
            f"- 状态：`{report['entry_smoke']['status']}`",
            f"- 返回码：`{report['entry_smoke']['returncode']}`",
            "",
            "```bash",
            report["entry_smoke"]["command"],
            "```",
            "",
            "```text",
            report["entry_smoke"]["output"].strip(),
            "```",
        ]
    )
    (reports_dir / "opengait_smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict, reports_dir: Path) -> None:
    print(f"python_bin: {report['python_bin']}")
    print(f"config_path: {report['config_path']}")
    print(f"master_port: {report['master_port']}")
    print(f"import_check: {report['import_check']['status']}")
    print(f"artifact_check: {report['artifact_check']['status']}")
    print(f"data_smoke: {report['data_smoke']['status']}")
    print(f"entry_smoke: {report['entry_smoke']['status']} (returncode={report['entry_smoke']['returncode']})")
    print(f"json_report: {reports_dir / 'opengait_smoke_test.json'}")
    print(f"md_report: {reports_dir / 'opengait_smoke_test.md'}")


if __name__ == "__main__":
    main()
