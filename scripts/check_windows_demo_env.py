from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"
DEFAULT_GALLERY_ROOT = ROOT / "data" / "private_gallery"
DEFAULT_JSON_PATH = ROOT / "reports" / "windows_env_check.json"
DEFAULT_TRAIN_ENTRY = ROOT / "scripts" / "run_opengait_main.py"
REQUIRED_IMPORTS = [
    ("cv2", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("Pillow", "PIL"),
    ("gradio", "gradio"),
    ("mediapipe", "mediapipe"),
    ("ultralytics", "ultralytics"),
    ("yaml", "yaml"),
    ("imageio", "imageio"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("einops", "einops"),
    ("kornia", "kornia"),
    ("tensorboard", "tensorboard"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 Win11 私有步态库 demo 的本地运行环境。")
    parser.add_argument("--repo-root", default=str(ROOT), help="项目根目录。")
    parser.add_argument("--cfg-path", default=str(DEFAULT_CFG_PATH), help="OpenGait 推理配置路径。")
    parser.add_argument("--checkpoint-iter", type=int, default=1000, help="默认检查的 checkpoint iter。")
    parser.add_argument("--gallery-root", default=str(DEFAULT_GALLERY_ROOT), help="私有步态库根目录。")
    parser.add_argument("--write-json", default=str(DEFAULT_JSON_PATH), help="可选：输出 JSON 报告路径。")
    parser.add_argument("--require-demo-checkpoint", action="store_true", help="把 GUI/demo 默认 checkpoint 视为必需项。")
    parser.add_argument("--strict", action="store_true", help="如果存在阻塞项则返回非 0。")
    return parser


def import_status(module_name: str) -> dict[str, Any]:
    if importlib.util.find_spec(module_name) is None:
        return {"status": "missing"}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive reporting
        return {"status": "error", "error": repr(exc)}
    return {"status": "ok", "version": getattr(module, "__version__", "unknown")}


def load_yaml(path: Path) -> dict[str, Any]:
    yaml_status = import_status("yaml")
    if yaml_status["status"] != "ok":
        raise RuntimeError("PyYAML 未安装，无法解析配置文件。")
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件不是 dict 结构: {path}")
    return data


def to_path_status(path: Path, *, required: bool = True) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "required": required,
    }


def expected_checkpoint_path(repo_root: Path, cfg_path: Path, checkpoint_iter: int) -> Path:
    cfg = load_yaml(cfg_path)
    dataset_name = str(cfg.get("data_cfg", {}).get("dataset_name", "CASIA-B"))
    model_name = str(cfg.get("model_cfg", {}).get("model", "GaitSet"))
    evaluator_cfg = cfg.get("evaluator_cfg", {}) or {}
    trainer_cfg = cfg.get("trainer_cfg", {}) or {}
    save_name = str(evaluator_cfg.get("save_name") or trainer_cfg.get("save_name") or "formal_conservative_real")
    restore_hint = int(checkpoint_iter or evaluator_cfg.get("restore_hint") or trainer_cfg.get("restore_hint") or 0)
    checkpoint_name = f"{save_name}-{restore_hint:05d}.pt"
    return (
        repo_root
        / "external"
        / "OpenGait"
        / "output"
        / dataset_name
        / model_name
        / save_name
        / "checkpoints"
        / checkpoint_name
    )


def inspect_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"status": "missing"}
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - defensive reporting
        return {"status": "error", "error": repr(exc)}

    distributed = getattr(torch, "distributed", None)
    nccl_available = False
    gloo_available = False
    if distributed is not None:
        nccl_checker = getattr(distributed, "is_nccl_available", None)
        gloo_checker = getattr(distributed, "is_gloo_available", None)
        nccl_available = bool(nccl_checker()) if callable(nccl_checker) else False
        gloo_available = bool(gloo_checker()) if callable(gloo_checker) else False

    result = {
        "status": "ok",
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "nccl_available": nccl_available,
        "gloo_available": gloo_available,
    }
    if result["cuda_available"]:
        result["device_name"] = str(torch.cuda.get_device_name(0))
    return result


def preferred_backend(torch_info: dict[str, Any]) -> str:
    if torch_info.get("status") != "ok":
        return "unknown"
    if not torch_info.get("cuda_available"):
        return "gloo"
    if sys.platform.startswith("win"):
        return "gloo"
    if not torch_info.get("nccl_available"):
        return "gloo"
    return "nccl"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    cfg_path = Path(args.cfg_path).expanduser().resolve()
    gallery_root = Path(args.gallery_root).expanduser().resolve()
    reports_dir = repo_root / "reports"
    opengait_root = repo_root / "external" / "OpenGait"
    datasets_root = repo_root / "datasets" / "processed" / "CASIA-B-pkl"
    checkpoint_path = expected_checkpoint_path(repo_root, cfg_path, int(args.checkpoint_iter)) if cfg_path.exists() else (
        opengait_root / "output" / "CASIA-B" / "GaitSet" / "formal_conservative_real" / "checkpoints" / "formal_conservative_real-01000.pt"
    )

    imports = {label: import_status(module_name) for label, module_name in REQUIRED_IMPORTS}
    torch_info = inspect_torch()
    python_version = sys.version_info
    python_ok = (python_version.major, python_version.minor) in {(3, 10), (3, 11)}

    path_report = {
        "repo_root": to_path_status(repo_root),
        "cfg_path": to_path_status(cfg_path),
        "reports_dir": to_path_status(reports_dir, required=False),
        "gallery_root": to_path_status(gallery_root, required=False),
        "train_entry": to_path_status(DEFAULT_TRAIN_ENTRY),
        "opengait_root": to_path_status(opengait_root),
        "opengait_main": to_path_status(opengait_root / "opengait" / "main.py"),
        "opengait_default_cfg": to_path_status(opengait_root / "configs" / "default.yaml"),
        "checkpoint": to_path_status(checkpoint_path, required=bool(args.require_demo_checkpoint)),
        "dataset_pkl_root": to_path_status(datasets_root, required=False),
        "yolo_weight": to_path_status(repo_root / "yolov8n.pt", required=False),
    }

    warnings: list[str] = []
    issues: list[str] = []

    if not python_ok:
        issues.append(
            f"当前 Python 版本为 {platform.python_version()}，推荐使用 3.10 或 3.11。"
        )

    missing_imports = [
        name for name, status in imports.items()
        if status["status"] != "ok"
    ]
    if missing_imports:
        issues.append(f"缺少或无法导入依赖：{', '.join(missing_imports)}")

    if torch_info.get("status") != "ok":
        issues.append("PyTorch 不可用，完整私有步态库 demo 无法启动。")
    else:
        if not torch_info.get("cuda_available"):
            issues.append(
                "当前 PyTorch 没有识别到 CUDA GPU。完整私有步态库 demo 依赖 CUDA；"
                " 如果你只想跑基础检测/跟踪流程，请改用 app.py。"
            )
        elif int(torch_info.get("device_count", 0)) < 1:
            issues.append("PyTorch 显示 CUDA 可用，但没有可用 GPU 设备。")
        if sys.platform.startswith("win") and not torch_info.get("gloo_available"):
            issues.append("当前 PyTorch 未提供 Gloo 分布式后端，Windows 上无法初始化 OpenGait 推理。")

    required_paths = ["repo_root", "cfg_path", "train_entry", "opengait_root", "opengait_main", "opengait_default_cfg"]
    if args.require_demo_checkpoint:
        required_paths.append("checkpoint")
    for name in required_paths:
        if not path_report[name]["exists"]:
            issues.append(f"缺少必需路径：{path_report[name]['path']}")

    if not args.require_demo_checkpoint and not path_report["checkpoint"]["exists"]:
        warnings.append(
            "当前未发现 GUI/demo 默认 checkpoint。"
            " 这不会阻塞从头训练，但会影响私有步态库 GUI 和基于现成模型的推理。"
        )

    if not path_report["yolo_weight"]["exists"]:
        warnings.append("未找到 yolov8n.pt，首次运行时会自动下载。")

    if not path_report["gallery_root"]["exists"]:
        warnings.append("当前还没有 data/private_gallery；第一次录入样本时会自动创建。")

    if not path_report["dataset_pkl_root"]["exists"]:
        warnings.append(
            "当前未发现 datasets/processed/CASIA-B-pkl。"
            " 这不会阻塞私有步态库 demo 启动，但会影响后续官方训练/评估。"
        )

    if shutil.which("nvidia-smi") is None:
        warnings.append("系统里未找到 nvidia-smi；如果 torch 也看不到 CUDA，请先检查显卡驱动。")

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_supported": python_ok,
        },
        "imports": imports,
        "torch": torch_info,
        "preferred_backend": preferred_backend(torch_info),
        "paths": path_report,
        "issues": issues,
        "warnings": warnings,
    }


def print_report(report: dict[str, Any]) -> None:
    print("# Win11 Demo 环境检查")
    print("")
    platform_info = report["platform"]
    print(f"- 系统：{platform_info['system']} {platform_info['release']} ({platform_info['machine']})")
    print(f"- Python：{platform_info['python_version']}")
    print(f"- Python 是否推荐：{'是' if platform_info['python_supported'] else '否'}")
    print(f"- OpenGait 进程组后端：{report['preferred_backend']}")
    print("")

    torch_info = report["torch"]
    print("## Torch")
    if torch_info.get("status") != "ok":
        print(f"- 状态：{torch_info.get('status')}")
    else:
        print(f"- torch：{torch_info['version']}")
        print(f"- CUDA 可用：{torch_info['cuda_available']}")
        print(f"- CUDA 版本：{torch_info['cuda_version']}")
        print(f"- GPU 数量：{torch_info['device_count']}")
        if torch_info.get("device_name"):
            print(f"- GPU 名称：{torch_info['device_name']}")
        print(f"- Gloo：{torch_info['gloo_available']}")
        print(f"- NCCL：{torch_info['nccl_available']}")
    print("")

    print("## 路径")
    for name, info in report["paths"].items():
        status = "ok" if info["exists"] else ("optional-missing" if not info["required"] else "missing")
        print(f"- {name}: {status} -> {info['path']}")
    print("")

    print("## 结果")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"- 阻塞项：{issue}")
    else:
        print("- 阻塞项：无")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"- 提示：{warning}")
    else:
        print("- 提示：无")


def main() -> int:
    args = build_arg_parser().parse_args()
    report = build_report(args)

    output_path = Path(args.write_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(report)
    print("")
    print(f"JSON 报告已写入：{output_path}")

    if args.strict and report["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
