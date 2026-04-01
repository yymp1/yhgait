from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_DIR = Path.home() / ".venvs" / "person_orientation_demo" / "opengait_pretreatment"
MINIMAL_PACKAGES = [
    "opencv-python-headless",
    "numpy",
    "tqdm",
]
TRAIN_SMOKE_PACKAGES = [
    "pyyaml",
    "scikit-learn",
    "tensorboard",
    "torch",
    "torchvision",
    "einops",
    "kornia",
    "matplotlib",
    "imageio",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 OpenGait pretreatment 创建最小独立运行环境")
    parser.add_argument(
        "--venv-dir",
        default=str(DEFAULT_VENV_DIR),
        help="虚拟环境目录，默认 ~/.venvs/person_orientation_demo/opengait_pretreatment",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="如果环境已存在则先删除再重建",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="跳过 pip 自升级",
    )
    parser.add_argument(
        "--with-train-smoke",
        action="store_true",
        help="额外安装 OpenGait 最小训练 smoke test 所需依赖",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    venv_dir = Path(args.venv_dir).expanduser().resolve()

    create_venv(venv_dir, recreate=args.recreate)
    python_bin = venv_python(venv_dir)
    removed_before_install = cleanup_appledouble(venv_dir)

    if not args.skip_pip_upgrade:
        run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        removed_before_install += cleanup_appledouble(venv_dir)
    run([str(python_bin), "-m", "pip", "install", *MINIMAL_PACKAGES])
    if args.with_train_smoke:
        run([str(python_bin), "-m", "pip", "install", *TRAIN_SMOKE_PACKAGES])
    removed_after_install = cleanup_appledouble(venv_dir)

    import_report = verify_imports(python_bin, with_train_smoke=args.with_train_smoke)
    installed = freeze_packages(python_bin)

    print(f"venv_dir: {venv_dir}")
    print(f"python_bin: {python_bin}")
    print(f"removed_appledouble_files: {removed_before_install + removed_after_install}")
    print(f"installed_packages: {', '.join(installed)}")
    print("import_check:")
    for line in import_report:
        print(f"- {line}")


def create_venv(venv_dir: Path, *, recreate: bool) -> None:
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if venv_dir.exists():
        return
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=True)
    builder.create(venv_dir)


def venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def verify_imports(python_bin: Path, *, with_train_smoke: bool) -> list[str]:
    smoke_block = """
import imageio
import kornia
import matplotlib
import sklearn
import tensorboard
import torch
import torchvision
import yaml
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"yaml={yaml.__version__}")
print(f"tensorboard={tensorboard.__version__}")
print(f"kornia={kornia.__version__}")
print(f"matplotlib={matplotlib.__version__}")
print(f"imageio={imageio.__version__}")
print(f"sklearn={sklearn.__version__}")
""" if with_train_smoke else ""
    code = f"""
import cv2
import numpy
import tqdm
print(f"cv2={{cv2.__version__}}")
print(f"numpy={{numpy.__version__}}")
print(f"tqdm={{tqdm.__version__}}")
{smoke_block}
"""
    completed = run([str(python_bin), "-c", code], capture_output=True)
    return [line for line in completed.stdout.splitlines() if line.strip()]


def freeze_packages(python_bin: Path) -> list[str]:
    completed = run([str(python_bin), "-m", "pip", "freeze"], capture_output=True)
    prefixes = (
        "einops==",
        "imageio==",
        "kornia==",
        "matplotlib==",
        "numpy==",
        "opencv-python-headless==",
        "PyYAML==",
        "scikit-learn==",
        "tensorboard==",
        "torch==",
        "torchvision==",
        "tqdm==",
    )
    return [line for line in completed.stdout.splitlines() if line.startswith(prefixes)]


def cleanup_appledouble(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def run(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"$ {shlex.join(command)}")
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


if __name__ == "__main__":
    main()
