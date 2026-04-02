from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="收集 Linux GPU/CUDA/NCCL 本地兼容性信息")
    parser.add_argument("--python-bin", default="", help="可选：额外检查某个 Python 环境里的 torch")
    args = parser.parse_args()

    report = {
        "nvidia_smi": run(["nvidia-smi"]),
        "nvcc": run(["bash", "-lc", "which nvcc || true && nvcc --version || true"]),
        "cuda_dirs": run(["bash", "-lc", "ls -d /usr/local/cuda* 2>/dev/null || true"]),
        "ldconfig_cuda": run(["bash", "-lc", "ldconfig -p | rg 'nccl|cudart|cuda' | sed -n '1,200p'"]),
        "dpkg_gpu": run(["bash", "-lc", "dpkg -l | rg 'nvidia|cuda|nccl' | sed -n '1,260p'"]),
    }

    if args.python_bin:
        python_bin = str(Path(args.python_bin).expanduser().resolve())
        code = """
import json
import importlib.util

result = {"python_bin": __import__("sys").executable}
if importlib.util.find_spec("torch") is None:
    result["torch"] = {"status": "missing"}
else:
    import torch
    result["torch"] = {
        "status": "ok",
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "nccl_available": torch.distributed.is_nccl_available(),
    }
    if torch.cuda.is_available():
        result["torch"]["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(result, ensure_ascii=False))
"""
        report["python_probe"] = run([python_bin, "-c", code])

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
