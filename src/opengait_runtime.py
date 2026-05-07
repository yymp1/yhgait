from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
OPENGAIT_PY_ROOT = OPENGAIT_ROOT / "opengait"


def add_opengait_to_path(opengait_root: str | Path | None = None) -> Path:
    root = Path(opengait_root).resolve() if opengait_root else OPENGAIT_ROOT
    py_root = root / "opengait"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))
    return root


def venv_python_path(venv_dir: str | Path) -> Path:
    root = Path(venv_dir).expanduser().resolve()
    if sys.platform.startswith("win"):
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def default_project_python(venv_name: str = "win11-demo") -> Path:
    return venv_python_path(ROOT / ".venvs" / venv_name)


def pick_master_port(preferred_port: int) -> int:
    if os.environ.get("MASTER_PORT"):
        return int(os.environ["MASTER_PORT"])

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", int(preferred_port)))
        return int(preferred_port)
    except OSError:
        pass
    finally:
        probe.close()

    fallback = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        fallback.bind(("127.0.0.1", 0))
        return int(fallback.getsockname()[1])
    finally:
        fallback.close()


def ensure_single_process_env(master_port: int) -> int:
    port = pick_master_port(master_port)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    return port


def preferred_distributed_backend() -> str:
    import torch

    if not torch.cuda.is_available():
        return "gloo"
    if sys.platform.startswith("win"):
        return "gloo"
    if not torch.distributed.is_nccl_available():
        return "gloo"
    return "nccl"


def init_single_process_distributed(master_port: int) -> str:
    import torch

    ensure_single_process_env(master_port)
    backend = preferred_distributed_backend()
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend, init_method="env://")
    return backend


def resolve_from_opengait_root(path_text: str | Path, opengait_root: str | Path | None = None) -> str:
    root = Path(opengait_root).resolve() if opengait_root else OPENGAIT_ROOT
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())
