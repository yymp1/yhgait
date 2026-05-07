#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
OPENGAIT_ROOT = ROOT / "external" / "OpenGait"
OPENGAIT_PY_ROOT = OPENGAIT_ROOT / "opengait"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(OPENGAIT_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENGAIT_PY_ROOT))

from opengait_runtime import add_opengait_to_path, init_single_process_distributed  # noqa: E402

add_opengait_to_path(OPENGAIT_ROOT)

from modeling import models  # noqa: E402
from utils import config_loader, get_ddp_module, get_msg_mgr, init_seeds, params_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建步态检索 demo 的 gallery/probe embedding 缓存。")
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_formal_conservative_eval.yaml"),
        help="评估配置路径。",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=int,
        default=1000,
        help="要加载的 checkpoint iter。",
    )
    parser.add_argument(
        "--cache-path",
        default=str(ROOT / "reports" / "checkpoint_eval_compare" / ".." / "gait_demo_cache_formal_conservative_real_iter01000.npz"),
        help="输出缓存路径。",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="是否把 OpenGait 推理日志写入 output 目录。",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29701,
        help="单卡分布式初始化端口。",
    )
    return parser.parse_args()


def ensure_single_process_env(master_port: int) -> None:
    init_single_process_distributed(master_port)


def resolve_from_opengait_root(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((OPENGAIT_ROOT / path).resolve())


def load_cfg(cfg_path: Path, checkpoint_iter: int) -> dict:
    previous_cwd = Path.cwd()
    os.chdir(OPENGAIT_ROOT)
    try:
        cfgs = config_loader(str(cfg_path.resolve()))
    finally:
        os.chdir(previous_cwd)
    cfgs["data_cfg"]["dataset_root"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_root"])
    cfgs["data_cfg"]["dataset_partition"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_partition"])
    cfgs["evaluator_cfg"]["restore_hint"] = checkpoint_iter
    cfgs["trainer_cfg"]["restore_hint"] = checkpoint_iter
    return cfgs


def init_logger(cfgs: dict, log_to_file: bool) -> None:
    msg_mgr = get_msg_mgr()
    output_path = (
        OPENGAIT_ROOT
        / "output"
        / cfgs["data_cfg"]["dataset_name"]
        / cfgs["model_cfg"]["model"]
        / cfgs["evaluator_cfg"]["save_name"]
    )
    msg_mgr.init_logger(str(output_path), log_to_file)
    msg_mgr.log_info(cfgs["evaluator_cfg"])


def build_model(cfgs: dict, log_to_file: bool):
    init_logger(cfgs, log_to_file=log_to_file)
    init_seeds(torch.distributed.get_rank())
    model_cfg = cfgs["model_cfg"]
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(model_cfg)
    model_class = getattr(models, model_cfg["model"])
    model = model_class(cfgs, training=False)
    if torch.distributed.get_world_size() > 1:
        model = get_ddp_module(model, cfgs["trainer_cfg"]["find_unused_parameters"])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Feature Cache Model Initialization Finished!")
    return model


def build_cache(model, cfgs: dict, cache_path: Path) -> None:
    rank = torch.distributed.get_rank()
    with torch.no_grad():
        info_dict = model.inference(rank)
    if rank != 0:
        return

    loader = model.test_loader
    seqs_info = loader.dataset.seqs_info
    pkl_paths = []
    seq_keys = []
    for lab, seq_type, view, paths in seqs_info:
        pkl_path = paths[0] if paths else ""
        pkl_paths.append(pkl_path)
        seq_keys.append(f"{lab}|{seq_type}|{view}")

    checkpoint_path = (
        OPENGAIT_ROOT
        / "output"
        / cfgs["data_cfg"]["dataset_name"]
        / cfgs["model_cfg"]["model"]
        / cfgs["evaluator_cfg"]["save_name"]
        / "checkpoints"
        / f"{cfgs['evaluator_cfg']['save_name']}-{cfgs['evaluator_cfg']['restore_hint']:05d}.pt"
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(info_dict["embeddings"], dtype=np.float32),
        labels=np.asarray(loader.dataset.label_list),
        types=np.asarray(loader.dataset.types_list),
        views=np.asarray(loader.dataset.views_list),
        pkl_paths=np.asarray(pkl_paths),
        seq_keys=np.asarray(seq_keys),
        dataset_root=np.asarray(cfgs["data_cfg"]["dataset_root"]),
        dataset_partition=np.asarray(cfgs["data_cfg"]["dataset_partition"]),
        checkpoint_path=np.asarray(str(checkpoint_path.resolve())),
        metric=np.asarray(cfgs["evaluator_cfg"].get("metric", "euc")),
        save_name=np.asarray(cfgs["evaluator_cfg"]["save_name"]),
        restore_hint=np.asarray(cfgs["evaluator_cfg"]["restore_hint"]),
    )
    print(f"cache_written={cache_path}")
    print(f"sequence_count={len(loader.dataset.label_list)}")
    print(f"embedding_shape={np.asarray(info_dict['embeddings']).shape}")


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.cfg_path)
    cache_path = Path(args.cache_path).resolve()
    ensure_single_process_env(args.master_port)
    previous_cwd = Path.cwd()
    os.chdir(OPENGAIT_ROOT)
    try:
        cfgs = load_cfg(cfg_path, args.checkpoint_iter)
        model = build_model(cfgs, log_to_file=args.log_to_file)
        build_cache(model, cfgs, cache_path)
    finally:
        os.chdir(previous_cwd)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
