#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opengait_runtime import add_opengait_to_path, init_single_process_distributed, resolve_from_opengait_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨平台单卡运行 OpenGait 训练/评估入口。")
    parser.add_argument(
        "--cfg-path",
        default=str(ROOT / "configs" / "opengait_casiab_smoke.yaml"),
        help="配置文件路径。",
    )
    parser.add_argument(
        "--phase",
        default="train",
        choices=["train", "test"],
        help="训练或评估。",
    )
    parser.add_argument(
        "--opengait-root",
        default=str(ROOT / "external" / "OpenGait"),
        help="OpenGait 仓库根目录。",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29521,
        help="单机单卡分布式端口。",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="可选：覆盖 trainer/evaluator 的 restore_hint。",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="是否把日志写入 OpenGait output 目录。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="可选：覆盖 data_cfg.num_workers。Windows 上推荐先用 0 或 1。",
    )
    return parser.parse_args()


def load_cfg(config_loader, cfg_path: Path, opengait_root: Path, restore_iter: int, num_workers: int) -> dict:
    previous_cwd = Path.cwd()
    os.chdir(opengait_root)
    try:
        cfgs = config_loader(str(cfg_path.resolve()))
    finally:
        os.chdir(previous_cwd)

    cfgs["data_cfg"]["dataset_root"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_root"], opengait_root)
    cfgs["data_cfg"]["dataset_partition"] = resolve_from_opengait_root(cfgs["data_cfg"]["dataset_partition"], opengait_root)
    if num_workers >= 0:
        cfgs["data_cfg"]["num_workers"] = num_workers

    if restore_iter:
        cfgs["evaluator_cfg"]["restore_hint"] = int(restore_iter)
        cfgs["trainer_cfg"]["restore_hint"] = int(restore_iter)

    import torch

    if torch.distributed.get_world_size() == 1:
        cfgs["evaluator_cfg"]["sampler"]["batch_size"] = 1
    return cfgs


def patch_single_process_runtime(torch_module) -> None:
    if torch_module.distributed.get_world_size() != 1:
        return

    import utils
    import utils.common as common
    import data.sampler as sampler_module

    common.get_ddp_module = lambda module, *args, **kwargs: module
    utils.get_ddp_module = common.get_ddp_module
    common.ddp_all_gather = lambda features, dim=0, requires_grad=True: features
    utils.ddp_all_gather = common.ddp_all_gather

    def local_sync_random_sample_list(obj_list, k, common_choice: bool = False):
        if common_choice or len(obj_list) < k:
            indices = random.choices(range(len(obj_list)), k=k)
        else:
            indices = torch_module.randperm(len(obj_list))[:k].tolist()
        return [obj_list[index] for index in indices]

    sampler_module.sync_random_sample_list = local_sync_random_sample_list


def init_logger(get_msg_mgr, cfgs: dict, training: bool, log_to_file: bool) -> None:
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs["trainer_cfg"] if training else cfgs["evaluator_cfg"]
    output_path = os.path.join(
        "output",
        cfgs["data_cfg"]["dataset_name"],
        cfgs["model_cfg"]["model"],
        engine_cfg["save_name"],
    )
    if training:
        msg_mgr.init_manager(
            output_path,
            log_to_file,
            engine_cfg["log_iter"],
            engine_cfg["restore_hint"] if isinstance(engine_cfg["restore_hint"], int) else 0,
        )
    else:
        msg_mgr.init_logger(output_path, log_to_file)
    msg_mgr.log_info(engine_cfg)


def main() -> int:
    args = parse_args()
    import torch

    opengait_root = Path(args.opengait_root).expanduser().resolve()
    cfg_path = Path(args.cfg_path).expanduser().resolve()

    if not torch.cuda.is_available():
        raise SystemExit("当前运行模式要求 CUDA GPU。请先确认 Win11 已安装 CUDA 版 PyTorch。")

    add_opengait_to_path(opengait_root)
    init_single_process_distributed(args.master_port)

    import utils
    from utils import config_loader, get_msg_mgr, init_seeds, params_count

    patch_single_process_runtime(torch)

    from modeling import models

    previous_cwd = Path.cwd()
    os.chdir(opengait_root)
    try:
        training = args.phase == "train"
        cfgs = load_cfg(config_loader, cfg_path, opengait_root, args.iter, args.num_workers)
        init_logger(get_msg_mgr, cfgs, training, args.log_to_file)
        init_seeds(torch.distributed.get_rank())

        model_cfg = cfgs["model_cfg"]
        msg_mgr = get_msg_mgr()
        msg_mgr.log_info(model_cfg)
        model_class = getattr(models, model_cfg["model"])
        model = model_class(cfgs, training=training)
        msg_mgr.log_info(params_count(model))
        msg_mgr.log_info("OpenGait single-process runtime initialized.")

        if training:
            model_class.run_train(model)
        else:
            model_class.run_test(model)
    finally:
        os.chdir(previous_cwd)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
