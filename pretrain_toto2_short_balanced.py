from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# Huawei Ascend migration hook.  The forecast_pretrain image described by the
# platform ships matching torch/torch_npu builds; CUDA remains the fallback for
# the original environment and for local development.
try:
    import torch_npu
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False
else:
    NPU_AVAILABLE = torch_npu.npu.is_available()
    if NPU_AVAILABLE:
        try:
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "torch_npu is available but torch_npu.contrib.transfer_to_npu is missing; "
                "install the torch_npu build matching the platform Torch version."
            ) from exc
        torch.npu.set_compile_mode(jit_compile=False)
        # dion 优化器内部用 @torch.compile(fullgraph=True) 装饰更新 kernel，
        # 昇腾上 Dynamo 追踪这些 kernel 会报错（例如 muon_update_pre_orthogonalize
        # 里 NoneType 与 tuple 做 '>=' 比较），因此全局禁用 TorchDynamo，
        # 让所有优化器 kernel 走纯 eager。这与上方跳过 Triton/jit_compile 的意图一致。
        import torch._dynamo
        torch._dynamo.config.disable = True

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def add_source_paths(args: argparse.Namespace) -> None:
    for source_path in (
        args.chronos_repo,
        args.toto2_src,
        args.dd_unit_scaling_src,
    ):
        resolved = str(Path(source_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def ensure_dion() -> str:
    """Prefer an installed Dion; use the bundled Torch 2.4 fallback only if absent."""
    try:
        module = importlib.import_module("dion")
    except ModuleNotFoundError as exc:
        if exc.name != "dion":
            raise
        vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "dion"
        if not (vendor_root / "dion" / "__init__.py").is_file():
            raise RuntimeError(f"Dion is not installed and the bundled fallback is missing: {vendor_root}") from exc
        # Append, never prepend: site-packages and any environment-installed
        # Dion keep priority over the bundled fallback.
        sys.path.append(str(vendor_root))
        module = importlib.import_module("dion")
    if not hasattr(module, "NorMuon"):
        raise ImportError(f"Dion at {module.__file__} does not export NorMuon")
    return str(Path(module.__file__).resolve())


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue pretraining Toto-2 22M with dynamic 8192-point CPM sequences."
    )
    parser.add_argument("--model_name_or_path", default="/home/Toto-2.0-22m")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./output/toto2-22m-short-balanced")
    parser.add_argument("--chronos_repo", default="/home/chronos-forecasting-main")
    parser.add_argument("--toto2_src", default="/home/toto-main/toto2")
    parser.add_argument("--dd_unit_scaling_src", default="/home/toto-main/dd_unit_scaling")

    parser.add_argument("--data_dir", default="/data/GIFTEvalPretrain")
    parser.add_argument("--gift_eval_path", default="/data/GIFTEval")
    parser.add_argument("--gift_eval_src", default="/home/chronos-forecasting-main/gift-eval/src")
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--cpm_max_span_patches", type=int, default=6)
    parser.add_argument("--cpm_max_mask_probability", type=float, default=0.4)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--stop_after_steps", type=int, default=27000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--decay_steps", type=int, default=0)
    parser.add_argument("--scheduler", choices=("cosine",), default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--normuon_learning_rate", type=float, default=3.25e-3)
    parser.add_argument("--adamw_learning_rate", type=float, default=6e-5)
    parser.add_argument("--normuon_min_lr", type=float, default=3.25e-4)
    parser.add_argument("--adamw_min_lr", type=float, default=6e-6)
    parser.add_argument("--normuon_mu", type=float, default=0.96)
    parser.add_argument("--normuon_beta2", type=float, default=0.999)
    parser.add_argument("--adamw_beta1", type=float, default=0.91)
    parser.add_argument("--adamw_beta2", type=float, default=0.972)
    parser.add_argument("--normuon_weight_decay", type=float, default=2e-8)
    parser.add_argument("--max_grad_norm", type=float, default=7.0)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--tf32", type=parse_bool, default=True)
    parser.add_argument("--freeze_variate_layers", type=parse_bool, default=True)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", type=parse_bool, default=True)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)

    parser.add_argument("--stream_sample_pool_size", type=int, default=61440)
    parser.add_argument("--stream_samples_per_arrow", type=int, default=512)
    parser.add_argument("--stream_samples_per_dataset_block", type=int, default=256)
    parser.add_argument("--stream_arrow_row_chunk_size", type=int, default=4096)
    parser.add_argument("--stream_windows_per_series", type=int, default=16)
    parser.add_argument("--stream_prefetch_pools", type=int, default=16)
    parser.add_argument("--stream_arrow_cache_size", type=int, default=1024)
    parser.add_argument("--stream_log_interval", type=int, default=4096)

    return parser.parse_args()


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    accelerator = torch.npu if NPU_AVAILABLE else torch.cuda
    device_type = "npu" if NPU_AVAILABLE else "cuda"
    # Some schedulers expose only one device to each process, while preserving
    # its node-wide LOCAL_RANK. CUDA indices are then local to that one device.
    if accelerator.device_count() == 1:
        local_rank = 0
    if world_size > 1:
        # 跨节点（如 4×8）下 HCCL 走 RoCE，叠加 obsfs 并发 IO 抖动，默认 30 分钟
        # 超时容易在集合通信等待处触发 EI0006。显式放宽到 2 小时覆盖慢 IO 窗口。
        dist.init_process_group(
            backend="hccl" if NPU_AVAILABLE else "nccl",
            timeout=datetime.timedelta(hours=2),
        )
    accelerator.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(device_type, local_rank)


def all_reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def freeze_variate_attention_layers(model: torch.nn.Module) -> list[int]:
    frozen_indices: list[int] = []
    transformer = model.transformer
    for layer_index, layer in enumerate(transformer.layers):
        if transformer._if_variate_layer(layer_index):
            layer.requires_grad_(False)
            frozen_indices.append(layer_index)
    return frozen_indices


def split_optimizer_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    normuon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Parameter {name} appeared more than once.")
        seen.add(id(parameter))
        mup_type = getattr(parameter, "mup_type", None)
        is_io_projection = name.startswith("patch_proj.") or name.startswith("output_head.")
        use_normuon = parameter.ndim == 2 and mup_type == "weight" and not is_io_projection
        (normuon_params if use_normuon else adamw_params).append(parameter)
    if not normuon_params or not adamw_params:
        raise RuntimeError(
            f"Invalid optimizer split: NorMuon={len(normuon_params)}, AdamW={len(adamw_params)}"
        )
    return normuon_params, adamw_params


def eager_polar_express(gradient: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """NPU-safe Polar Express without Triton or torch.compile."""
    coefficients = (
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    )
    x = gradient.to(dtype=torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon)
    if gradient.size(-2) > gradient.size(-1):
        for a, b, c in coefficients:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in coefficients:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def lr_factor(step: int, *, max_steps: int, warmup_steps: int, decay_steps: int, min_ratio: float) -> float:
    """Linear warmup followed by cosine decay to min_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-12)
    decay_span = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps + 1) / decay_span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    max_steps: int,
    warmup_steps: int,
    decay_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0.0 <= min_lr <= base_lr:
        raise ValueError(f"Expected 0 <= min_lr <= base_lr, got {min_lr} and {base_lr}.")
    min_ratio = min_lr / base_lr if base_lr else 0.0
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_ratio=min_ratio,
        ),
    )


def sample_contiguous_patch_mask(
    target_mask: torch.Tensor,
    *,
    patch_size: int,
    max_span_patches: int,
    max_mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_span_patches < 1:
        raise ValueError("max_span_patches must be positive.")
    if not 0.0 < max_mask_probability <= 1.0:
        raise ValueError("max_mask_probability must be in (0, 1].")
    if target_mask.shape[-1] % patch_size:
        raise ValueError("The target length must be divisible by patch_size.")

    patch_valid = target_mask.unflatten(
        -1, (target_mask.shape[-1] // patch_size, patch_size)
    ).any(dim=-1)
    masked_patches = torch.zeros_like(patch_valid)
    flat_valid = patch_valid.reshape(-1, patch_valid.shape[-1])
    flat_masked = masked_patches.reshape(-1, masked_patches.shape[-1])

    for row in range(flat_valid.shape[0]):
        valid_indices = torch.where(flat_valid[row])[0]
        if valid_indices.numel() < 2:
            continue
        first_valid = int(valid_indices[0].item())
        last_valid = int(valid_indices[-1].item())
        eligible = flat_valid[row].clone()
        eligible[: first_valid + 1] = False
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            continue

        probability = float(torch.rand((), device=target_mask.device).item())
        probability *= max_mask_probability
        target_count = max(1, int(round(probability * eligible_count)))
        attempts = 0
        max_attempts = 8 * patch_valid.shape[-1]
        while (
            int((flat_masked[row] & eligible).sum().item()) < target_count
            and attempts < max_attempts
        ):
            span = int(
                torch.randint(
                    1,
                    max_span_patches + 1,
                    (),
                    device=target_mask.device,
                ).item()
            )
            start = int(
                torch.randint(
                    first_valid + 1,
                    last_valid + 1,
                    (),
                    device=target_mask.device,
                ).item()
            )
            flat_masked[row, start : min(last_valid + 1, start + span)] = True
            attempts += 1

        if not (flat_masked[row] & eligible).any():
            fallback = torch.where(eligible)[0]
            choice = int(
                fallback[
                    torch.randint(
                        0, fallback.numel(), (), device=target_mask.device
                    )
                ].item()
            )
            flat_masked[row, choice] = True

    cpm_mask = (~masked_patches).repeat_interleave(patch_size, dim=-1)
    return cpm_mask, masked_patches


def prepare_toto_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    patch_size: int = 32,
    cpm_max_span_patches: int = 16,
    cpm_max_mask_probability: float = 0.4,
) -> dict[str, torch.Tensor]:
    raw_target = batch["target"].to(
        device=device, dtype=torch.float32, non_blocking=True
    ).unsqueeze(1)
    target_mask = batch["target_mask"].to(
        device=device, non_blocking=True
    ).bool().unsqueeze(1)
    target = torch.nan_to_num(raw_target, nan=0.0, posinf=0.0, neginf=0.0)
    cpm_mask, masked_patches = sample_contiguous_patch_mask(
        target_mask,
        patch_size=patch_size,
        max_span_patches=cpm_max_span_patches,
        max_mask_probability=cpm_max_mask_probability,
    )
    if not masked_patches.any():
        raise RuntimeError("The current batch has no CPM-supervised patch.")
    series_ids = torch.zeros((target.shape[0], 1), dtype=torch.long, device=device)
    return {
        "target": target,
        "target_mask": target_mask,
        "cpm_mask": cpm_mask,
        "masked_patches": masked_patches,
        "series_ids": series_ids,
    }


def toto_pinball_loss(
    outputs,
    prepared: dict[str, torch.Tensor],
    *,
    patch_size: int,
    world_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = prepared["target"]
    target_mask = prepared["target_mask"]
    total_length = target.shape[-1]
    if total_length % patch_size:
        raise ValueError(
            f"Toto inputs must be patch aligned: total={total_length}, patch={patch_size}."
        )

    num_patches = total_length // patch_size
    # Output patch i predicts target patch i+1. CPM supervision is therefore
    # selected with masked_patches[..., 1:] against outputs[..., :-1, :].
    predicted = outputs.quantiles[..., :-1, :]
    scaled_target = torch.asinh((target - outputs.loc) / outputs.scale)
    target_patches = scaled_target.unflatten(-1, (num_patches, patch_size))[
        ..., 1:, :
    ]
    point_valid = target_mask.unflatten(-1, (num_patches, patch_size))[
        ..., 1:, :
    ]
    supervised_patch = prepared["masked_patches"][..., 1:]
    valid = point_valid & supervised_patch.unsqueeze(-1)
    if predicted.shape[1:] != target_patches.shape:
        raise RuntimeError(
            f"Unexpected Toto output shape {tuple(predicted.shape)} for "
            f"target patches {tuple(target_patches.shape)}."
        )

    quantile_levels = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device=predicted.device,
        dtype=predicted.dtype,
    ).view(-1, *([1] * (predicted.ndim - 1)))
    errors = target_patches.unsqueeze(0) - predicted
    pinball = torch.maximum(quantile_levels * errors, (quantile_levels - 1.0) * errors)
    point_loss = pinball.mean(dim=0, dtype=torch.float32)
    valid_count_per_patch = valid.sum(dim=-1)
    valid_supervised_patch = supervised_patch & (valid_count_per_patch > 0)
    patch_loss = (point_loss * valid).sum(dim=-1) / valid_count_per_patch.clamp_min(1)
    local_loss_sum = (patch_loss * valid_supervised_patch).sum(dtype=torch.float32)
    local_patch_count = valid_supervised_patch.sum(dtype=torch.float32)
    global_patch_count = all_reduce_sum(
        local_patch_count.detach().clone(), world_size
    )
    if global_patch_count.item() <= 0:
        raise RuntimeError("The current distributed micro-batch has no valid CPM targets.")

    # DDP averages gradients. Multiplying by world_size makes the resulting
    # gradient equal to a global average over the sampled CPM patch set M.
    loss = local_loss_sum * world_size / global_patch_count
    with torch.no_grad():
        global_loss_sum = all_reduce_sum(local_loss_sum.detach().clone(), world_size)
        global_loss = global_loss_sum / global_patch_count
        global_valid_points = all_reduce_sum(
            valid.sum(dtype=torch.float32), world_size
        )
    return loss, {
        "loss": global_loss,
        "valid_masked_points": global_valid_points,
        "masked_patches": global_patch_count,
    }


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    step: int,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    schedulers: tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    accelerator = torch.npu if NPU_AVAILABLE else torch.cuda
    state = {
        "global_step": step,
        "normuon_optimizer": optimizers[0].state_dict(),
        "adamw_optimizer": optimizers[1].state_dict(),
        "normuon_scheduler": schedulers[0].state_dict(),
        "adamw_scheduler": schedulers[1].state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state_all": accelerator.get_rng_state_all(),
    }
    torch.save(state, checkpoint_dir / "training_state.pt")
    (checkpoint_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
    )


def load_training_state(
    checkpoint: str | None,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    schedulers: tuple[torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.LambdaLR],
) -> int:
    if checkpoint is None:
        return 0
    state_path = Path(checkpoint) / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing resume state: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    optimizers[0].load_state_dict(state["normuon_optimizer"])
    optimizers[1].load_state_dict(state["adamw_optimizer"])
    schedulers[0].load_state_dict(state["normuon_scheduler"])
    schedulers[1].load_state_dict(state["adamw_scheduler"])
    torch.set_rng_state(state["torch_rng_state"])
    accelerator = torch.npu if NPU_AVAILABLE else torch.cuda
    rng_state = state.get("accelerator_rng_state_all", state.get("cuda_rng_state_all"))
    if accelerator.is_available() and rng_state is not None:
        accelerator.set_rng_state_all(rng_state)
    return int(state["global_step"])


def build_dataloader(args: argparse.Namespace, start_step: int) -> DataLoader:
    from load_gifteval_toto2_cpm_pretrain import (
        create_toto2_cpm_pretraining_dataset,
    )

    dataset = create_toto2_cpm_pretraining_dataset(
        data_dir=args.data_dir,
        gift_eval_path=args.gift_eval_path,
        gift_eval_src=args.gift_eval_src,
        batch_size=args.per_device_train_batch_size,
        patch_size=args.patch_size,
        max_sequence_length=args.max_sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        start_optimizer_step=start_step,
        seed=args.seed,
        sample_pool_size=args.stream_sample_pool_size,
        samples_per_arrow=args.stream_samples_per_arrow,
        samples_per_dataset_block=args.stream_samples_per_dataset_block,
        arrow_row_chunk_size=args.stream_arrow_row_chunk_size,
        windows_per_series=args.stream_windows_per_series,
        prefetch_pools=args.stream_prefetch_pools,
        arrow_cache_size=args.stream_arrow_cache_size,
        log_interval=args.stream_log_interval,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    # Validate the FINAL argparse values after Bash/platform overrides.
    output_check = Path(args.output_dir)
    if output_check.exists() and any(output_check.iterdir()):
        raise ValueError(f"Use a new empty output directory: {output_check}")
    if args.resume_from_checkpoint:
        if Path(args.model_name_or_path).resolve() != Path(args.resume_from_checkpoint).resolve():
            raise ValueError("Model and optimizer must resume from the same checkpoint")
    if not 0 <= args.warmup_steps < args.max_steps or not 0 < args.stop_after_steps <= args.max_steps:
        raise ValueError("Invalid warmup/max_steps/stop_after_steps")
    add_source_paths(args)
    dion_source = ensure_dion()
    import dd_unit_scaling as uu
    from toto2 import Toto2Model

    rank, local_rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    accelerator = torch.npu if NPU_AVAILABLE else torch.cuda
    accelerator.manual_seed_all(args.seed + rank)
    if not NPU_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    model_path = args.resume_from_checkpoint or args.model_name_or_path
    model = Toto2Model.from_pretrained(model_path, map_location="cpu", mmap=False)
    args.patch_size = int(model.config.patch_size)
    if args.max_sequence_length % args.patch_size:
        raise ValueError(
            f"max_sequence_length={args.max_sequence_length} must be divisible by "
            f"the checkpoint patch size {args.patch_size}."
        )
    if model.config.dropout_p != 0.0:
        raise ValueError("Toto-2 continued pretraining requires dropout_p=0.")
    frozen_variate_layers = (
        freeze_variate_attention_layers(model) if args.freeze_variate_layers else []
    )

    uu.cache_fan_values(model.named_parameters())
    uu.init_world_size_cache(world_size=world_size)
    uu.set_grad_accumulation_steps(args.gradient_accumulation_steps)
    normuon_params, adamw_params = split_optimizer_parameters(model)
    model.to(device)
    # Keep master parameters and NorMuon state in FP32. BF16 is used only for
    # forward/backward autocast; Dion's normalization updates require FP32.
    if world_size > 1:
        # 所有 rank 都完成权重加载并搬运到本地 NPU 后再进入 DDP 的集合通信，
        # 防止 obsfs 并发加载进度差触发 HCCL 等待超时（EI0006）。
        dist.barrier()
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model

    normuon = uu.NorMuon(
        normuon_params,
        lr=args.normuon_learning_rate,
        mu=args.normuon_mu,
        muon_beta2=args.normuon_beta2,
        weight_decay=args.normuon_weight_decay,
        nesterov=True,
        cautious_wd=True,
        use_polar_express=not NPU_AVAILABLE,
        # CUDA Triton kernels are not available on Ascend. The pure-Torch
        # eager implementation also avoids torch.compile on torch_npu 2.3.
        use_triton=not NPU_AVAILABLE,
        newton_schulz_func=eager_polar_express if NPU_AVAILABLE else None,
    )
    adamw = uu.AdamW(
        adamw_params,
        lr=args.adamw_learning_rate,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=0.0,
    )
    normuon_scheduler = make_scheduler(
        normuon,
        base_lr=args.normuon_learning_rate,
        min_lr=args.normuon_min_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
    )
    adamw_scheduler = make_scheduler(
        adamw,
        base_lr=args.adamw_learning_rate,
        min_lr=args.adamw_min_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
    )
    optimizers = (normuon, adamw)
    schedulers = (normuon_scheduler, adamw_scheduler)
    global_step = load_training_state(args.resume_from_checkpoint, optimizers, schedulers)
    dataloader = build_dataloader(args, global_step)
    iterator = iter(dataloader)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"NorMuon implementation: {dion_source}", flush=True)
        trainable = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in raw_model.parameters())
        print(
            f"Toto-2 continued pretraining: total={total:,} trainable={trainable:,} "
            f"world_size={world_size} local_batch={args.per_device_train_batch_size} "
            f"effective_batch={args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps} "
            f"frozen_variate_layers={frozen_variate_layers}",
            flush=True,
        )

    normuon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    model.train()
    last_log_time = time.time()
    step_limit = min(args.max_steps, args.stop_after_steps)
    while global_step < step_limit:
        accumulated_metrics: list[dict[str, torch.Tensor]] = []
        for micro_step in range(args.gradient_accumulation_steps):
            batch = next(iterator)
            prepared = prepare_toto_batch(
                batch,
                device,
                patch_size=raw_model.config.patch_size,
                cpm_max_span_patches=args.cpm_max_span_patches,
                cpm_max_mask_probability=args.cpm_max_mask_probability,
            )
            sync_gradients = micro_step == args.gradient_accumulation_steps - 1
            sync_context = (
                contextlib.nullcontext()
                if sync_gradients or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    outputs = model(
                        target=prepared["target"],
                        target_mask=prepared["target_mask"],
                        cpm_mask=prepared["cpm_mask"],
                        series_ids=prepared["series_ids"],
                    )
                    loss, metrics = toto_pinball_loss(
                        outputs,
                        prepared,
                        patch_size=raw_model.config.patch_size,
                        world_size=world_size,
                    )
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                accumulated_metrics.append(metrics)

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        normuon.step()
        adamw.step()
        normuon_scheduler.step()
        adamw_scheduler.step()
        normuon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        global_step += 1

        if rank == 0 and global_step % args.logging_steps == 0:
            mean_loss = torch.stack([item["loss"] for item in accumulated_metrics]).mean().item()
            valid_points = sum(
                item["valid_masked_points"].item() for item in accumulated_metrics
            )
            masked_patches = sum(
                item["masked_patches"].item() for item in accumulated_metrics
            )
            now = time.time()
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "loss": mean_loss,
                        "grad_norm": float(grad_norm),
                        "normuon_lr": max(group["lr"] for group in normuon.param_groups),
                        "adamw_lr": max(group["lr"] for group in adamw.param_groups),
                        "valid_masked_points": valid_points,
                        "masked_patches": masked_patches,
                        "seconds_per_step": (now - last_log_time) / args.logging_steps,
                        "peak_memory_allocated_gib": accelerator.max_memory_allocated(device)
                        / (1024**3),
                        "peak_memory_reserved_gib": accelerator.max_memory_reserved(device)
                        / (1024**3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_log_time = now

        if global_step % args.save_steps == 0 or global_step == step_limit:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                save_checkpoint(raw_model, output_dir, global_step, optimizers, schedulers, args)
                print(f"Saved {output_dir / f'checkpoint-{global_step}'}", flush=True)
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
