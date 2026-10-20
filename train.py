"""One continuous endogenous run: heads first, then layers 16--23 + heads."""
import argparse
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import random
import sys


def parse_bool(value):
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", "--model-name-or-path")
    p.add_argument("--data-dir", "--data-dirs", "--multimodal-data-dirs",
                   action="append", nargs="+", required=True, dest="data_groups",
                   help="Data root(s); append ':FRAC' to a root to keep only that fraction, e.g. /data/a:0.5")
    p.add_argument("--output-dir")
    p.add_argument("--head-steps", "--stage1-steps", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=200000, help="Cosine schedule endpoint")
    p.add_argument("--stop-after-steps", type=int, help="Stop earlier without changing cosine")
    p.add_argument("--batch-size", "--per-device-train-batch-size", type=int, default=224)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--min-learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--cpm-max-span-patches", type=int, default=4)
    p.add_argument("--cpm-max-mask-probability", type=float, default=0.4)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=260817)
    p.add_argument("--bf16", type=parse_bool, default=False)
    p.add_argument("--tf32", type=parse_bool, default=False)
    p.add_argument("--sequence-length", type=int, default=8192)
    p.add_argument("--resume-dir", "--resume-from-checkpoint")
    p.add_argument("--check-data", action="store_true", help="Inspect Zarr metadata; no training")
    p.add_argument("--device", choices=["auto", "cuda", "npu", "cpu"], default="auto",
                   help="auto picks NPU then CUDA; cpu is for tiny verification models only")
    argv = [v.split("=", 1)[0].replace("_", "-") +
            ("=" + v.split("=", 1)[1] if "=" in v else "")
            if v.startswith("--") else v for v in sys.argv[1:]]
    a = p.parse_args(argv)
    a.data_dirs = []
    for group in a.data_groups:
        for d in group:
            path, frac = d, 1.0
            if ":" in d:
                head, _, tail = d.rpartition(":")
                try:
                    parsed = float(tail)
                except ValueError:
                    parsed = None
                if parsed is not None and 0 < parsed <= 1:
                    path, frac = head, parsed
            a.data_dirs.append((str(Path(path).expanduser().resolve()), frac))
    del a.data_groups
    a.stop_after_steps = a.max_steps if a.stop_after_steps is None else a.stop_after_steps
    for key in ["max_steps", "stop_after_steps", "batch_size", "gradient_accumulation_steps",
                "save_steps", "logging_steps", "sequence_length"]:
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.sequence_length % 32 or a.sequence_length < 64:
        p.error("--sequence-length must be a multiple of 32 and at least 64")
    if not 0 <= a.head_steps <= a.max_steps or a.stop_after_steps > a.max_steps:
        p.error("Require 0 <= head-steps <= max-steps and stop-after-steps <= max-steps")
    if a.num_workers < 0 or not 0 <= a.warmup_steps <= a.max_steps:
        p.error("Invalid num-workers or warmup-steps")
    if not 0 <= a.min_learning_rate <= a.learning_rate or a.learning_rate <= 0:
        p.error("Require 0 <= min-learning-rate <= learning-rate and learning-rate > 0")
    if a.cpm_max_span_patches < 1 or not 0 < a.cpm_max_mask_probability <= 1:
        p.error("Invalid CPM span or mask probability")
    if a.resume_dir:
        a.resume_dir = str(Path(a.resume_dir).expanduser().resolve())
        if a.model_dir and Path(a.model_dir).expanduser().resolve() != Path(a.resume_dir):
            p.error("For full resume, --model-dir must equal --resume-dir (or omit model-dir)")
        a.model_dir = a.resume_dir
    if not a.check_data and (not a.model_dir or not a.output_dir):
        p.error("Training needs --model-dir and --output-dir")
    return a


def cosine_factor(step, max_steps, min_lr, lr, warmup_steps):
    current = step + 1
    if warmup_steps and current <= warmup_steps:
        return current / warmup_steps
    progress = min(1.0, max(0.0, (current - warmup_steps) / max(1, max_steps - warmup_steps)))
    return min_lr / lr + (1 - min_lr / lr) * 0.5 * (1 + math.cos(math.pi * progress))


def resolve_device(local_rank, requested):
    """Pick an accelerator: NPU, then CUDA (under 'auto'), else CPU."""
    import torch
    npu = None
    if requested in ("auto", "npu"):
        try:
            import torch_npu
            npu = torch_npu.npu
        except Exception:
            npu = None
    if npu is not None and npu.is_available():
        count = npu.device_count()
        if count < 1:
            raise RuntimeError("No NPU is visible to this rank")
        index = 0 if count == 1 else local_rank
        if index >= count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {count} NPUs are visible")
        device = torch.device("npu", index)
        npu.set_device(device)
        return device
    if requested == "npu":
        raise RuntimeError("NPU requested but unavailable; install torch_npu or choose another device")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        count = torch.cuda.device_count()
        index = 0 if count == 1 else local_rank
        if index >= count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {count} GPUs are visible")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        return device
    if requested == "cuda":
        raise RuntimeError("CUDA unavailable; activate your training environment first")
    return torch.device("cpu")


def amp_autocast(device, enabled):
    """BF16 autocast context that also works on NPU (torch.autocast lacks 'npu')."""
    import torch
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "npu":
        import torch_npu
        return torch_npu.npu.amp.autocast(dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def main():
    args = parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from data import MultimodalZarrDataset
    from training_ops import modal_loss, pinball_loss, prepare_tail_first_batch, set_stage
    from qiyaom import EndoMultimodalPredictor

    dataset = MultimodalZarrDataset(args.data_dirs, seed=args.seed,
                                    sequence_length=args.sequence_length)
    report = dataset.report()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
                          [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")])
    if rank == 0:
        print(json.dumps({"event": "data", **report}), flush=True)
        if any("test" in str(i["intended_use"]).lower() or
               "test_context" in i["path"] for i in report["stores"]):
            print("WARNING: Test-history adaptation data selected; this is NOT clean zero-shot.",
                  file=sys.stderr, flush=True)
    if args.check_data:
        return
    workers = max(1, args.num_workers)
    if len(dataset.refs) < world * workers:
        raise ValueError("Not enough data chunks for all ranks/workers; reduce workers/GPUs")
    model_path = Path(args.model_dir).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing local model config: {model_path}/config.json")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume_dir:
        raise FileExistsError(f"Output is not empty: {output}; use a new directory or --resume-dir")
    device = resolve_device(local, args.device)
    if device.type == "cpu" and args.bf16:
        raise ValueError("CPU verification uses --bf16 false")
    if world > 1:
        backend = "nccl" if device.type == "cuda" else ("hccl" if device.type == "npu" else "gloo")
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=30))
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "npu":
        import torch_npu
        torch_npu.npu.manual_seed_all(args.seed + rank)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        if args.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Visible GPU does not support bf16; use --bf16 false")
    raw = EndoMultimodalPredictor.from_pretrained(model_path, map_location="cpu").to(device)
    if raw.patch_size != 32 or raw.image_dim != 512 or raw.text_dim != 512:
        raise ValueError("Expected patch32, image512, text512 generator/model format")
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda s: cosine_factor(s, args.max_steps, args.min_learning_rate,
                                 args.learning_rate, args.warmup_steps))
    start = 0
    if args.resume_dir:
        previous = json.loads((model_path / "training_args.json").read_text())
        for key in ["head_steps", "max_steps", "learning_rate", "min_learning_rate",
                    "warmup_steps", "weight_decay", "gradient_accumulation_steps"]:
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"Resume changes {key}; repeat original training settings")
        state = torch.load(model_path / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["global_step"])
        rng = state.get("rng_by_rank", [])
        if len(rng) == world:
            torch.set_rng_state(rng[rank]["torch"])
            if device.type == "npu":
                import torch_npu
                torch_npu.npu.set_rng_state(rng[rank]["npu"], device)
            elif device.type == "cuda":
                torch.cuda.set_rng_state(rng[rank]["cuda"], device)
        if rank == 0:
            print(json.dumps({"event": "resume", "step": start,
                "optimizer_restored": True, "scheduler_restored": True,
                "data_cursor_restored": False}), flush=True)
    if start >= args.stop_after_steps:
        raise ValueError("stop-after-steps must be greater than resumed global_step")
    head_only = start < args.head_steps
    set_stage(raw, stage1=head_only)

    def wrap():
        return DDP(raw, device_ids=[local] if device.type in ("cuda", "npu") else None,
                   find_unused_parameters=False) if world > 1 else raw

    model = wrap()
    model.train()
    iterator = iter(DataLoader(dataset, batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=device.type == "cuda",
                    drop_last=True, persistent_workers=args.num_workers > 0))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        (output / "data_manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    def log_stage(event, step):
        if rank == 0:
            print(json.dumps({"event": event, "step": step,
                "stage": "heads_only" if head_only else "last8_and_heads",
                "parameters": sum(p.numel() for p in raw.parameters()),
                "trainable_parameters": sum(p.numel() for p in raw.parameters() if p.requires_grad),
                "world_size": world,
                "effective_batch": args.batch_size * world * args.gradient_accumulation_steps,
                "loss_reduction": "global_masked_patch_average_for_all_three_losses",
                "optimizer_restarted": False, "scheduler_restarted": False}), flush=True)

    def save(step):
        rng = {"torch": torch.get_rng_state()}
        if device.type == "npu":
            import torch_npu
            rng["npu"] = torch_npu.npu.get_rng_state(device)
        elif device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(device)
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        status = [None]
        if rank == 0:
            try:
                final = output / f"checkpoint-{step}"
                if final.exists():
                    raise FileExistsError(f"Refusing to overwrite {final}")
                temporary = output / f"checkpoint-{step}.incomplete-{os.getpid()}"
                raw.save_pretrained(temporary)
                torch.save({"global_step": step, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "head_only": head_only,
                            "rng_by_rank": all_rng}, temporary / "training_state.pt")
                (temporary / "training_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
                temporary.rename(final)
                print(json.dumps({"event": "checkpoint", "step": step, "path": str(final)}), flush=True)
            except Exception as exc:
                status[0] = str(exc)
        if world > 1:
            dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(status[0])

    log_stage("start", start)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start, args.stop_after_steps):
        if head_only and step >= args.head_steps:
            if world > 1:
                dist.barrier()
                del model
            set_stage(raw, stage1=False)
            model = wrap()  # Rebuild DDP hooks for newly trainable parameters.
            model.train()
            head_only = False
            log_stage("unfreeze_last8", step + 1)
        stats = []
        for micro in range(args.gradient_accumulation_steps):
            batch = next(iterator)
            prepared = prepare_tail_first_batch(batch, device, patch_size=32,
                max_span_patches=args.cpm_max_span_patches,
                max_mask_probability=args.cpm_max_mask_probability)
            image = batch["image_tokens"].to(device, non_blocking=True).unsqueeze(1)
            text = batch["text_tokens"].to(device, non_blocking=True).unsqueeze(1)
            sync = micro + 1 == args.gradient_accumulation_steps
            context = model.no_sync() if world > 1 and not sync else contextlib.nullcontext()
            with context:
                with amp_autocast(device, args.bf16):
                    outputs, hidden = model(**{k: prepared[k] for k in
                        ["target", "target_mask", "cpm_mask", "series_ids"]},
                        image_tokens=image, text_tokens=text)
                    lt, metrics = pinball_loss(outputs, prepared, patch_size=32, world_size=world)
                    li, gi, count = modal_loss(raw.image_decoder(hidden), image,
                                               prepared["masked_patches"], world)
                    lx, gx, _ = modal_loss(raw.text_decoder(hidden), text,
                                           prepared["masked_patches"], world)
                    loss = (lt + li + lx) / args.gradient_accumulation_steps
                loss.backward()
            stats.append(torch.stack([metrics["loss"].detach(), gi.detach(), gx.detach(), count.detach()]))
        grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.max_grad_norm,
                                              error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and (step + 1) % args.logging_steps == 0:
            a, b, c, count = torch.stack(stats).mean(0).tolist()
            print(json.dumps({"step": step + 1, "stage": 1 if head_only else 2,
                "loss": a + b + c, "loss_task": a, "loss_image": b, "loss_text": c,
                "masked_patches": count, "grad_norm": float(grad),
                "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        if (step + 1) % args.save_steps == 0 or step + 1 == args.stop_after_steps:
            save(step + 1)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
