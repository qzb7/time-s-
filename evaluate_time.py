#!/usr/bin/env python3
"""TIME endogenous evaluation, optionally with the joint-training shared head."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import sys

import numpy as np

from time_common import (Cache, DEFAULT_CONFIG, ROOT, dataset, digest, history_batches,
                         rank_info, sample_count, signature, task_metadata, tasks, write_json)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--time-data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--shared-head', type=Path, help='Optional shared_head_state.pt from joint checkpoint-1000 (ignored with --unimodal)')
    p.add_argument('--unimodal', action='store_true', help='Load a pure-numerical (Toto2 architecture) checkpoint; skip image/text entirely')
    p.add_argument('--toto2-src', type=Path, default=None, help='Path to the stage1/toto2 package (auto-resolved if omitted)')
    p.add_argument('--output-dir', type=Path, required=True, help='ONE model directory, not results root')
    p.add_argument('--model-name', default='QiYao-M')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='auto')
    p.add_argument('--scaler-fallback-min-obs', type=int, default=8)
    p.add_argument('--include', action='append', default=[])
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or not 0 <= a.shard_id < a.num_shards or a.scaler_fallback_min_obs < 0:
        p.error('Invalid batch/shard/fallback settings')
    return a


def weights(a):
    if a.unimodal:
        paths = [a.checkpoint / 'config.json']
    else:
        paths = [a.checkpoint / n for n in ['config.json', 'layer16_concat_additive_state.pt']]
    index = a.checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        paths += [index] + [a.checkpoint / n for n in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    else:
        paths += [a.checkpoint / 'model.safetensors']
    if not a.unimodal and a.shared_head:
        paths.append(a.shared_head)
    # Fail closed: the native loader would otherwise initialize random fusion weights.
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {str(p.resolve()): digest(p) for p in paths}


def _resolve_toto2_sys_path(toto2_src_arg, script_root):
    """Return the directory whose subdirectory ``toto2/`` is the package.

    Accepts three user-supplied forms (all resolved to absolute paths):
      1. project root containing ``toto2/__init__.py``
      2. package directory itself (``.../toto2/`` containing ``__init__.py``)
      3. None — auto-search script_root upward for ``stage1/toto2`` or ``toto2``

    Raises FileNotFoundError with a concrete hint if nothing resolves.
    """
    candidates = []
    if toto2_src_arg is not None:
        arg = Path(toto2_src_arg).resolve()
        candidates.append(arg)          # form 1: project root
        candidates.append(arg.parent)   # form 2: package dir itself

    for cand in candidates if candidates else []:
        pkg_init = cand / 'toto2' / '__init__.py'
        if pkg_init.is_file():
            return str(cand), str(pkg_init)

    # Auto-search fallback.
    here = script_root.resolve()
    for _ in range(8):
        for cand in (here / 'stage1' / 'toto2', here / 'toto2'):
            pkg_init = cand / '__init__.py'
            if pkg_init.is_file():
                # cand is the package dir; sys.path needs its parent.
                return str(cand.parent), str(pkg_init)
        here = here.parent
        if not here or here == here.parent:
            break

    msg = [f'Cannot resolve Toto-2 package.']
    if candidates:
        shown = '  '.join(str(c) for c in candidates)
        msg.append(f'--toto2-src={toto2_src_arg!r} resolved to candidates: {shown}')
        msg.append('Expected form A: a directory that CONTAINS toto2/__init__.py (e.g. stage1/toto2/)')
        msg.append('Expected form B: the package dir itself (e.g. stage1/toto2/toto2/) — code will use its parent')
    else:
        msg.append('No --toto2-src provided; auto-search walked upward from script root and found nothing.')
    raise FileNotFoundError('\n'.join(msg))


def load_model(a, device):
    import torch
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    if a.unimodal:
        sys_path, pkg_init = _resolve_toto2_sys_path(a.toto2_src, ROOT)
        print(f'[unimodal] Adding to sys.path: {sys_path}  (found package at {pkg_init})', flush=True)
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from toto2 import Toto2Model
        model = Toto2Model.from_pretrained(a.checkpoint, map_location='cpu')
        if model.config.patch_size != 32:
            raise ValueError(f'Expected patch-size 32, got {model.config.patch_size}')
        n_var = 1  # TIME benchmark is always univariate
        model_type = f'Toto2 (patch={model.config.patch_size}, d_model={model.config.d_model})'
    else:
        from qiyaom.modeling_qiyaom import EndoMultimodalPredictor
        model = EndoMultimodalPredictor.from_pretrained(a.checkpoint, map_location='cpu')
        if a.shared_head:
            head = torch.load(a.shared_head, map_location='cpu', weights_only=True)
            model.base_model.output_head.load_state_dict(head, strict=True)
            print('VERIFIED_SHARED_HEAD', len(head), flush=True)
        if model.patch_size != 32:
            raise ValueError('Expected patch-size 32')
        n_var = 1
        model_type = f'EndoMultimodalPredictor (patch={model.patch_size})'
    # Same parameter-wise FP32 move as the old 207 evaluator.
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.to(device=device, dtype=torch.float32)
            param.grad = None
        for buf in model.buffers():
            buf.data = buf.data.to(device=device, dtype=torch.float32 if buf.is_floating_point() else buf.dtype)
    print(f'LOADED_MODEL {model_type} n_var={n_var} device={device}', flush=True)
    return model.eval()


def _sync(device):
    import torch
    t = torch.device(device)
    if t.type == 'npu':
        import torch_npu
        torch_npu.npu.synchronize()
    elif t.type == 'cuda':
        torch.cuda.synchronize()


def main():
    a = arguments()
    rank, world, device = rank_info(a.device)
    t_start = time.perf_counter()
    inference_seconds = 0.0
    batch_latencies = []
    total_samples = 0
    if world > 1:
        if (a.shard_id, a.num_shards) != (0, 1):
            raise ValueError('Do not combine explicit shards with torchrun')
        a.shard_id, a.num_shards = rank, world
    config, selected = tasks(a.config, a.include)
    cache = Cache(a.cache_dir)
    plans = []
    for name, term in selected:
        ds = dataset(name, term, config, a.time_data)
        task = dict(name=name, term=term, **task_metadata(ds))
        if cache.counts[name, term] != task['samples']:
            raise ValueError(f'{name}/{term}: official={task["samples"]} cache={cache.counts[name, term]}')
        plans.append(task)
    # Assign whole tasks by count, matching the original two-worker scheme.
    bins = [[] for _ in range(a.num_shards)]
    loads = [0] * a.num_shards
    for task in sorted(plans, key=lambda x: x['samples'], reverse=True):
        worker = min(range(a.num_shards), key=lambda i: (loads[i], i))
        bins[worker].append(task); loads[worker] += task['samples']
    print(json.dumps(dict(event='plan', tasks=len(plans), assigned=len(bins[a.shard_id]), loads=loads,
                          shard=a.shard_id, device=device)), flush=True)
    if a.plan_only:
        return
    protocol = dict(checkpoints=weights(a), config_sha256=digest(a.config),
                    cache=str(a.cache_dir.resolve()), cache_receipts=cache.receipts,
                    context_length=8192, patch_size=32, dtype='float32',
                    scaler_fallback_min_obs=a.scaler_fallback_min_obs, batch_size=a.batch_size,
                    model_name=a.model_name, external_attention=False, unimodal=a.unimodal,
                    code_sha256={f: digest(ROOT / f) for f in
                                 ['evaluate_time.py', 'time_common.py', 'qiyaom/modeling_qiyaom.py',
                                  'qiyaom/backbone_qiyaom.py', 'timebench/evaluation/data.py',
                                  'timebench/evaluation/metrics.py', 'timebench/evaluation/saver.py']})
    import torch
    from gluonts.time_feature import get_seasonality
    from patch_encoder.core import lock_file
    from timebench.evaluation.saver import save_window_predictions
    model = None
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(a.output_dir / f'protocol-shard{a.shard_id}.json', protocol)
    for task in sorted(bins[a.shard_id], key=lambda x: (x['name'], x['term'])):
        name, term = task['name'], task['term']
        ds = dataset(name, term, config, a.time_data)
        freq = str(ds.freq)
        ds_config = f'{name}/{term}' if name.rsplit('/', 1)[-1].lower() == freq.lower() else f'{name}/{freq}/{term}'
        destination = a.output_dir / ds_config
        destination.parent.mkdir(parents=True, exist_ok=True)
        sig = signature(dict(protocol=protocol, task=task))
        with lock_file(destination.parent / (term + '.lock')):
            if destination.exists():
                receipt = destination / 'evaluation_complete.json'
                if not receipt.is_file() or json.loads(receipt.read_text()).get('signature') != sig:
                    raise ValueError(f'Existing output has a different/incomplete protocol: {destination}; choose NEW output')
                for filename in ['predictions.npz', 'metrics.npz', 'config.json']:
                    if not (destination / filename).is_file():
                        raise FileNotFoundError(destination / filename)
                print(f'SKIP {ds_config}', flush=True)
                continue
            if model is None:
                model = load_model(a, device)
            predictions = []
            for values, masks, pads, keys in history_batches(ds, name, term, a.batch_size):
                batch = cache.fetch(keys)
                # Refuse a stale/wrong cache even when sample counts and IDs match.
                if not np.array_equal(batch['mask'], masks) or not np.array_equal(batch['target'][masks], values[masks]):
                    raise ValueError(f'{ds_config}: cached history does not match official input')
                target = torch.from_numpy(batch['target']).to(device).unsqueeze(1)
                mask = torch.from_numpy(batch['mask']).to(device).unsqueeze(1)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    if a.unimodal:
                        series_ids = torch.zeros(len(keys), 1, dtype=torch.long, device=device)
                        q = model.forecast(
                            dict(target=target, target_mask=mask, series_ids=series_ids),
                            int(ds.prediction_length),
                            has_missing_values=True,
                            scaler_fallback_min_obs=a.scaler_fallback_min_obs,
                        )
                    else:
                        image = torch.from_numpy(batch['image']).to(device).unsqueeze(1)
                        text = torch.from_numpy(batch['text']).to(device).unsqueeze(1)
                        q = model.forecast_parallel(target, mask, image, text, int(ds.prediction_length),
                                                    has_missing_values=True,
                                                    scaler_fallback_min_obs=a.scaler_fallback_min_obs)
                _sync(device)
                dt = time.perf_counter() - t0
                inference_seconds += dt
                batch_latencies.append(dt)
                total_samples += len(keys)
                if q.is_sparse:
                    q = q.to_dense()
                if q.ndim == 5:
                    q = q.flatten(-2, -1)
                pred = q.permute(1, 2, 0, 3)[:, 0].float().cpu().numpy()
                if pred.shape != (len(keys), 9, int(ds.prediction_length)) or not np.isfinite(pred).all():
                    raise ValueError(f'Invalid forecast shape/values: {pred.shape}')
                predictions.append(pred)
                print(f'{ds_config}: {sum(len(p) for p in predictions)}/{task["samples"]}', flush=True)
            fc = np.concatenate(predictions)
            if len(fc) != task['samples']:
                raise ValueError('Official forecast count mismatch')
            work = Path(tempfile.mkdtemp(prefix='.time-result-work-', dir=a.output_dir))
            # ONLY this original saver reads labels, after prediction has finished.
            save_window_predictions(dataset=ds, fc_quantiles=fc, ds_config=ds_config,
                                    output_base_dir=str(work), seasonality=get_seasonality(ds.freq),
                                    model_hyperparams=dict(model_name=a.model_name, checkpoint=str(a.checkpoint),
                                                          shared_head=str(a.shared_head) if a.shared_head else None,
                                                          dtype='float32', multimodal=not a.unimodal, external_attention=False),
                                    quantile_levels=[i / 10 for i in range(1, 10)])
            write_json(work / ds_config / 'evaluation_complete.json', dict(signature=sig, task=task))
            os.rename(work / ds_config, destination)
            # Empty temporary parent directories are harmless and can be inspected.
    write_json(a.output_dir / f'shard{a.shard_id}_complete.json', dict(tasks=bins[a.shard_id], protocol=protocol))
    steady = batch_latencies[1:] if len(batch_latencies) > 1 else []
    report = dict(
        benchmark='time', model=a.model_name, device=device, batch_size=a.batch_size,
        shard_id=a.shard_id, num_shards=a.num_shards,
        total_batches=len(batch_latencies), total_samples=total_samples,
        inference_seconds=round(inference_seconds, 3),
        wall_seconds=round(time.perf_counter() - t_start, 3),
        avg_latency_ms_per_batch=round(inference_seconds / len(batch_latencies) * 1000, 2) if batch_latencies else None,
        avg_latency_ms_per_batch_excl_warmup=round(sum(steady) / len(steady) * 1000, 2) if steady else None,
        avg_latency_ms_per_sample=round(inference_seconds / total_samples * 1000, 4) if total_samples else None,
        throughput_samples_per_sec=round(total_samples / inference_seconds, 2) if inference_seconds else None,
        first_batch_ms=round(batch_latencies[0] * 1000, 2) if batch_latencies else None,
        note=('inference = model.forecast_parallel forward only (image/text from cache)'
              if not a.unimodal else
              'inference = model.forecast forward only (pure numerical, no image/text); '
              'first batch includes NPU/CUDA graph compile'),
    )
    name = 'inference_speed.json' if a.num_shards == 1 else f'inference_speed.shard-{a.shard_id}.json'
    write_json(a.output_dir / name, report)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
