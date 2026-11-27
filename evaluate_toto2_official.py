#!/usr/bin/env python3
"""Evaluate a Stage1 Toto-2 checkpoint on TIME following the official protocol.

This entry point mirrors ``TIME-main/experiments/toto_model.py`` exactly instead
of the QiYao-M multimodal pipeline:

  * multivariate datasets stay multivariate (``to_univariate=False``); the
    per-variate ``series_ids`` let the model use cross-variate attention,
  * NaN are linearly imputed with ``clean_nan_target`` (not zero-filled),
  * the context is ``--context-length`` (default 4096, official default),
    right-aligned, and the last patch of the context mask is forced valid by
    the model itself,
  * labels are read only by the original ``save_window_predictions`` after all
    predictions for a dataset have been produced.

Data loading uses the vendored ``timebench`` package (identical to TIME-main) so
the test windows and metrics match the leaderboard. Point ``--time-data`` at the
``Real-TSF/TIME`` HuggingFace snapshot that ``datasets.load_from_disk`` reads.

Usage (from this directory):

    python evaluate_toto2_official.py \
        --checkpoint /path/to/checkpoint-27000 \
        --toto2-src /path/to/stage1/toto2 \
        --time-data /path/to/TIME \
        --output-dir /path/to/results

    # single task / smoke test
    python evaluate_toto2_official.py --checkpoint ... --dataset Agriculture --terms short \
        --time-data /path/to/TIME --output-dir /path/to/results --limit-windows 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Stage1 Toto-2 checkpoint dir (config.json + weights)')
    p.add_argument('--toto2-src', type=Path, default=None,
                   help='Path to the stage1/toto2 project dir (auto-resolved if omitted)')
    p.add_argument('--time-data', type=Path, default=None,
                   help='Real-TSF/TIME root; defaults to $TIME_DATASET')
    p.add_argument('--config', type=Path, default=None,
                   help='datasets.yaml; defaults to the vendored timebench config')
    p.add_argument('--output-dir', type=Path, required=True,
                   help='Results directory (must be new or empty)')
    p.add_argument('--dataset', nargs='+', default=None,
                   help="Dataset name(s), e.g. Agriculture 'SG_PM25/H'. Default: all.")
    p.add_argument('--terms', nargs='+', default=None, choices=['short', 'medium', 'long'],
                   help='Horizon terms; default: auto-detect per dataset')
    p.add_argument('--context-length', type=int, default=4096,
                   help='Max context length, right-aligned (official default: 4096)')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Windows per forward (official runs one window at a time)')
    p.add_argument('--device', default='auto', help='auto, npu, cuda, or cpu')
    p.add_argument('--npu-id', type=int, default=None, help='Accelerator index (NPU/CUDA)')
    p.add_argument('--scaler-fallback-min-obs', type=int, default=0,
                   help='Passed to model.forecast (official reference uses the model default)')
    p.add_argument('--quantile-real-cap-k', type=float, default=0.0,
                   help='Passed to model.forecast (0 = disabled)')
    p.add_argument('--limit-windows', type=int, default=None,
                   help='Only the first N windows per task (smoke test, not a score)')
    p.add_argument('--model-name', default=None,
                   help='Label in output config (default: checkpoint dir name)')
    a = p.parse_args()
    if a.batch_size < 1 or a.context_length < 1:
        p.error('--batch-size and --context-length must be positive')
    return a


def resolve_device(requested, npu_id):
    """Resolve --device into a concrete torch device: NPU, then CUDA, then CPU."""
    import torch
    kind, _, idx = requested.partition(':')
    index = int(idx) if idx else npu_id
    if kind in ('auto', 'npu'):
        try:
            import torch_npu  # noqa: F401  (registers the 'npu' backend)
            npu = torch_npu.npu
        except ImportError:
            npu = None
        if npu is not None and npu.is_available():
            target = 0 if index is None else index
            if target >= npu.device_count():
                raise RuntimeError(f'NPU index {target} but only {npu.device_count()} visible')
            npu.set_device(torch.device('npu', target))
            return torch.device('npu', target)
        if kind == 'npu':
            raise RuntimeError('NPU requested but torch_npu unavailable; pass --device cpu/cuda')
    if kind in ('auto', 'cuda'):
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            return torch.device('cuda', 0 if index is None else min(index, count - 1))
        if kind == 'cuda':
            raise RuntimeError('CUDA requested but unavailable; pass --device cpu')
    return torch.device('cpu')


def _resolve_toto2_sys_path(toto2_src_arg, script_root):
    """Return the directory whose subdirectory ``toto2/`` is the importable package."""
    script_root = script_root.resolve()
    candidates = []
    if toto2_src_arg is not None:
        p = Path(toto2_src_arg).expanduser()
        if p.is_absolute():
            candidates += [p.resolve(), p.resolve().parent]
        else:
            candidates += [(script_root / p).resolve(), (script_root / p).resolve().parent,
                           (Path.cwd() / p).resolve(), (Path.cwd() / p).resolve().parent]
    for cand in candidates:
        if (cand / 'toto2' / '__init__.py').is_file():
            return str(cand)
    here = script_root
    for _ in range(8):
        for cand in (here / 'stage1' / 'toto2', here / 'toto2'):
            if (cand / '__init__.py').is_file():
                return str(cand.parent)
        parent = here.parent
        if parent == here:
            break
        here = parent
    shown = '\n    '.join(str(c) for c in candidates) or '(none)'
    raise FileNotFoundError(
        f'Cannot resolve Toto-2 package (script_root={script_root}).\n'
        f'Candidates tried:\n    {shown}\n'
        'Pass --toto2-src as a dir containing toto2/__init__.py (e.g. stage1/toto2/) '
        'or the package dir itself.')


def load_model(a, device):
    import torch
    sys_path = _resolve_toto2_sys_path(a.toto2_src, ROOT)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    print(f'[toto2] sys.path += {sys_path}', flush=True)
    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained(a.checkpoint, map_location='cpu')
    model = model.to(device=device, dtype=torch.float32).eval()
    print(f'LOADED {type(model).__name__} patch={model.config.patch_size} '
          f'd_model={model.config.d_model} device={device}', flush=True)
    if model.config.patch_size <= 0:
        raise ValueError('Invalid patch_size in checkpoint config')
    return model


def _sync(device):
    import torch
    if device.type == 'npu':
        import torch_npu
        torch_npu.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def _freq_to_seconds(freq):
    from pandas.tseries.frequencies import to_offset
    try:
        offset = to_offset(freq)
        if getattr(offset, 'delta', None) is not None:
            return float(offset.delta.total_seconds())
    except Exception:
        pass
    return 0.0


def _build_inputs(target, device, interval_seconds, patch_size):
    """Official layout: (n_var, time) -> (1, n_var, time), padded to patch_size.

    Toto-2 splits the context into patches, so the time axis must be a multiple
    of ``patch_size``. The official Toto-1 notebook relies on the model to pad;
    Stage1's Toto-2 requires the caller to hand over an aligned length. We pad on
    the LEFT with NaN so the most recent observation stays adjacent to the
    forecast region (right-aligned), and mark the padded positions as unobserved.
    """
    import torch
    target = np.asarray(target)
    n_var, length = target.shape
    pad = (-length) % patch_size
    if pad:
        target = np.concatenate(
            [np.full((n_var, pad), np.nan, dtype=target.dtype), target], axis=-1)
    t = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.isfinite(t)
    # The scaler must not see NaN; unobserved slots are excluded via the mask.
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    series_ids = torch.arange(n_var, device=device, dtype=torch.long).unsqueeze(0)
    return dict(target=t, target_mask=mask, series_ids=series_ids)


def build_tasks(a, config):
    from timebench.evaluation.utils import get_available_terms
    names = a.dataset or list(config.get('datasets', {}).keys())
    tasks = []
    for name in names:
        terms = a.terms or get_available_terms(name, config)
        if not terms:
            print(f'[skip] {name}: no terms in config', flush=True)
            continue
        for term in terms:
            tasks.append((name, term))
    if not tasks:
        raise ValueError('No tasks selected')
    return tasks


def main():
    a = arguments()
    import torch
    from gluonts.time_feature import get_seasonality

    from timebench.evaluation.data import (Dataset, get_dataset_settings,
                                           load_dataset_config)
    from timebench.evaluation.saver import save_window_predictions
    from timebench.evaluation.utils import clean_nan_target

    data_root = a.time_data
    if data_root is None:
        env = os.getenv('TIME_DATASET')
        if not env:
            raise SystemExit('Pass --time-data or set TIME_DATASET to the Real-TSF/TIME root')
        data_root = Path(env)
    if not data_root.exists():
        raise FileNotFoundError(f'--time-data not found: {data_root}')

    config = load_dataset_config(a.config)
    if not config.get('datasets'):
        raise ValueError(f'Empty dataset config: {a.config or "vendored default"}')

    device = resolve_device(a.device, a.npu_id)
    print(f'device={device} data={data_root}', flush=True)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = a.model_name or a.checkpoint.name

    model = None
    inference_seconds = 0.0
    total_windows = 0
    for name, term in build_tasks(a, config):
        settings = get_dataset_settings(name, term, config)
        ds = Dataset(name=name, term=term, to_univariate=False,
                     prediction_length=settings.get('prediction_length'),
                     test_length=settings.get('test_length'),
                     val_length=settings.get('val_length'),
                     storage_path=data_root)
        freq = str(ds.freq)
        ds_config = f'{name}/{term}'
        horizon = int(ds.prediction_length)
        interval = _freq_to_seconds(freq)
        print(f'\n=== {ds_config} horizon={horizon} n_var={ds.target_dim} '
              f'windows={ds.windows} freq={freq} ===', flush=True)

        if model is None:
            model = load_model(a, device)

        forecasts = []
        inputs_iter = ds.test_data.input
        for i, item in enumerate(inputs_iter):
            if a.limit_windows is not None and i >= a.limit_windows:
                break
            target = np.asarray(item['target'])
            if target.ndim == 1:
                target = target[np.newaxis, :]
            target = target[..., -a.context_length:]
            target = clean_nan_target(target)
            inputs = _build_inputs(target, device, interval, model.config.patch_size)
            t0 = time.perf_counter()
            with torch.inference_mode():
                q = model.forecast(
                    inputs, horizon,
                    has_missing_values=True,
                    scaler_fallback_min_obs=a.scaler_fallback_min_obs,
                    quantile_real_cap_k=a.quantile_real_cap_k,
                )
            _sync(device)
            inference_seconds += time.perf_counter() - t0
            if q.is_sparse:
                q = q.to_dense()
            if q.ndim == 5:
                q = q.flatten(-2, -1)
            # [n_quantiles, batch, n_var, horizon] -> [n_quantiles, n_var, horizon]
            pred = q.float().cpu().numpy()
            if pred.ndim == 4:
                pred = pred[:, 0]
            if pred.shape != (len(DEFAULT_QUANTILES), ds.target_dim, horizon):
                raise ValueError(f'{ds_config}: forecast shape {pred.shape} != '
                                 f'{(len(DEFAULT_QUANTILES), ds.target_dim, horizon)}')
            if not np.isfinite(pred).all():
                raise ValueError(f'{ds_config}: non-finite forecast at window {i}')
            forecasts.append(pred)
            total_windows += 1
            if (i + 1) % 100 == 0:
                print(f'  {i + 1} windows', flush=True)

        if not forecasts:
            print(f'[skip] {ds_config}: no windows', flush=True)
            continue
        fc = np.stack(forecasts, axis=0).astype(np.float32, copy=False)
        save_window_predictions(
            dataset=ds, fc_quantiles=fc, ds_config=ds_config,
            output_base_dir=str(a.output_dir), seasonality=get_seasonality(freq),
            model_hyperparams=dict(model_name=model_name, checkpoint=str(a.checkpoint),
                                   context_length=a.context_length,
                                   to_univariate=False, multimodal=False,
                                   scaler_fallback_min_obs=a.scaler_fallback_min_obs,
                                   quantile_real_cap_k=a.quantile_real_cap_k),
            quantile_levels=DEFAULT_QUANTILES,
        )
        print(f'SAVED {ds_config} ({len(forecasts)} windows)', flush=True)

    report = dict(
        benchmark='time', protocol='official_toto2', model=model_name,
        checkpoint=str(a.checkpoint.resolve()), device=str(device),
        context_length=a.context_length, to_univariate=False, batch_size=a.batch_size,
        total_windows=total_windows,
        inference_seconds=round(inference_seconds, 3),
        avg_latency_ms_per_window=round(inference_seconds / total_windows * 1000, 2)
        if total_windows else None,
    )
    (a.output_dir / 'inference_speed.json').write_text(json.dumps(report, indent=2))
    print(f'\nDONE -> {a.output_dir}', flush=True)


if __name__ == '__main__':
    main()
