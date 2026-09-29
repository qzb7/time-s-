"""Evaluate a Stage1 Toto-2 (time-only) checkpoint on GIFT-Eval.

Loads a ``checkpoint-*`` directory produced by ``save_pretrained`` in
``pretrain_toto2_short_balanced.py``, runs causal forecasting over the GIFT-Eval
test windows, and reports the standard GIFT-Eval metrics (MSE / MAE / MASE /
MAPE / sMAPE / MSIS / RMSE / NRMSE / ND / weighted-quantile CRPS) per
``(dataset, term)`` into a single CSV.

The dataset list and the ``gift_eval.data.Dataset`` loader are reused from the
training code path (``load_gifteval_pretrain_dataset_dynamic``) so evaluation is
run against exactly the same data layout as training.

Usage (from the ``stage1/`` directory so the chronos loader is importable):

    python evaluate_stage1_checkpoint.py \
        --checkpoint /path/to/checkpoint-27000 \
        --gift_eval_path /data/GIFTEval \
        --output_dir /data/eval_results

Smoke test (only 5 windows per dataset, short term):

    python evaluate_stage1_checkpoint.py \
        --checkpoint /path/to/checkpoint-27000 \
        --gift_eval_path /data/GIFTEval \
        --output_dir /data/eval_results \
        --terms short --max_samples 5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
CHRONOS_DIR = REPO_ROOT / "chronos"
TOTO2_SRC = REPO_ROOT / "toto2"
DD_UNIT_SCALING_SRC = REPO_ROOT / "dd_unit_scaling"

# Matches ``QuantileKnotsOutputHead``'s default knot list in model.py.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_KEYS = [str(q) for q in QUANTILES]

CSV_HEADER = [
    "dataset", "checkpoint", "model", "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]", "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss", "domain", "num_variates", "term",
]


def detect_device() -> str:
    try:
        import torch_npu  # type: ignore
        if torch_npu.npu.is_available():
            return "npu:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class FixedPredictor:
    """Minimal GluonTS Predictor that replays precomputed forecasts in order."""

    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset, **kwargs):
        yield from self.forecasts


def metric_value(result, name):
    v = result[name]
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v[0])


def prepare_input(entry, device, max_context, patch_size):
    """Convert one univariate GIFT-Eval test window into Toto2Model.forecast inputs.

    Returns ``(inputs, ok)``. ``ok=False`` means the window was skipped (e.g. an
    empty / all-NaN series).

    ``Toto2Model.forecast`` requires the context length to be a multiple of
    ``patch_size`` (the official GluonTS ``instance_splitter`` floors
    ``past_length`` to a patch multiple; we instead pad the tail so no history
    is dropped).  Padding positions are masked out via ``target_mask=False``.
    """
    target = np.asarray(entry["target"], dtype=np.float32)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    elif target.ndim == 2 and target.shape[0] == 1:
        pass  # already [1, len]
    else:
        # Should not happen: multi-variate datasets are evaluated univariately.
        return None, False

    if target.shape[-1] < 2:
        return None, False

    if max_context and target.shape[-1] > max_context:
        target = target[:, -max_context:]

    original_valid = np.isfinite(target)
    length = target.shape[-1]
    pad_len = (-length) % patch_size
    if pad_len:
        target = np.concatenate(
            [target, np.zeros((target.shape[0], pad_len), dtype=np.float32)], axis=-1
        )
        valid = np.concatenate(
            [original_valid, np.zeros((target.shape[0], pad_len), dtype=bool)], axis=-1
        )
    else:
        valid = original_valid

    if not valid.any():
        return None, False

    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    inputs = {
        "target": torch.from_numpy(target).to(device),
        "target_mask": torch.from_numpy(valid).to(device),
        "series_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
    }
    return inputs, True


def build_configs(args, short_datasets, med_long_datasets):
    """Resolve the ``(dataset_name, term)`` list to evaluate."""
    if args.datasets:
        specs = []
        for item in args.datasets:
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                name, term = item.split(":", 1)
                specs.append((name.strip(), term.strip()))
            else:
                for term in args.terms:
                    specs.append((item, term))
        return specs

    specs = []
    for term in args.terms:
        pool = short_datasets if term == "short" else med_long_datasets
        for name in pool:
            specs.append((name, term))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stage1 checkpoint directory (contains config.json + weights).")
    parser.add_argument("--output_dir", required=True, help="Directory for the results CSV.")
    parser.add_argument("--gift_eval_path", default=os.environ.get("GIFT_EVAL", "/data/GIFTEval"), help="GIFT-Eval data root.")
    parser.add_argument("--gift_eval_src", default=str(REPO_ROOT / "chronos" / "gift-eval" / "src"), help="GIFT-Eval src (contains gift_eval package).")
    parser.add_argument("--toto2_src", default=str(TOTO2_SRC), help="toto2 package dir.")
    parser.add_argument("--dd_unit_scaling_src", default=str(DD_UNIT_SCALING_SRC), help="dd_unit_scaling package dir.")
    parser.add_argument("--device", default=None, help="Device (auto-detects npu/cuda/cpu if omitted).")
    parser.add_argument("--datasets", nargs="*", default=None, help="Space/comma separated dataset names (name or name:term). Default: derived from --terms.")
    parser.add_argument("--terms", nargs="*", default=["short"], help="Terms to evaluate: short/medium/long (default: short).")
    parser.add_argument("--max_context", type=int, default=8192, help="Truncate context to last N observations (default 8192, match training).")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit windows per dataset (for smoke tests).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size passed to evaluate_model (metrics only).")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for parallel eval.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards for parallel eval.")
    parser.add_argument("--model_name", default=None, help="Label in the CSV (default: checkpoint dir name).")
    args = parser.parse_args()

    # Ensure the chronos loader, toto2 and dd_unit_scaling are importable.
    for p in (CHRONOS_DIR, args.toto2_src, args.dd_unit_scaling_src):
        p = str(Path(p))
        if p not in sys.path:
            sys.path.insert(0, p)

    from load_gifteval_pretrain_dataset_dynamic import (
        GIFT_EVAL_MED_LONG_DATASETS,
        GIFT_EVAL_SHORT_DATASETS,
        import_gifteval_dataset,
    )
    from toto2 import Toto2Model

    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.model.forecast import QuantileForecast
    from gluonts.time_feature import get_seasonality

    Dataset = import_gifteval_dataset(args.gift_eval_path, args.gift_eval_src)

    device = torch.device(args.device or detect_device())
    print(f"device={device} checkpoint={args.checkpoint}", flush=True)

    model = Toto2Model.from_pretrained(args.checkpoint, map_location="cpu", mmap=False)
    model = model.to(device=device, dtype=torch.float32).eval()
    print(f"loaded model; patch_size={model.config.patch_size}", flush=True)

    configs = build_configs(args, GIFT_EVAL_SHORT_DATASETS, GIFT_EVAL_MED_LONG_DATASETS)
    # Parallel-eval sharding: stable ordering, each shard takes every num_shards-th config.
    configs = [c for i, c in enumerate(configs) if i % args.num_shards == args.shard_id]
    print(f"evaluating {len(configs)} configs (shard {args.shard_id}/{args.num_shards})", flush=True)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or Path(args.checkpoint).name
    csv_path = outdir / f"all_results.shard-{args.shard_id}.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    for name, term in configs:
        row = None
        try:
            meta = Dataset(name=name, term=term, to_univariate=False)
            ds = Dataset(name=name, term=term, to_univariate=meta.target_dim != 1)
        except Exception as exc:
            print(f"[skip] {name}/{term}: Dataset load failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        horizon = int(ds.prediction_length)
        pairs = list(zip(ds.test_data.input, ds.test_data.label))
        if args.max_samples is not None:
            pairs = pairs[: args.max_samples]

        eval_inputs, eval_labels, forecasts = [], [], []
        patch_size = model.config.patch_size
        for inp, lab in pairs:
            inputs, ok = prepare_input(inp, device, args.max_context, patch_size)
            if not ok:
                continue
            with torch.no_grad():
                q = model.forecast(
                    inputs,
                    horizon,
                    has_missing_values=True,
                    scaler_fallback_min_obs=8,
                    quantile_real_cap_k=1e4,
                )
            # q: [n_quantiles, 1, 1, horizon] -> [n_quantiles, horizon]
            q = q[:, 0, 0, :].cpu().numpy()
            if not np.isfinite(q).all():
                print(f"[skip-window] {name}/{term}: non-finite forecast for {lab.get('item_id')}", flush=True)
                continue
            forecasts.append(
                QuantileForecast(q, lab["start"], QUANTILE_KEYS, item_id=lab.get("item_id"))
            )
            eval_inputs.append(inp)
            eval_labels.append(lab)

        if not forecasts:
            print(f"[skip] {name}/{term}: no valid windows", flush=True)
            continue

        class TD:
            pass

        td = TD()
        td.input = eval_inputs
        td.label = eval_labels

        result = evaluate_model(
            FixedPredictor(forecasts),
            test_data=td,
            metrics=[
                MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
                MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
                MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES),
            ],
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(ds.freq),
        )

        row = [
            f"{name}/{ds.freq}/{term}", args.checkpoint, model_name,
            metric_value(result, "MSE[mean]"), metric_value(result, "MSE[0.5]"),
            metric_value(result, "MAE[0.5]"), metric_value(result, "MASE[0.5]"),
            metric_value(result, "MAPE[0.5]"), metric_value(result, "sMAPE[0.5]"),
            metric_value(result, "MSIS"), metric_value(result, "RMSE[mean]"),
            metric_value(result, "NRMSE[mean]"), metric_value(result, "ND[0.5]"),
            metric_value(result, "mean_weighted_sum_quantile_loss"),
            "Unknown", meta.target_dim, term,
        ]
        print(f"[ok] {name}/{term}: {len(forecasts)} windows, MASE={row[5]:.4f}, CRPS={row[13]:.4f}", flush=True)
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

    print(f"done. results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
