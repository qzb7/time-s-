"""对照评测：用官方 Toto-2.0 原始模型跑 GIFT-Eval。

目的：判断自训练 Stage1 checkpoint 在 GIFT-Eval 上的低分（例如 m4_yearly 的 MASE 偏高）
到底是评测脚本 bug，还是模型本身的问题。用**同一套评测协议**跑官方模型即可一锤定音：
如果官方模型分数正常（MASE ~个位数甚至 <1），说明脚本没问题、是我们 checkpoint 的问题；
如果官方模型 MASE 也很高，说明评测协议本身有偏差。

官方 GIFT-Eval 协议（来自 gift-eval/notebooks/toto_2_0.ipynb）：
  - context_length = 4096（不是 Stage1 训练用的 8192）
  - 单次 forward（decode_block_size=None，即 0）
  - scaler_fallback_min_obs=8、quantile_real_cap_k=1e4（Toto2GluonTSModelConfig 默认值）

用法（从 stage1/ 目录运行）：

    python evaluate_toto2_reference.py \
        --checkpoint /path/to/Toto-2.0-313m \
        --gift_eval_path /data/GIFTEval \
        --output_dir /data/eval_results \
        --datasets m4_yearly:short

``--checkpoint`` 可以是本地目录（必须含 config.json + model.safetensors）或 HuggingFace
repo id（如 ``Datadog/Toto-2.0-313m``，需要联网）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
CHRONOS_DIR = REPO_ROOT / "chronos"
TOTO2_SRC = REPO_ROOT / "toto2"
DD_UNIT_SCALING_SRC = REPO_ROOT / "dd_unit_scaling"

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
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def predict(self, dataset, **kwargs):
        yield from self.forecasts


def metric_value(result, name):
    v = result[name]
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v[0])


def prepare_batch(windows, device, max_context, patch_size):
    """把多个单变量 GIFT-Eval 窗口拼成一个 batch 喂给 ``forecast``。

    训练/官方评测都是固定长度、左补齐（数据右对齐，padding/missing 用 target_mask=0 标记）。
    这里复现同样的布局：长历史截断到最后 max_context 个点，短历史左补齐到 max_context。
    """
    rows = []  # (target_1d, valid_1d, input_entry, label_entry)
    for inp, lab in windows:
        t = np.asarray(inp["target"], dtype=np.float32)
        if t.ndim == 1:
            pass
        elif t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        else:
            continue
        if t.shape[0] < 2:
            continue
        if max_context and t.shape[0] > max_context:
            t = t[-max_context:]
        valid = np.isfinite(t)
        if not valid.any():
            continue
        rows.append((t, valid, inp, lab))

    if not rows:
        return None, []

    ctx_len = max_context or max(r[0].shape[0] for r in rows)
    if ctx_len % patch_size:
        ctx_len = ((ctx_len + patch_size - 1) // patch_size) * patch_size

    batch = len(rows)
    target = np.zeros((batch, 1, ctx_len), dtype=np.float32)
    mask = np.zeros((batch, 1, ctx_len), dtype=bool)
    for b, (t, valid, _, _) in enumerate(rows):
        n = t.shape[0]
        start = ctx_len - n
        target[b, 0, start:] = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        mask[b, 0, start:] = valid

    inputs = {
        "target": torch.from_numpy(target).to(device),
        "target_mask": torch.from_numpy(mask).to(device),
        "series_ids": torch.zeros((batch, 1), dtype=torch.long, device=device),
    }
    metas = [(r[2], r[3]) for r in rows]
    return inputs, metas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Official model dir (config.json + safetensors) or HF repo id.")
    parser.add_argument("--output_dir", required=True, help="Directory for the results CSV.")
    parser.add_argument("--gift_eval_path", default=os.environ.get("GIFT_EVAL", "/data/GIFTEval"), help="GIFT-Eval data root.")
    parser.add_argument("--gift_eval_src", default=str(REPO_ROOT / "chronos" / "gift-eval" / "src"), help="GIFT-Eval src.")
    parser.add_argument("--toto2_src", default=str(TOTO2_SRC), help="toto2 package dir.")
    parser.add_argument("--dd_unit_scaling_src", default=str(DD_UNIT_SCALING_SRC), help="dd_unit_scaling package dir.")
    parser.add_argument("--device", default=None, help="Device (auto-detects npu/cuda/cpu if omitted).")
    parser.add_argument("--datasets", nargs="*", default=None, help="dataset names (name or name:term). Default: m4_yearly:short.")
    parser.add_argument("--terms", nargs="*", default=["short"], help="Terms to evaluate (default: short).")
    parser.add_argument("--max_context", type=int, default=4096, help="Fixed context length, left-padded. Official GIFT-Eval uses 4096.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit windows per dataset (smoke test).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluate_model (metrics only).")
    parser.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for batched forecast inference.")
    parser.add_argument("--model_name", default=None, help="Label in CSV (default: checkpoint dir name).")
    args = parser.parse_args()

    for p in (CHRONOS_DIR, args.toto2_src, args.dd_unit_scaling_src):
        p = str(Path(p))
        if p not in sys.path:
            sys.path.insert(0, p)

    from load_gifteval_pretrain_dataset_dynamic import import_gifteval_dataset
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
    cfg = model.config
    print(
        f"loaded model: patch_size={cfg.patch_size} d_model={cfg.d_model} "
        f"num_layers={cfg.num_layers} num_heads={cfg.num_heads} "
        f"residual_attn_ratio={cfg.residual_attn_ratio}",
        flush=True,
    )

    # 解析 (dataset, term) 列表
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
    else:
        specs = [("m4_yearly", "short")]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or Path(args.checkpoint).name
    csv_path = outdir / "reference_results.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    for name, term in specs:
        try:
            meta = Dataset(name=name, term=term, to_univariate=False)
            ds = Dataset(name=name, term=term, to_univariate=meta.target_dim != 1)
        except Exception as exc:
            print(f"[skip] {name}/{term}: Dataset load failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        horizon = int(ds.prediction_length)
        print(f"[loading] {name}/{term}: generating test windows (horizon={horizon}) ...", flush=True)
        test_data = ds.test_data
        pairs = list(zip(test_data.input, test_data.label))
        if args.max_samples is not None:
            pairs = pairs[: args.max_samples]
        print(f"[loaded] {name}/{term}: {len(pairs)} windows", flush=True)

        eval_inputs, eval_labels, forecasts = [], [], []
        neg_median_windows = 0
        worst_windows = []  # (median_mae, item_id, ctx_tail, med, lab)
        patch_size = model.config.patch_size
        infer_bs = max(1, args.infer_batch_size)
        for start in range(0, len(pairs), infer_bs):
            chunk = pairs[start:start + infer_bs]
            inputs, metas = prepare_batch(chunk, device, args.max_context, patch_size)
            if inputs is None:
                continue
            with torch.no_grad():
                q = model.forecast(
                    inputs,
                    horizon,
                    has_missing_values=True,
                    scaler_fallback_min_obs=8,
                    quantile_real_cap_k=1e4,
                )
            q = q.cpu().numpy()
            for b, (inp, lab) in enumerate(metas):
                qb = q[:, b, 0, :]
                if not np.isfinite(qb).all():
                    print(f"[skip-window] {name}/{term}: non-finite forecast for {lab.get('item_id')}", flush=True)
                    continue
                lab_t = np.asarray(lab["target"], dtype=np.float32).reshape(-1)
                med = qb[4]
                if (med < 0).any():
                    neg_median_windows += 1
                lab_valid = np.isfinite(lab_t)
                if lab_valid.any():
                    med_mae = float(np.mean(np.abs(med[lab_valid] - lab_t[lab_valid])))
                    ctx_tail = inputs["target"][b, 0, -6:].cpu().numpy()
                    worst_windows.append((med_mae, lab.get("item_id"), np.round(ctx_tail, 1).tolist(), np.round(med, 1).tolist(), np.round(lab_t, 1).tolist()))
                    worst_windows.sort(key=lambda x: x[0], reverse=True)
                    del worst_windows[3:]
                if not eval_labels:
                    ctx = inputs["target"][b, 0].cpu().numpy()
                    ctxm = inputs["target_mask"][b, 0].cpu().numpy()
                    print(f"[first-window] {name}/{term} item={lab.get('item_id')}", flush=True)
                    print(f"  ctx_len={ctx.shape[0]} n_valid={int(ctxm.sum())} tail={np.round(ctx[-6:], 1).tolist()}", flush=True)
                    print(f"  median_pred = {np.round(qb[4], 3).tolist()}", flush=True)
                    print(f"  label       = {np.round(lab_t, 3).tolist()}", flush=True)
                forecasts.append(QuantileForecast(qb, lab["start"], QUANTILE_KEYS, item_id=lab.get("item_id")))
                eval_inputs.append(inp)
                eval_labels.append(lab)

        print(f"[worst] {name}/{term}: neg_median={neg_median_windows}/{len(forecasts)}", flush=True)
        for med_mae, item_id, ctx_tail, med, lab in worst_windows:
            print(f"  mae={med_mae:.1f} item={item_id} ctx_tail={ctx_tail} med={med} lab={lab}", flush=True)

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
