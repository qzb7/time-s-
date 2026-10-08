"""计算 GIFT-Eval 总分（官方口径）。

官方 GIFT-Eval 总分不是简单平均，而是：
    1. 对每个 dataset，用 Seasonal_Naive 在该 dataset 的值归一化
       normalized_MASE_i = MASE_i / Seasonal_Naive_MASE_i
       normalized_CRPS_i = CRPS_i / Seasonal_Naive_CRPS_i
    2. 对全部 normalized 值取几何平均  (∏x)^(1/n)

参考官方 notebook（gift-eval/notebooks/*.ipynb 的 run_gift_eval）。

用法（官方口径）：
    python average_results.py --input ours.mase_crps.csv \
        --baseline /path/to/gift-eval/results/Seasonal_Naive/all_results.csv

用法（简单算术平均，仅参考）：
    python average_results.py --input ours.mase_crps.csv --method mean

--input 支持两种格式：
    1. extract_mase_crps.py 输出（dataset, MASE, CRPS）
    2. compare_results.py 输出（dataset, {label}_MASE, {label}_CRPS, ...）
"""

import argparse
import csv
import math
from pathlib import Path


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def default_label(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".mase_crps"):
        stem = stem[: -len(".mase_crps")]
    return stem


def geo_mean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def arith_mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def load_model_csv(path: Path) -> dict:
    """加载模型结果，返回 {name: {dataset: (mase, crps)}}。"""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    mase_cols = [c for c in fieldnames if c.endswith("_MASE")]
    crps_cols = [c for c in fieldnames if c.endswith("_CRPS")]

    if mase_cols and crps_cols:
        result = {}
        for mase_col in mase_cols:
            label = mase_col[: -len("_MASE")]
            crps_col = f"{label}_CRPS"
            if crps_col not in crps_cols:
                continue
            result[label] = {
                r["dataset"]: (to_float(r[mase_col]), to_float(r[crps_col]))
                for r in rows
            }
        return result
    elif "MASE" in fieldnames and "CRPS" in fieldnames:
        label = default_label(path)
        return {
            label: {
                r["dataset"]: (to_float(r["MASE"]), to_float(r["CRPS"]))
                for r in rows
            }
        }
    raise SystemExit(f"{path}: 无法识别的列结构: {fieldnames}")


def load_baseline(path: Path) -> dict:
    """加载 Seasonal_Naive 完整 all_results.csv，返回 {dataset: (mase, crps)}。"""
    MASE_COL = "eval_metrics/MASE[0.5]"
    CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if MASE_COL not in fieldnames or CRPS_COL not in fieldnames:
            raise SystemExit(f"baseline 需要 {MASE_COL!r} / {CRPS_COL!r}，实际列: {fieldnames}")
        return {
            r["dataset"]: (to_float(r[MASE_COL]), to_float(r[CRPS_COL]))
            for r in reader
        }


def official_score(model_data: dict, baseline: dict):
    """归一化 + 几何平均，返回 (mase_score, crps_score, n)。"""
    norm_mase, norm_crps = [], []
    for ds, (mase, crps) in model_data.items():
        if ds not in baseline:
            continue
        b_mase, b_crps = baseline[ds]
        if mase is not None and b_mase:
            norm_mase.append(mase / b_mase)
        if crps is not None and b_crps:
            norm_crps.append(crps / b_crps)
    return geo_mean(norm_mase), geo_mean(norm_crps), len(norm_mase)


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, type=Path,
                   help="模型结果 CSV（extract 或 compare 格式）")
    p.add_argument("--baseline", type=Path,
                   help="Seasonal_Naive 完整 all_results.csv；--method official 时必需")
    p.add_argument("--method", choices=["official", "mean"], default="official",
                   help="official=归一化几何平均（官方口径）；mean=简单算术平均")
    p.add_argument("--output", type=Path, help="输出 CSV 路径；默认打印")
    args = p.parse_args()

    baseline = None
    if args.method == "official":
        if args.baseline is None:
            raise SystemExit("--method official 需要 --baseline（Seasonal_Naive 的 all_results.csv）")
        baseline = load_baseline(args.baseline)

    results = []  # (name, mase, crps, n)
    for path in args.input:
        for name, data in load_model_csv(path).items():
            if args.method == "official":
                mase, crps, n = official_score(data, baseline)
            else:
                mase = arith_mean([m for m, _ in data.values()])
                crps = arith_mean([c for _, c in data.values()])
                n = len(data)
            results.append((name, mase, crps, n))

    score_name = "MASE_score" if args.method == "official" else "MASE_mean"
    crps_name = "CRPS_score" if args.method == "official" else "CRPS_mean"

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", score_name, crps_name, "n_datasets"])
            writer.writeheader()
            for name, mase, crps, n in results:
                writer.writerow({"name": name, score_name: fmt(mase),
                                 crps_name: fmt(crps), "n_datasets": n})
        print(f"已写出 {len(results)} 行到 {args.output}")
    else:
        print(f"{'name':<24} {score_name:>12} {crps_name:>12} {'n_datasets':>10}")
        for name, mase, crps, n in results:
            print(f"{name:<24} {fmt(mase):>12} {fmt(crps):>12} {n:>10}")


if __name__ == "__main__":
    main()
