"""计算 GIFT-Eval 结果 CSV 中所有数据集的平均 MASE 与 CRPS。

支持两种输入，自动识别列结构：
1. extract_mase_crps.py 输出（dataset, MASE, CRPS）→ 算该文件的均值。
2. compare_results.py 输出（dataset, {label}_MASE, {label}_CRPS, ...）→ 算每个 label 的均值。

均值口径：对所有有效数据集做简单算术平均（跳过空值）。

用法：
    python average_results.py --input a.mase_crps.csv
    python average_results.py --input compare.csv
    python average_results.py --input a.csv b.csv --output avg.csv
"""

import argparse
import csv
from pathlib import Path


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(values):
    vals = [x for x in (to_float(v) for v in values) if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def default_label(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".mase_crps"):
        stem = stem[: -len(".mase_crps")]
    return stem


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, type=Path,
                   help="一个或多个结果 CSV（extract 或 compare 格式）")
    p.add_argument("--output", type=Path,
                   help="输出 CSV 路径；默认打印到屏幕")
    args = p.parse_args()

    results = []  # (name, mase_mean, crps_mean, n_datasets)

    for path in args.input:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        mase_cols = [c for c in fieldnames if c.endswith("_MASE")]
        crps_cols = [c for c in fieldnames if c.endswith("_CRPS")]

        if mase_cols and crps_cols:
            # compare 格式：每个 *_MASE 对应一个模型
            for mase_col in mase_cols:
                label = mase_col[: -len("_MASE")]
                crps_col = f"{label}_CRPS"
                if crps_col not in crps_cols:
                    continue
                n = sum(1 for r in rows if to_float(r.get(mase_col)) is not None)
                results.append((
                    label,
                    mean(r[mase_col] for r in rows),
                    mean(r[crps_col] for r in rows),
                    n,
                ))
        elif "MASE" in fieldnames and "CRPS" in fieldnames:
            # extract 格式：单模型
            label = default_label(path)
            results.append((
                label,
                mean(r["MASE"] for r in rows),
                mean(r["CRPS"] for r in rows),
                len(rows),
            ))
        else:
            raise SystemExit(f"{path}: 无法识别的列结构: {fieldnames}")

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "MASE_mean", "CRPS_mean", "n_datasets"])
            writer.writeheader()
            for name, mase, crps, n in results:
                writer.writerow({
                    "name": name,
                    "MASE_mean": fmt(mase),
                    "CRPS_mean": fmt(crps),
                    "n_datasets": n,
                })
        print(f"已写出 {len(results)} 行到 {args.output}")
    else:
        print(f"{'name':<24} {'MASE_mean':>10} {'CRPS_mean':>10} {'n_datasets':>10}")
        for name, mase, crps, n in results:
            print(f"{name:<24} {fmt(mase):>10} {fmt(crps):>10} {n:>10}")


if __name__ == "__main__":
    main()
