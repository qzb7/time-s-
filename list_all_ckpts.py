"""一键列出所有 checkpoint 的最终 MASE / CRPS。

扫描 test20000 ~ test30000（每 500 一个）结果目录，读取每个目录里的
all_results.csv（缺省时回退 partial_results.csv / preview_results.csv），
提取每个 dataset 的 MASE / CRPS 并汇总成一个整体分数。

用法（简单算术平均，默认）：
    python list_all_ckpts.py --root /path/to/results

用法（官方口径：Seasonal_Naive 归一化后几何平均）：
    python list_all_ckpts.py --root /path/to/results --method official \
        --baseline /path/to/gift-eval/results/Seasonal_Naive/all_results.csv

复用本目录已有代码：
    - extract_mase_crps.py 的列名常量 MASE_COL / CRPS_COL
    - average_results.py 的 official_score / arith_mean / load_baseline / fmt
"""

import argparse
import csv
from pathlib import Path

from extract_mase_crps import MASE_COL, CRPS_COL
from average_results import official_score, arith_mean, load_baseline, to_float, fmt


def find_result_csv(folder: Path):
    """返回目录里的评测结果 CSV，优先 all_results.csv。"""
    for name in ("all_results.csv", "partial_results.csv", "preview_results.csv"):
        p = folder / name
        if p.is_file():
            return p
    return None


def load_dataset_metrics(csv_path: Path) -> dict:
    """读取单个 checkpoint 的 all_results.csv，返回 {dataset: (mase, crps)}。"""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if MASE_COL not in fieldnames or CRPS_COL not in fieldnames:
            raise SystemExit(f"{csv_path}: 找不到列 {MASE_COL!r} / {CRPS_COL!r}，实际列: {fieldnames}")
        return {r["dataset"]: (to_float(r[MASE_COL]), to_float(r[CRPS_COL])) for r in reader}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."),
                   help="所有 testXXXXX 目录所在的父目录，默认当前目录")
    p.add_argument("--prefix", default="test", help="目录名前缀，默认 test")
    p.add_argument("--start", type=int, default=20000)
    p.add_argument("--stop", type=int, default=30000)
    p.add_argument("--step", type=int, default=500)
    p.add_argument("--method", choices=["mean", "official"], default="mean",
                   help="mean=简单算术平均；official=Seasonal_Naive 归一化几何平均")
    p.add_argument("--baseline", type=Path,
                   help="Seasonal_Naive 的 all_results.csv；--method official 时必需")
    p.add_argument("--output", type=Path, help="额外写出 CSV；默认只打印")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    baseline = None
    if args.method == "official":
        if args.baseline is None:
            raise SystemExit("--method official 需要 --baseline（Seasonal_Naive 的 all_results.csv）")
        baseline = load_baseline(Path(args.baseline))

    rows = []
    missing = []
    for step in range(args.start, args.stop + 1, args.step):
        name = f"{args.prefix}{step}"
        folder = root / name
        if not folder.is_dir():
            missing.append(name)
            continue
        csv_path = find_result_csv(folder)
        if csv_path is None:
            missing.append(f"{name} (无结果 CSV)")
            continue
        data = load_dataset_metrics(csv_path)
        if args.method == "official":
            mase, crps, n = official_score(data, baseline)
        else:
            mase = arith_mean([m for m, _ in data.values()])
            crps = arith_mean([c for _, c in data.values()])
            n = len(data)
        rows.append((name, mase, crps, n))

    score_name = "MASE_score" if args.method == "official" else "MASE_mean"
    crps_name = "CRPS_score" if args.method == "official" else "CRPS_mean"

    print(f"root={root}  method={args.method}  共 {len(rows)} 个 checkpoint")
    print(f"{'checkpoint':<14} {score_name:>12} {crps_name:>12} {'n_datasets':>10}")
    for name, mase, crps, n in rows:
        print(f"{name:<14} {fmt(mase):>12} {fmt(crps):>12} {n:>10}")

    if missing:
        print("\n缺失/跳过:")
        for m in missing:
            print(f"  - {m}")

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", score_name, crps_name, "n_datasets"])
            writer.writeheader()
            for name, mase, crps, n in rows:
                writer.writerow({"checkpoint": name, score_name: fmt(mase),
                                 crps_name: fmt(crps), "n_datasets": n})
        print(f"\n已写出 {len(rows)} 行到 {args.output}")


if __name__ == "__main__":
    main()
