"""从 QiYao-M GIFT-Eval 评测结果 CSV 中提取 MASE 与 CRPS。

输入是 evaluate_unimodal.py 生成的 all_results.csv / preview_results.csv /
partial_results.csv。输出只保留三列：dataset、MASE、CRPS。

用法：
    python extract_mase_crps.py --input all_results.csv
    python extract_mase_crps.py --input all_results.csv --output mase_crps.csv
"""

import argparse
import csv
from pathlib import Path

MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="all_results.csv 等评测结果文件")
    p.add_argument("--output", type=Path,
                   help="输出 CSV 路径；默认 <input> 同目录下 <stem>.mase_crps.csv")
    args = p.parse_args()

    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"输入文件不存在: {inp}")
    out = args.output or inp.with_name(inp.stem + ".mase_crps.csv")

    with inp.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if MASE_COL not in fieldnames or CRPS_COL not in fieldnames:
            raise SystemExit(f"找不到列 {MASE_COL!r} / {CRPS_COL!r}，实际列: {fieldnames}")

        rows = [
            {"dataset": r["dataset"], "MASE": r[MASE_COL], "CRPS": r[CRPS_COL]}
            for r in reader
        ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "MASE", "CRPS"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已写出 {len(rows)} 行到 {out}")


if __name__ == "__main__":
    main()
