"""对比多个 extract_mase_crps.py 输出，横向合并为一个对比 CSV。

每个输入文件是 extract_mase_crps.py 生成的 CSV，包含 dataset / MASE / CRPS 三列。
输出按 dataset 对齐，每个输入文件占两列：{label}_MASE 和 {label}_CRPS。

用法：
    python compare_results.py --input official.csv ours.csv --label official ours
    python compare_results.py --input a.csv b.csv c.csv --output compare.csv
"""

import argparse
import csv
from pathlib import Path


def default_label(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".mase_crps"):
        stem = stem[: -len(".mase_crps")]
    return stem


def load(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for required in ("dataset", "MASE", "CRPS"):
            if required not in fieldnames:
                raise SystemExit(f"{path}: 找不到列 {required!r}，实际列: {fieldnames}")
        return {r["dataset"]: (r["MASE"], r["CRPS"]) for r in reader}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, type=Path,
                   help="多个 extract_mase_crps.py 输出的 CSV")
    p.add_argument("--label", nargs="*",
                   help="每个输入文件的标签，数量和 --input 对应；默认取文件名")
    p.add_argument("--output", type=Path,
                   help="输出 CSV 路径；默认 compare_results.csv")
    args = p.parse_args()

    inputs = args.input
    labels = args.label or [default_label(x) for x in inputs]
    if len(labels) != len(inputs):
        raise SystemExit(f"--label 数量({len(labels)}) 与 --input 数量({len(inputs)}) 不一致")

    datas = [load(x) for x in inputs]

    # 保持第一个文件的 dataset 顺序，再追加其他文件独有项
    order = []
    for d in datas:
        for ds in d:
            if ds not in order:
                order.append(ds)

    fieldnames = ["dataset"]
    for label in labels:
        fieldnames += [f"{label}_MASE", f"{label}_CRPS"]

    out = args.output or Path("compare_results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ds in order:
            row = {"dataset": ds}
            for label, d in zip(labels, datas):
                mase, crps = d.get(ds, ("", ""))
                row[f"{label}_MASE"] = mase
                row[f"{label}_CRPS"] = crps
            writer.writerow(row)

    print(f"已写出 {len(order)} 个 dataset × {len(inputs)} 个模型 到 {out}")


if __name__ == "__main__":
    main()
