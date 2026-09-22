"""Standalone data sanity checker for the Stage1 Toto-2 CPM pretraining loader.

Reads the *same* sources that `load_gifteval_toto2_cpm_pretrain.py` consumes and
reports per-dataset statistics that are relevant to the NaN/Inf debugging of the
`PatchedCausalStdScaler` (the `(data - loc) / scale` step):

  1. pretrain side  : `data_dir/<dataset>/data-*.arrow`  (raw `target` series)
  2. gifteval side  : `gift_eval.data.Dataset(name, term).training_dataset`

The scaler computes `(data - loc) / scale` in float32 with `scale` floored at
`minimum_scale = 1e-6`.  A non-finite result can only come from:
    * raw `inf` in the source (should be impossible after `_sanitize_series`),
    * `data - loc` overflow (needs |data| on the order of float32 max), or
    * `(data - loc) / scale` overflow (needs |data - loc| > 3.4e32 when scale
      sits at the 1e-6 floor).

This script is intentionally dependency-light: it reads Arrow directly with
`pyarrow` (the loader's own fallback path) and only imports GIFT-Eval through the
loader's `import_gifteval_dataset` for the gifteval side.

Usage (from the stage1/ directory so the chronos loader is importable):

    python check_data.py \
        --data_dir /data/GIFTEvalPretrain \
        --gift_eval_path /data/GIFTEval \
        --gift_eval_src /home/chronos-forecasting-main/gift-eval/src

Optional limits keep the scan bounded on the huge full corpus:
    --max_files_per_dataset N     cap Arrow files scanned per pretrain dataset
    --max_series_per_dataset N    cap series scanned per dataset
    --abs-warn FLOAT              warn when |value| exceeds this (default 1e20)
    --abs-error FLOAT             error when |value| exceeds this (default 3.4e32)
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

FLOAT32_MAX = np.finfo(np.float32).max  # ~3.4e38
SCALE_FLOOR = 1e-6
# |data - loc| / 1e-6 overflows float32 when |data - loc| exceeds this.
OVERFLOW_ABS = FLOAT32_MAX * SCALE_FLOOR  # ~3.4e32


def _series_stats(series: np.ndarray) -> dict:
    """Return finite/nan/inf counts and value-range stats for one 1-D series."""
    s = np.asarray(series, dtype=np.float32).reshape(-1)
    finite = np.isfinite(s)
    n = int(s.size)
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(s).sum())
    n_inf = int(np.isinf(s).sum())
    if n_finite:
        vals = s[finite]
        mn, mx = float(vals.min()), float(vals.max())
        abs_max = float(np.abs(vals).max())
    else:
        mn = mx = abs_max = float("nan")
    return {
        "n": n,
        "n_finite": n_finite,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "min": mn,
        "max": mx,
        "abs_max": abs_max,
    }


class _Accumulator:
    """Merge per-series stats into per-dataset aggregates."""

    def __init__(self):
        self.n_series = 0
        self.n_points = 0
        self.n_finite = 0
        self.n_nan = 0
        self.n_inf = 0
        self.global_min = float("inf")
        self.global_max = float("-inf")
        self.global_abs_max = 0.0
        self.abs_warn_series = 0
        self.abs_error_series = 0
        self.empty_series = 0

    def add(self, stats: dict, *, abs_warn: float, abs_error: float) -> None:
        self.n_series += 1
        self.n_points += stats["n"]
        self.n_finite += stats["n_finite"]
        self.n_nan += stats["n_nan"]
        self.n_inf += stats["n_inf"]
        if stats["n_finite"] == 0:
            self.empty_series += 1
        else:
            self.global_min = min(self.global_min, stats["min"])
            self.global_max = max(self.global_max, stats["max"])
            self.global_abs_max = max(self.global_abs_max, stats["abs_max"])
            if stats["abs_max"] > abs_warn:
                self.abs_warn_series += 1
            if stats["abs_max"] > abs_error:
                self.abs_error_series += 1

    def summary(self, *, abs_warn: float, abs_error: float) -> str:
        nan_frac = (self.n_nan / self.n_points) if self.n_points else 0.0
        lines = [
            f"    series={self.n_series} points={self.n_points} "
            f"finite={self.n_finite} nan={self.n_nan} ({nan_frac:.4%}) inf={self.n_inf}",
            f"    min={self.global_min:.6g} max={self.global_max:.6g} "
            f"abs_max={self.global_abs_max:.6g}",
        ]
        if self.empty_series:
            lines.append(f"    ! all-non-finite series={self.empty_series}")
        if self.abs_warn_series:
            lines.append(
                f"    ! series with |value| > {abs_warn:.3g} = {self.abs_warn_series}"
            )
        if self.abs_error_series:
            lines.append(
                f"    !! series with |value| > {abs_error:.3g} "
                f"(would overflow (data-loc)/scale) = {self.abs_error_series}"
            )
        if self.n_inf:
            lines.append(f"    !! raw inf present in {self.n_inf} points")
        return "\n".join(lines)


def _iter_target_channels(target) -> list[np.ndarray]:
    """Normalize an Arrow `target` cell into a list of 1-D channel arrays.

    Mirrors `_sample_from_record`: a 1-D array is one channel, a 2-D array is
    (channel, time).  Anything else yields no channels.
    """
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr[i] for i in range(arr.shape[0])]
    return []


def _read_pretrain_arrow_targets(arrow_path: Path):
    """Yield per-row channel arrays from a prepared-arrow file using pyarrow."""
    import pyarrow as pa

    with pa.memory_map(str(arrow_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    names = set(table.column_names)
    if "target" not in names:
        print(f"      [skip] {arrow_path.name}: no 'target' column (has {sorted(names)})")
        return
    target_col = table["target"]
    for row in target_col.to_pylist():
        yield from _iter_target_channels(row)


def check_pretrain(data_dir: Path, *, abs_warn: float, abs_error: float,
                   max_files_per_dataset: int, max_series_per_dataset: int):
    if not data_dir.is_dir():
        print(f"[pretrain] data_dir not found: {data_dir}")
        return

    acc = defaultdict(_Accumulator)
    for ds_dir in sorted(data_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        files = sorted(
            p for p in ds_dir.iterdir()
            if p.name.startswith("data-") and p.name.endswith(".arrow")
        )
        if not files:
            continue
        if max_files_per_dataset:
            files = files[:max_files_per_dataset]

        ds_acc = acc[ds_dir.name]
        for fp in files:
            try:
                channels = _read_pretrain_arrow_targets(fp)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  !! failed to read {fp}: {exc}")
                continue
            for ch in channels:
                if max_series_per_dataset and ds_acc.n_series >= max_series_per_dataset:
                    break
                ds_acc.add(_series_stats(ch), abs_warn=abs_warn, abs_error=abs_error)

    print(f"\n=== pretrain ({data_dir}) ===")
    print(f"datasets scanned: {len(acc)}")
    for name in sorted(acc):
        print(f"  [{name}]")
        print(acc[name].summary(abs_warn=abs_warn, abs_error=abs_error))


def check_gifteval(gift_eval_path: Path, gift_eval_src: str | None, *,
                   abs_warn: float, abs_error: float, max_series_per_dataset: int):
    del gift_eval_src  # raw HF-on-disk read below does not need the src package
    if not gift_eval_path or not Path(gift_eval_path).exists():
        print(f"[gifteval] gift_eval_path not found: {gift_eval_path}")
        return

    try:
        import datasets
    except ImportError:
        print("[gifteval] 'datasets' not installed; skipping (pretrain check unaffected)")
        return

    # Dataset.__init__ resolves storage via the GIFT_EVAL env var and
    # `datasets.load_from_disk(storage / name)`.  We read the same on-disk
    # dataset directly (the raw `target` column), avoiding the train/test split.
    os.environ.setdefault("GIFT_EVAL", str(gift_eval_path))
    os.environ.setdefault("GIFT_EVAL_PATH", str(gift_eval_path))

    names = sorted(
        p.name for p in gift_eval_path.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not names:
        print(f"[gifteval] no dataset subdirectories under {gift_eval_path}")
        return

    print(f"\n=== gifteval ({gift_eval_path}) ===")
    print(f"datasets found: {len(names)}")
    for name in names:
        try:
            ds = datasets.load_from_disk(str(gift_eval_path / name)).with_format("numpy")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] !! load_from_disk failed: {exc}")
            continue
        acc = _Accumulator()
        try:
            target_col = ds["target"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] !! no 'target' column: {exc}")
            continue
        for target in target_col:
            for ch in _iter_target_channels(target):
                if max_series_per_dataset and acc.n_series >= max_series_per_dataset:
                    break
                acc.add(_series_stats(ch), abs_warn=abs_warn, abs_error=abs_error)
        print(f"  [{name}]")
        print(acc.summary(abs_warn=abs_warn, abs_error=abs_error))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Stage1 Toto-2 CPM pretraining data for NaN/Inf and extreme values."
    )
    parser.add_argument("--data_dir", default="/data/GIFTEvalPretrain")
    parser.add_argument("--gift_eval_path", default="/data/GIFTEval")
    parser.add_argument("--gift_eval_src", default=None)
    parser.add_argument("--max_files_per_dataset", type=int, default=0,
                        help="0 = scan all files")
    parser.add_argument("--max_series_per_dataset", type=int, default=0,
                        help="0 = scan all series")
    parser.add_argument("--abs_warn", type=float, default=1e20)
    parser.add_argument("--abs_error", type=float, default=OVERFLOW_ABS)
    parser.add_argument("--skip_gifteval", action="store_true")
    args = parser.parse_args()

    print(f"thresholds: abs_warn={args.abs_warn:.3g} abs_error={args.abs_error:.3g} "
          f"(overflow floor scale={SCALE_FLOOR}, float32 max={FLOAT32_MAX:.3g})")

    check_pretrain(
        Path(args.data_dir),
        abs_warn=args.abs_warn,
        abs_error=args.abs_error,
        max_files_per_dataset=args.max_files_per_dataset,
        max_series_per_dataset=args.max_series_per_dataset,
    )
    if not args.skip_gifteval:
        check_gifteval(
            Path(args.gift_eval_path),
            args.gift_eval_src,
            abs_warn=args.abs_warn,
            abs_error=args.abs_error,
            max_series_per_dataset=args.max_series_per_dataset,
        )


if __name__ == "__main__":
    main()
