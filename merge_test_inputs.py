#!/usr/bin/env python3
"""Merge two independently-generated test_inputs datasets (different seeds) into one.

Produces a combined dataset indistinguishable (for consumption) from a single run:
  output/
    plan.json                 # tasks concatenated, B's global_offset shifted by A.total
    merged/
      block-000000.zarr ...   # A's blocks then B's blocks (renumbered)
      COMPLETE.json           # written_count = A.total + B.total
      samples-merged.jsonl    # combined audit lines, B's global_id shifted by A.total

Usage:
  python merge_test_inputs.py --dir-a /data/.../test_inputs \
      --dir-b /data/.../test_inputs --output /data/.../test_inputs_combined
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_plan(d: Path) -> dict:
    p = d / "plan.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text())


def block_index(name: str) -> int:
    # "block-000003.zarr" -> 3
    return int(name.split(".")[0].split("-")[1])


def check_schema(block_a: Path, block_b: Path) -> None:
    import zarr

    ga = zarr.open_group(str(block_a), mode="r")
    gb = zarr.open_group(str(block_b), mode="r")
    keys_a = sorted(ga.array_keys())
    keys_b = sorted(gb.array_keys())
    if keys_a != keys_b:
        raise ValueError(f"Schema mismatch: A={keys_a} B={keys_b}")
    for k in keys_a:
        if ga[k].dtype != gb[k].dtype or ga[k].shape[1:] != gb[k].shape[1:]:
            raise ValueError(f"Array '{k}' dtype/shape mismatch")


def read_complete(d: Path) -> dict:
    p = d / "merged" / "COMPLETE.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text())


def copy_one(src: Path, dst: Path) -> str:
    shutil.copytree(src, dst)
    return f"{src.name} -> {dst.name}"


def merge_audit(dir_a: Path, dir_b: Path, out_merged: Path, shift: int) -> int:
    """Concatenate A's audit lines + B's audit lines (global_id shifted)."""
    n = 0
    with (out_merged / "samples-merged.jsonl").open("w", encoding="utf-8") as out:
        for d, do_shift in ((dir_a, False), (dir_b, True)):
            for p in sorted((d / "merged").glob("samples-*.jsonl")):
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    if do_shift:
                        rec = json.loads(line)
                        rec["global_id"] = rec.get("global_id", 0) + shift
                        out.write(json.dumps(rec) + "\n")
                    else:
                        out.write(line + "\n")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", type=Path, required=True)
    ap.add_argument("--dir-b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0,
                    help="并行拷贝 block 的线程数（0 = 自动，取 min(CPU, 32)）")
    a = ap.parse_args()

    plan_a = load_plan(a.dir_a)
    plan_b = load_plan(a.dir_b)

    # ---- compatibility checks ----
    for key in ("version", "intended_use", "sequence_length", "patch_size", "crop_lengths"):
        if plan_a.get(key) != plan_b.get(key):
            raise ValueError(f"Plan mismatch on '{key}': {plan_a.get(key)!r} vs {plan_b.get(key)!r}")
    if plan_a.get("labels_accessed") is not False or plan_b.get("labels_accessed") is not False:
        raise ValueError("Both plans must be history-only (labels_accessed=False)")
    if plan_a.get("version") != 3:
        raise ValueError("Only version-3 test plans are supported")
    if plan_a.get("seed") == plan_b.get("seed"):
        print("WARNING: same seed detected. The smaller set is a subset of the larger; "
              "merging will duplicate samples.", file=sys.stderr)

    total_a = int(plan_a["total"])
    total_b = int(plan_b["total"])
    print(f"A: seed={plan_a['seed']} total={total_a} tasks={len(plan_a['tasks'])}")
    print(f"B: seed={plan_b['seed']} total={total_b} tasks={len(plan_b['tasks'])}")

    # ---- merge tasks: shift B's global_offset ----
    tasks = list(plan_a["tasks"])
    for t in plan_b["tasks"]:
        t2 = dict(t)
        t2["global_offset"] = int(t["global_offset"]) + total_a
        tasks.append(t2)

    # ---- merge benchmark summaries ----
    benchmarks = {}
    for fam in set(plan_a["benchmarks"]) | set(plan_b["benchmarks"]):
        x = plan_a["benchmarks"].get(fam)
        y = plan_b["benchmarks"].get(fam)
        if x is None:
            benchmarks[fam] = y
        elif y is None:
            benchmarks[fam] = x
        else:
            benchmarks[fam] = {
                "requested": int(x.get("requested", 0)) + int(y.get("requested", 0)),
                "actual": int(x.get("actual", 0)) + int(y.get("actual", 0)),
                "available_unique_windows": max(int(x.get("available_unique_windows", 0)),
                                                int(y.get("available_unique_windows", 0))),
                "installed_settings": max(int(x.get("installed_settings", 0)),
                                          int(y.get("installed_settings", 0))),
                "settings_with_samples": int(x.get("settings_with_samples", 0)) + int(y.get("settings_with_samples", 0)),
                "repeated_views": int(x.get("repeated_views", 0)) + int(y.get("repeated_views", 0)),
            }

    # ---- merge excluded (dedupe) ----
    seen = set()
    excluded = []
    for rec in list(plan_a.get("excluded", [])) + list(plan_b.get("excluded", [])):
        k = (rec.get("benchmark"), rec.get("dataset"), rec.get("term"), rec.get("reason"))
        if k not in seen:
            seen.add(k)
            excluded.append(rec)

    combined = dict(plan_a)
    combined.update({
        "total": total_a + total_b,
        "tasks": tasks,
        "benchmarks": benchmarks,
        "excluded": excluded,
        "seed": plan_a["seed"],
        "merged_from_seeds": [plan_a["seed"], plan_b["seed"]],
        "merged_from": [str(a.dir_a), str(a.dir_b)],
    })

    # ---- validate COMPLETE.json counts ----
    ca = read_complete(a.dir_a)
    cb = read_complete(a.dir_b)
    if int(ca["written_count"]) != total_a:
        raise ValueError(f"A COMPLETE.json written_count={ca['written_count']} != plan total={total_a}")
    if int(cb["written_count"]) != total_b:
        raise ValueError(f"B COMPLETE.json written_count={cb['written_count']} != plan total={total_b}")

    # ---- blocks ----
    merged_a = a.dir_a / "merged"
    merged_b = a.dir_b / "merged"
    blocks_a = sorted(merged_a.glob("block-*.zarr"), key=lambda p: block_index(p.name))
    blocks_b = sorted(merged_b.glob("block-*.zarr"), key=lambda p: block_index(p.name))
    if not blocks_a or not blocks_b:
        raise ValueError("Missing block-*.zarr in one or both merged dirs")
    idx_a = [block_index(p.name) for p in blocks_a]
    if idx_a != list(range(len(blocks_a))):
        print("WARNING: A's blocks are not 0-indexed; renumbering them to 0-based.", file=sys.stderr)
    # verify schema between a representative block from each side
    check_schema(blocks_a[0], blocks_b[0])
    print(f"Blocks: A={len(blocks_a)} B={len(blocks_b)} -> combined={len(blocks_a) + len(blocks_b)}")

    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Output not empty: {a.output}")
    out_merged = a.output / "merged"
    out_merged.mkdir(parents=True, exist_ok=True)

    # build copy jobs: A's blocks as-is, then B's blocks renumbered after A
    base = len(blocks_a)
    jobs = [(p, out_merged / p.name) for p in blocks_a]
    jobs += [(p, out_merged / f"block-{base + j:06d}.zarr") for j, p in enumerate(blocks_b)]

    workers = a.workers or min(32, os.cpu_count() or 8)
    print(f"Copying {len(jobs)} blocks with {workers} threads")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(copy_one, s, d): (s, d) for s, d in jobs}
        done = 0
        for fut in as_completed(futures):
            s, d = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed copying {s.name} -> {d.name}: {exc}") from exc
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  copied {done}/{len(jobs)} blocks", flush=True)

    # ---- COMPLETE.json ----
    (out_merged / "COMPLETE.json").write_text(json.dumps({
        "written_count": total_a + total_b,
        "source_type": ca.get("source_type", "official_test_input_history"),
        "labels_accessed": False,
        "intended_use": ca.get("intended_use", "official_test_history_views"),
        "source_shards_retained": True,
        "merged_from": [str(a.dir_a), str(a.dir_b)],
    }, indent=2) + "\n", encoding="utf-8")

    # ---- audit ----
    n_audit = merge_audit(a.dir_a, a.dir_b, out_merged, total_a)

    # ---- combined plan ----
    (a.output / "plan.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    print(f"DONE: total={total_a + total_b} blocks={len(blocks_a) + len(blocks_b)} "
          f"audit_lines={n_audit}")
    print(f"Combined plan: {a.output / 'plan.json'}")
    print(f"Combined merged: {out_merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
