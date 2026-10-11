"""Merge only after every rank exits successfully; retain all original shards."""
import argparse
from pathlib import Path
import json
import os
import shutil
import numpy as np
import zarr
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def log(*args):
    print(*args, flush=True)


def merge_block(block, paths, output_dir):
    """在 worker 进程里合并单个 block 的所有 rank 分片。"""
    log(f"[{block}] open {len(paths)} source stores")
    groups = [zarr.open_group(str(path), mode="r") for path in paths]
    log(f"[{block}] source stores opened, reading keys")
    keys = sorted(groups[0].array_keys())
    if any(sorted(g.array_keys()) != keys for g in groups):
        raise ValueError("Schema mismatch")
    count = sum(g[keys[0]].shape[0] for g in groups)
    log(f"[{block}] keys={keys}, count={count}")
    dest_path = Path(output_dir) / (block + ".zarr")
    log(f"[{block}] creating dest {dest_path.name}")
    dest = zarr.open_group(str(dest_path), mode="w", zarr_format=2)
    dest.attrs.update(dict(groups[0].attrs))
    for key in keys:
        src = groups[0][key]
        dest.create_dataset(key, shape=(count,) + src.shape[1:], chunks=src.chunks,
                            dtype=src.dtype, compressor=src.compressor)
    log(f"[{block}] datasets created, copying rows")
    offset = 0
    for gi, g in enumerate(groups):
        n = g[keys[0]].shape[0]
        log(f"[{block}] copy rank {gi}: {n} rows")
        for left in range(0, n, 64):
            right = min(n, left + 64)
            for key in keys:
                dest[key][offset+left:offset+right] = g[key][left:right]
        offset += n
    dest.attrs["written_count"] = count
    log(f"[{block}] DONE count={count}")
    return block, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--expected", type=int, required=True)
    p.add_argument("--workers", type=int, default=8,
                   help="并行 merge 的进程数（默认 8；0 = 自动按 CPU 核数，上限为 block 数）")
    a = p.parse_args()

    log(f"[merge] shards={a.shards}, output={a.output}, expected={a.expected}, workers={a.workers}")
    markers = sorted(a.shards.glob("DONE-rank-*.json"))
    log(f"[merge] DONE markers: {len(markers)}")
    if not markers:
        raise ValueError("No completed ranks")
    done = [json.loads(p.read_text()) for p in markers]
    world = done[0]["world"]
    if len(done) != world or {d["rank"] for d in done} != set(range(world)):
        raise ValueError("Some ranks are missing; wait for ALL nodes/ranks")
    if any(d["world"] != world or d["total"] != a.expected or d["samples"] != d["expected"] for d in done):
        raise ValueError("Inconsistent completion metadata")
    log(f"[merge] world={world}, DONE metadata OK")

    stores = sorted(a.shards.glob("block-*.rank-*.zarr"))
    log(f"[merge] found {len(stores)} block stores")
    if not stores:
        raise SystemExit("No stores")
    blocks = {}
    total = 0
    for i, path in enumerate(stores):
        if i % 100 == 0:
            log(f"[merge] scanning store {i}/{len(stores)}: {path.name}")
        g = zarr.open_group(str(path), mode="r")
        policy=(g.attrs.get('intended_use'),g.attrs.get('labels_accessed'))
        if total==0:
            expected_policy=policy
        elif policy!=expected_policy:
            raise ValueError('Cannot merge shards with different data split/label policies')
        sizes = {g[k].shape[0] for k in g.array_keys()}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent array lengths: {path}")
        total += next(iter(sizes))
        blocks.setdefault(path.name.split(".rank-")[0], []).append(path)
    log(f"[merge] scanned all stores, total={total}, expected={a.expected}, blocks={len(blocks)}")
    if total != a.expected:
        raise ValueError(f"Found {total}, expected {a.expected}; do not merge incomplete run")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    log(f"[merge] output dir created: {a.output}")

    workers = a.workers
    if workers <= 0:
        workers = min(len(blocks), os.cpu_count() or 1)
    log(f"[merge] using workers={workers}")

    if workers > 1 and len(blocks) > 1:
        log(f"[merge] parallel mode (spawn, {workers} workers)")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(merge_block, block, paths, a.output): block
                       for block, paths in blocks.items()}
            for fut in as_completed(futures):
                block, count = fut.result()
                log(f"[merge] merged {block}: {count}")
    else:
        log(f"[merge] serial mode")
        for block, paths in blocks.items():
            block, count = merge_block(block, paths, a.output)
            log(f"[merge] merged {block}: {count}")

    log(f"[merge] writing COMPLETE.json (total={total})")
    (a.output / "COMPLETE.json").write_text(json.dumps({"written_count": total,
        "source_type": "official_test_input_history" if expected_policy[0]=="official_test_history_views" else "four_source_training_only",
        "labels_accessed": bool(expected_policy[1]), "intended_use": expected_policy[0],
        "source_shards_retained": True}, indent=2))
    for path in a.shards.glob("*.json*"):
        if path.name != "COMPLETE.json":
            shutil.copy2(path, a.output / path.name)
    log(f"[merge] ALL DONE")


if __name__ == "__main__":
    main()
