"""Read completed patch-level generator Zarr stores without re-encoding CLIP."""
import os
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import IterableDataset, get_worker_info


class MultimodalZarrDataset(IterableDataset):
    KEYS = ("time_series_8192", "target_mask_8192", "image_global_fp16",
            "image_scale", "text_global_fp16", "text_scale")

    def __init__(self, data_dirs, seed=260817, sequence_length=8192, patch_size=32,
                 modal_dim=512):
        super().__init__()
        self.seed = seed
        self.length = sequence_length
        self.patches = sequence_length // patch_size
        self.modal_dim = modal_dim
        self.stores, self.frac = self.discover(data_dirs)
        self.info = [self.inspect(p) for p in self.stores]
        self.refs = []
        for p, info in zip(self.stores, self.info):
            limit = int(info["count"] * self.frac.get(p, 1.0))
            if limit <= 0:
                continue
            for left in range(0, limit, info["chunk"]):
                self.refs.append((p, left, min(left + info["chunk"], limit)))

    @staticmethod
    def discover(roots):
        # roots is a list of (path, frac); frac in (0, 1] keeps that fraction of rows.
        candidates = []
        for value, frac in roots:
            root = Path(value).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {root}")
            if (root / ".zgroup").exists():
                candidates.append((root, frac))
                continue
            # Prefer merged output to avoid counting both merged and rank copies.
            # Deliberately do NOT recursively include test_inputs/.
            bases = [root / "merged"] if (root / "merged").is_dir() else [root]
            if (root / "generation").is_dir():
                bases.append(root / "generation")
            found = []
            for base in bases:
                found.extend(sorted(base.glob("block-*.zarr")))
                found.extend(sorted(base.glob("block-*/block-*.rank-*.zarr")))
            if not found and (root / "shards").is_dir():
                found.extend(sorted((root / "shards").glob("block-*.rank-*.zarr")))
            if not found:
                raise FileNotFoundError(f"No completed block-*.zarr stores under {root}")
            candidates.extend((p, frac) for p in found)
        dedup = {}
        for p, frac in candidates:
            p = p.resolve()
            if p.is_dir() and not p.name.endswith(".tmp"):
                dedup.setdefault(p, frac)
        result = sorted(dedup.items())  # [(Path, frac), ...]
        if not result:
            raise ValueError("No readable Zarr input was selected")
        # Also protect legacy roots containing both formats for the same block.
        merged = {(p.parent, p.name[:-5]) for p, _ in result if ".rank-" not in p.name}
        for p, _ in result:
            if ".rank-" in p.name and (p.parent, p.name.split(".rank-")[0]) in merged:
                raise ValueError(f"Merged and rank copies overlap: {p}; select only one")
        return [p for p, _ in result], {p: f for p, f in result}

    def inspect(self, path):
        g = zarr.open_group(str(path), mode="r")
        missing = [k for k in self.KEYS if k not in g]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        capacity = g["time_series_8192"].shape[0]
        count = int(g.attrs.get("written_count", capacity))
        if not 0 < count <= capacity:
            raise ValueError(f"{path}: invalid written_count={count}, capacity={capacity}")
        for key in self.KEYS:
            a = g[key]
            if a.shape[0] < count:
                raise ValueError(f"{path}/{key}: shorter than written_count")
            expected = ((self.length,) if key in self.KEYS[:2] else
                        (self.patches, self.modal_dim) if "fp16" in key else
                        (self.patches, 1))
            if a.shape[1:] != expected:
                raise ValueError(f"{path}/{key}: shape={a.shape}; expected (*,{expected})")
        return {"path": str(path), "count": count,
                "chunk": int(g["time_series_8192"].chunks[0]),
                "intended_use": g.attrs.get("intended_use", "legacy_unspecified"),
                "labels_accessed": g.attrs.get("labels_accessed", "unspecified")}

    def report(self):
        return {"rows": sum(right - left for _, left, right in self.refs),
                "chunks": len(self.refs),
                "sequence_length": self.length, "patches": self.patches,
                "modal_dim": self.modal_dim, "stores": self.info}

    def read_chunk(self, path, left, right, order):
        g = zarr.open_group(str(path), mode="r")
        target = np.asarray(g["time_series_8192"][left:right], dtype=np.float32)
        mask = np.asarray(g["target_mask_8192"][left:right], dtype=np.bool_)
        image = (np.asarray(g["image_global_fp16"][left:right], dtype=np.float32)
                 * np.asarray(g["image_scale"][left:right], dtype=np.float32))
        text = (np.asarray(g["text_global_fp16"][left:right], dtype=np.float32)
                * np.asarray(g["text_scale"][left:right], dtype=np.float32))
        if not np.isfinite(target[mask]).all():
            raise ValueError(f"Non-finite observed values: {path}, rows {left}:{right}")
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError(f"Non-finite modality embeddings: {path}, rows {left}:{right}")
        for i in order:
            yield {"target": torch.from_numpy(np.nan_to_num(target[i])),
                   "target_mask": torch.from_numpy(mask[i]),
                   "image_tokens": torch.from_numpy(image[i]),
                   "text_tokens": torch.from_numpy(text[i])}

    def __iter__(self):
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        shard, shards = rank * workers + worker_id, world * workers
        if len(self.refs) < shards:
            raise ValueError(f"Only {len(self.refs)} chunks for {shards} rank/workers; "
                             "reduce --gpus/--num-workers or provide more data")
        cycle = 0
        while True:
            refs = list(self.refs)
            np.random.default_rng(self.seed + cycle).shuffle(refs)
            for ref_index, (path, left, right) in enumerate(refs[shard::shards]):
                order = np.arange(right - left)
                seed = self.seed + cycle * 1_000_003 + shard * 10_007 + ref_index
                np.random.default_rng(seed).shuffle(order)
                yield from self.read_chunk(path, left, right, order)
            cycle += 1
