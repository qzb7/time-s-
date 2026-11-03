#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa


def check_resume(registry, args):
    """Check scientific inputs before reusing existing generated cases."""
    saved = json.loads(registry.read_text())
    sources = saved.get('sources', [])
    if not sources or saved.get('seed') != args.seed:
        raise ValueError('Existing registry has different/missing seed; choose a new output')
    roots = {'pretrain': Path(args.pretrain_root).resolve(), 'gifteval': Path(args.gifteval_root).resolve()}
    for source in sources:
        expected = roots[source['corpus']] / source['dataset_relpath']
        if source['quota'] != args.cases_per_source or Path(source['dataset_path']).resolve() != expected.resolve():
            raise ValueError('Existing registry has different roots/quota; choose a new output')
    previous = saved.get('model_inputs', {})
    # The previous optional Python launcher recorded these fields separately.
    config = registry.parent/'generation_config.json'
    if config.is_file() and not previous:
        previous = json.loads(config.read_text())
    for key in ['ts_checkpoint','clip_checkpoint']:
        requested = getattr(args,key)
        if requested and (not previous.get(key) or Path(previous[key]).resolve() != Path(requested).resolve()):
            raise ValueError('Existing output lacks matching model paths; use the original stage commands or a new output')
    print(json.dumps({'status':'registry_verified','sources':len(sources),'registry':str(registry)}),flush=True)


def norm_freq(value: str) -> str:
    s = str(value).strip().upper().replace("MIN", "T")
    aliases = {"1H": "H", "1D": "D", "1W": "W", "1M": "M", "HOURLY": "H",
               "DAILY": "D", "WEEKLY": "W", "MONTHLY": "M", "15MIN": "15T"}
    s = aliases.get(s, s)
    s = re.sub(r"^1([HDTMWQAY])$", r"\1", s)
    return s


def dataset_dirs(root: Path):
    for info in sorted(root.rglob("dataset_info.json")):
        yield info.parent


def arrow_freqs_and_rows(path: Path) -> tuple[list[str], int]:
    """Read only Arrow metadata/frequency buffers, never materialize targets."""
    freqs: set[str] = set()
    # HF save_to_disk 的 dataset_info.json 已含每 split 行数，直接读避免遍历全部 batch
    # （在 OBS 挂载上遍历 batch 等于把整个数据集下载一遍）。
    rows = 0
    info = path / "dataset_info.json"
    if info.is_file():
        info_data = json.loads(info.read_text(encoding="utf-8"))
        rows = sum(int(split.get("num_examples", 0))
                   for split in info_data.get("splits", {}).values())
    # freq 在同一数据集内固定，只读每个 arrow 文件的第一个 batch 即可。
    for arrow in sorted(path.glob("*.arrow")):
        source = pa.memory_map(str(arrow), "r")
        try:
            try:
                reader = pa.ipc.open_stream(source)
                first = next(iter(reader), None)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_file(source)
                first = reader.get_batch(0) if reader.num_record_batches else None
            if first is not None:
                idx = first.schema.get_field_index("freq")
                if idx >= 0:
                    freqs.update(norm_freq(x) for x in first.column(idx).unique().to_pylist())
        finally:
            source.close()
    return sorted(freqs), rows


def canonical_dataset(rel: str, freq: str) -> str:
    parts = rel.split("/")
    if len(parts) > 1 and norm_freq(parts[-1]) == freq:
        parts = parts[:-1]
    return "/".join(parts).lower()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-root", required=True)
    ap.add_argument("--gifteval-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--cases-per-source", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=260918)
    ap.add_argument("--ts-checkpoint", help="Optional model provenance for merged-SH resume checks")
    ap.add_argument("--clip-checkpoint", help="Optional model provenance for merged-SH resume checks")
    ap.add_argument("--resume", action="store_true", help="Check matching existing registry instead of overwriting it")
    args = ap.parse_args()
    out = Path(args.output_root)
    if args.cases_per_source < 1:
        ap.error("--cases-per-source must be positive")
    if (out / "source_registry.json").exists():
        if args.resume:
            check_resume(out/'source_registry.json',args)
            return
        raise FileExistsError("Registry already exists; resume later stages or choose a new output")
    registry: list[dict] = []
    for corpus, root_value in (("pretrain", args.pretrain_root), ("gifteval", args.gifteval_root)):
        root = Path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in dataset_dirs(root):
            rel = path.relative_to(root).as_posix()
            try:
                freqs, raw_record_count = arrow_freqs_and_rows(path)
                if not freqs:
                    raise ValueError("no frequency values")
            except Exception as exc:
                raise RuntimeError(f"Cannot discover source {path}: {exc}") from exc
            for freq in freqs:
                name = canonical_dataset(rel, freq)
                registry.append({
                    "corpus": corpus,
                    "dataset_name": name,
                    "dataset_relpath": rel,
                    "dataset_path": str(path),
                    "frequency": freq,
                    "source_id": f"{corpus}/{name}/{freq}",
                    "raw_record_count": raw_record_count,
                })
    # Some exported GIFTEval layouts contain both ``dataset`` and
    # ``dataset/frequency`` directories for the same logical source. Keep one
    # source deterministically so different ranks never write the same slug.
    deduplicated: dict[str, dict] = {}
    for row in registry:
        previous = deduplicated.get(row["source_id"])
        if previous is None or row["dataset_relpath"].count("/") > previous["dataset_relpath"].count("/"):
            deduplicated[row["source_id"]] = row
    registry = sorted(deduplicated.values(), key=lambda x: x["source_id"])
    pre = [x for x in registry if x["corpus"] == "pretrain"]
    gift = [x for x in registry if x["corpus"] == "gifteval"]
    if not pre or not gift:
        raise ValueError("Both roots must contain eligible Hugging Face save_to_disk Arrow datasets")
    for source in registry:
        source["quota"] = args.cases_per_source
    payload = {
        "version": "rag_gifteval_train_v1",
        "source_definition": "corpus/dataset/frequency",
        "generation_version": 10,
        "sampling": "80pct_longcontext_20pct_fullrange_train_only",
        "seed": args.seed,
        "cases_per_source": args.cases_per_source,
        "source_count": len(registry),
        "model_inputs": {key: str(Path(getattr(args,key)).resolve())
                         for key in ['ts_checkpoint','clip_checkpoint'] if getattr(args,key)},
        "sources": registry,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "source_registry.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sources": len(registry), "pretrain_sources": len(pre), "gifteval_sources": len(gift),
                      "requested_cases": sum(x["quota"] for x in registry), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
