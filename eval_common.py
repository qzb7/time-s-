"""Portable, history-only inference and GIFT-Eval metrics for two model modes."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
QUANTILES = [i / 10 for i in range(1, 10)]
METRICS = ['MSE[mean]', 'MSE[0.5]', 'MAE[0.5]', 'MASE[0.5]', 'MAPE[0.5]',
           'sMAPE[0.5]', 'MSIS', 'RMSE[mean]', 'NRMSE[mean]', 'ND[0.5]',
           'mean_weighted_sum_quantile_loss']
HEADER = ['dataset', 'model'] + ['eval_metrics/' + k for k in METRICS] + ['domain', 'num_variates']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def arguments(mode):
    p = argparse.ArgumentParser(description=f'GIFT-Eval {mode}: 97 reference tasks, no training.')
    p.add_argument('--gift-eval-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path, required=mode == 'unimodal',
                   default=ROOT / 'checkpoints/internal-22000')
    if mode == 'endogenous':
        p.add_argument('--cache-dir', required=True, type=Path)
        p.add_argument('--shared-head', type=Path, default=ROOT / 'checkpoints/joint-1000/shared_head_state.pt')
        p.add_argument('--no-shared-head', action='store_true',
                       help="Do not override output_head; use the checkpoint's own trained head")
    p.add_argument('--model-name', default='QiYao-M' if mode == 'endogenous' else 'QiYao-M-Numeric')
    p.add_argument('--context-length', type=int, default=8192)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='auto')
    p.add_argument('--include', action='append', default=[], help='Exact dataset[/frequency]/term; repeatable')
    p.add_argument('--max-samples', type=int, default=0, help='Preview only: first N inputs per configuration')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--finalize-only', action='store_true')
    a = p.parse_args()
    a.mode = mode
    if a.context_length <= 0 or a.context_length % 32 or a.batch_size <= 0 or a.max_samples < 0:
        p.error('Context must be positive and divisible by 32; batch > 0; max-samples >= 0')
    if mode == 'endogenous' and a.context_length != 8192:
        p.error('The reference endogenous cache requires context-length 8192')
    if not a.gift_eval_dir.is_dir():
        p.error('GIFT-Eval load_from_disk root not found')
    return a


def align_history(values, context=8192):
    values = np.asarray(values, dtype=np.float32).reshape(-1)[-context:]
    row = np.full(context, np.nan, dtype=np.float32)
    pad = np.ones(context, dtype=bool)
    if len(values):
        row[-len(values):] = values
        pad[-len(values):] = False
    return row, np.isfinite(row) & ~pad, pad


def impute(target, valid):
    """Exact vectorized forward-fill used by the 207 endogenous evaluator."""
    valid = valid & np.isfinite(target)
    safe = np.nan_to_num(target, nan=0., posinf=0., neginf=0.)
    positions = np.arange(target.shape[1])[None]
    source = np.maximum.accumulate(np.where(valid, positions, 0), axis=1)
    out = np.take_along_axis(safe, source, axis=1)
    first = valid.argmax(axis=1)
    out = np.where(positions < first[:, None], safe[np.arange(len(target)), first][:, None], out)
    out[~valid.any(axis=1)] = 0
    return out.astype(np.float32, copy=False)


def plan(args):
    os.environ['GIFT_EVAL'] = os.environ['GIFT_EVAL_PATH'] = str(args.gift_eval_dir.resolve())
    from gift_eval.data import Dataset
    catalogue = json.loads((ROOT / 'catalogue.json').read_text())
    requested = set(args.include)
    unknown = requested - {x['dataset'] + '/' + x['term'] for x in catalogue}
    if unknown:
        raise ValueError(f'Unknown configurations: {sorted(unknown)}')
    tasks = []
    for item in catalogue:
        key = item['dataset'] + '/' + item['term']
        if requested and key not in requested:
            continue
        meta = Dataset(name=item['dataset'], term=item['term'], to_univariate=False)
        count = len(meta.hf_dataset) * meta.target_dim * meta.windows
        if count != item['written_samples'] or meta.target_dim != item['num_variates']:
            raise ValueError(f'{key}: Dataset version/count does not match reference207')
        tasks.append(dict(item, key=key, frequency=str(meta.freq), horizon=meta.prediction_length,
                          fingerprint=meta.hf_dataset._fingerprint,
                          evaluate_samples=min(count, args.max_samples) if args.max_samples else count))
    return tasks


def checkpoint_files(args):
    root = args.checkpoint.resolve()
    if not (root / 'config.json').is_file():
        raise FileNotFoundError(f'Local checkpoint config.json missing: {root}')
    if (root / 'model.safetensors.index.json').exists():
        index = json.loads((root / 'model.safetensors.index.json').read_text())
        paths = [root / name for name in sorted(set(index['weight_map'].values()))]
        paths.append(root / 'model.safetensors.index.json')
    else:
        paths = [root / 'model.safetensors']
    paths.append(root / 'config.json')
    if args.mode == 'endogenous':
        paths += [root / 'layer16_concat_additive_state.pt', root / 'layer16_concat_additive_config.json']
        if not args.no_shared_head:
            paths.append(args.shared_head.resolve())
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError(f'Required checkpoint files missing: {[str(p) for p in paths if not p.is_file()]}')
    return {str(p): digest(p) for p in paths}


def protocol(args, tasks, device):
    sources = [ROOT / 'eval_common.py', ROOT / 'catalogue.json']
    for directory in ('qiyaom', 'dd_unit_scaling', 'gift_eval'):
        sources.extend(sorted((ROOT / directory).glob('*.py')))
    result = dict(mode=args.mode, checkpoint_sha256=checkpoint_files(args), model=args.model_name,
                  tasks=tasks, context_length=args.context_length, max_samples=args.max_samples,
                  batch_size=args.batch_size, backend=torch.device(device).type,
                  torch=torch.__version__, gluonts=importlib.metadata.version('gluonts'),
                  numpy=np.__version__, packages={k: importlib.metadata.version(k) for k in
                      ('datasets', 'pandas', 'unit-scaling', 'einops', 'huggingface-hub', 'safetensors')},
                  external_attention=False, quantiles=QUANTILES,
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources})
    if args.mode == 'endogenous':
        result['cache_root'] = str(args.cache_dir.resolve())
        result['cache_manifests'] = {}
        for item in tasks:
            manifest = cache_manifest(args, item)
            result['cache_manifests'][item['config']] = digest(manifest)
        result['training_exposure'] = ('Bundled weights: GIFT-Eval/TIME test-input adaptation; not clean zero-shot. '
                                       'For replacement weights, disclose their own provenance separately.')
    return result


def cache_manifest(args, task):
    root = args.cache_dir / task['config']
    path = root / 'manifest.json'
    if not path.exists() and args.max_samples:
        path = root / 'preview-manifest.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_device(local: int = 0) -> str:
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return f'npu:{local}'
    except Exception:
        pass
    if torch.cuda.is_available():
        return f'cuda:{local}'
    return 'cpu'


def load_model(args, device):
    # These match the native reference process defaults. No prediction clipping
    # or SDPA backend forcing is introduced by the evaluation wrapper.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    if args.mode == 'unimodal':
        from qiyaom import QiYaoMBackbone
        model = QiYaoMBackbone.from_pretrained(args.checkpoint, map_location='cpu', strict=True)
    else:
        from qiyaom.modeling_qiyaom import EndoMultimodalPredictor
        model = EndoMultimodalPredictor.from_pretrained(args.checkpoint, map_location='cpu')
        if args.no_shared_head:
            print(json.dumps(dict(event='shared_head_skipped', using_checkpoint_head=True, external_attention=False)), flush=True)
        else:
            head = torch.load(args.shared_head, map_location='cpu', weights_only=True)
            model.base_model.output_head.load_state_dict(head, strict=True)
            assert all(torch.equal(model.base_model.output_head.state_dict()[k].cpu(), v.cpu()) for k, v in head.items())
            print(json.dumps(dict(event='shared_head_verified', tensors=len(head), external_attention=False)), flush=True)
    if model.config.patch_size != 32:
        raise ValueError('This evaluator expects the compatible patch32 backbone')
    # Move each shared parameter once; no optimizer/training objects required.
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(device=device, dtype=torch.float32)
        for b in model.buffers():
            b.data = b.data.to(device=device, dtype=torch.float32 if b.is_floating_point() else b.dtype)
    return model.eval()


class Limited:
    def __init__(self, iterable, count):
        self.iterable, self.count = iterable, count

    def __iter__(self):
        return itertools.islice(iter(self.iterable), self.count)


class Predictor:
    def __init__(self, model, args, task, device):
        self.model, self.args, self.task, self.device = model, args, task, device
        self.seen = 0

    def cached_batches(self, dataset):
        args, task = self.args, self.task
        path = cache_manifest(args, task)
        manifest = json.loads(path.read_text())
        if manifest['dataset'] != task['dataset'] or manifest['term'] != task['term']:
            raise ValueError('Cache dataset/term mismatch')
        if not args.max_samples and (manifest.get('preview_only', False) or manifest['written_samples'] != task['written_samples']):
            raise ValueError('Cannot evaluate full dataset from a partial cache')
        if manifest['written_samples'] < task['evaluate_samples']:
            raise ValueError('Cache contains too few samples')
        iterator = iter(dataset)
        cursor = 0
        for shard in manifest['shards']:
            with np.load(path.parent / shard['file'], allow_pickle=False) as z:
                raw, valid, pad = (z[k] for k in ('time_series_8192', 'target_mask_8192', 'padding_mask_8192'))
                image, text = z['image_global'], z['text_global']
                wi, vi = z['window_index'], z['variate_index']
                n = len(wi)
                if raw.shape != (n, 8192) or valid.shape != raw.shape or pad.shape != raw.shape:
                    raise ValueError('Malformed numeric/mask cache')
                if image.shape != (n, 256, 512) or text.shape != image.shape:
                    raise ValueError('Malformed patch embedding cache')
                for left in range(0, n, args.batch_size):
                    take = min(args.batch_size, n - left, task['evaluate_samples'] - cursor)
                    if take <= 0:
                        return
                    sl = slice(left, left + take)
                    if not np.array_equal(wi[sl], np.arange(cursor, cursor + take)) or np.any(vi[sl] != 0):
                        raise ValueError('Cache windows duplicated, missing, or out of order')
                    entries = list(itertools.islice(iterator, take))
                    if len(entries) != take:
                        raise ValueError('Official inputs ended early')
                    # Critical safety check: embeddings must be aligned with the
                    # actual original historical values, not only window counts.
                    for j, entry in enumerate(entries, left):
                        x, m, p = align_history(entry['target'])
                        np.testing.assert_array_equal(raw[j], x)
                        np.testing.assert_array_equal(valid[j], m)
                        np.testing.assert_array_equal(pad[j], p)
                    if not np.isfinite(image[sl]).all() or not np.isfinite(text[sl]).all():
                        raise ValueError('Nonfinite cache embeddings; no samples are silently dropped')
                    cursor += take
                    yield entries, raw[sl], valid[sl] & ~pad[sl], image[sl], text[sl]
                    if cursor == task['evaluate_samples']:
                        return
        if cursor != task['evaluate_samples']:
            raise ValueError('Cache ended before every expected window was evaluated')

    def numeric_batches(self, dataset):
        iterator = iter(dataset)
        remaining = self.task['evaluate_samples']
        while remaining:
            entries = list(itertools.islice(iterator, min(remaining, self.args.batch_size)))
            if not entries:
                raise ValueError('Official inputs ended early')
            aligned = [align_history(x['target'], self.args.context_length) for x in entries]
            raw, valid, _ = map(np.stack, zip(*aligned))
            remaining -= len(entries)
            yield entries, raw, valid, None, None

    @torch.inference_mode()
    def predict(self, dataset, **kwargs):
        from gluonts.model.forecast import QuantileForecast
        batches = self.cached_batches(dataset) if self.args.mode == 'endogenous' else self.numeric_batches(dataset)
        for entries, raw, mask, image, text in batches:
            target = torch.from_numpy(impute(raw, mask)).to(self.device).unsqueeze(1)
            valid = torch.from_numpy(mask).to(self.device).unsqueeze(1)
            if self.args.mode == 'endogenous':
                im = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).unsqueeze(1)
                tx = torch.from_numpy(np.ascontiguousarray(text)).to(self.device).unsqueeze(1)
                q = self.model.forecast_parallel(target, valid, im, tx, self.task['horizon'],
                                                has_missing_values=True, scaler_fallback_min_obs=8)
            else:
                inputs = dict(target=target, target_mask=valid,
                              series_ids=torch.zeros(target.shape[:-1], dtype=torch.long, device=self.device))
                q = self.model.forecast(inputs, self.task['horizon'], has_missing_values=True,
                                       scaler_fallback_min_obs=8, decode_block_size=0, quantile_real_cap_k=0)
            if q.is_sparse:
                q = q.to_dense()
            if q.ndim == 5:
                q = q.flatten(-2, -1)
            expected = (9, len(entries), 1, self.task['horizon'])
            if tuple(q.shape) != expected or not torch.isfinite(q).all():
                raise ValueError(f'Invalid forecasts, expected {expected}, got {q.shape}; no filtering allowed')
            forecasts = q[:, :, 0].permute(1, 0, 2).cpu().numpy()
            for entry, forecast in zip(entries, forecasts):
                # Forecast time is computed from history; target labels are not
                # passed into either model or the predictor.
                start = entry['start'] + np.asarray(entry['target']).shape[-1]
                self.seen += 1
                yield QuantileForecast(forecast, start, [str(q) for q in QUANTILES], item_id=entry.get('item_id'))
            print(json.dumps(dict(event='progress', config=self.task['key'], done=self.seen,
                                  total=self.task['evaluate_samples'])), flush=True)


def evaluate(model, args, task, device):
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.ev.metrics import MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss
    from gluonts.time_feature import get_seasonality
    data = Dataset(name=task['dataset'], term=task['term'], to_univariate=task['num_variates'] != 1)
    original = data.test_data
    td = SimpleNamespace(input=Limited(original.input, task['evaluate_samples']),
                         label=Limited(original.label, task['evaluate_samples']))
    predictor = Predictor(model, args, task, device)
    result = evaluate_model(predictor, test_data=td,
        metrics=[MSE(forecast_type='mean'), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
                 RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=QUANTILES)],
        batch_size=args.batch_size, axis=None, mask_invalid_label=True,
        allow_nan_forecast=False, seasonality=get_seasonality(data.freq))
    if predictor.seen != task['evaluate_samples']:
        raise ValueError(f'Metrics consumed {predictor.seen} predictions, expected {task["evaluate_samples"]}')
    row = dict(dataset=task['official_dataset'], model=args.model_name,
               domain=task['domain'], num_variates=task['num_variates'])
    for key in METRICS:
        value = float(result[key].iloc[0])
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite metric {key} for {task["key"]}; not publishing an invalid result')
        row['eval_metrics/' + key] = value
    return dict(row=row, samples=predictor.seen, horizon=task['horizon'])


def finalize(args, tasks, signature):
    complete, missing = [], []
    for task in sorted(tasks, key=lambda x: x['official_order']):
        path = args.output / 'tasks' / (task['config'] + '.json')
        if not path.exists():
            missing.append(task['key']); continue
        result = json.loads(path.read_text())
        if result['signature'] != signature or result['samples'] != task['evaluate_samples']:
            raise ValueError(f'Incompatible result: {path}')
        complete.append(result)
    official = len(tasks) == 97 and not args.max_samples and not missing
    name = 'all_results.csv' if official else ('preview_results.csv' if args.max_samples else 'partial_results.csv')
    fd, temp = tempfile.mkstemp(dir=args.output, suffix='.csv.tmp')
    with os.fdopen(fd, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(x['row'] for x in complete)
    os.replace(temp, args.output / name)
    summary = dict(complete=not missing, leaderboard_ready_shape=official, configurations=len(complete),
                   samples=sum(x['samples'] for x in complete), missing=missing,
                   preview=bool(args.max_samples), results_file=name,
                   note='CSV schema alignment is not proof of zero-shot eligibility or leaderboard acceptance.')
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return not missing


def main(mode):
    args = arguments(mode)
    rank, world, local = (int(os.getenv(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if not 0 <= rank < world:
        raise ValueError('Invalid rank/world size')
    tasks = plan(args)
    print(json.dumps(dict(event='plan', configs=len(tasks), samples=sum(x['evaluate_samples'] for x in tasks),
                          rank=rank, world=world, preview=bool(args.max_samples))), flush=True)
    if args.plan_only:
        return
    device = args.device
    if device == 'auto':
        device = resolve_device(local)
    elif device == 'cuda':
        device = f'cuda:{local}' if torch.cuda.is_available() else 'cpu'
    if world > 1 and device.startswith(('cuda:', 'npu:')) and device != f'{device.split(":")[0]}:{local}':
        raise ValueError('Use --device auto under torchrun')
    torch.set_num_threads(max(1, int(os.getenv('OMP_NUM_THREADS', '1'))))
    proto = protocol(args, tasks, device)
    sig = hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / '.protocol.lock'):
        path = args.output / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text())['signature'] != sig:
                raise ValueError('Output protocol changed. Use a new output directory; never mix checkpoints/settings.')
        else:
            atomic_json(path, dict(signature=sig, **proto))
    if args.finalize_only:
        if not finalize(args, tasks, sig):
            raise RuntimeError('Some configurations are not finished')
        return
    # Greedy balancing by windows * forecast length, not GPU-to-GPU communication.
    bins, loads = [[] for _ in range(world)], [0] * world
    for task in sorted(tasks, key=lambda x: x['evaluate_samples'] * x['horizon'], reverse=True):
        owner = min(range(world), key=lambda i: (loads[i], i))
        bins[owner].append(task); loads[owner] += task['evaluate_samples'] * task['horizon']
    model = None
    for task in bins[rank]:
        path = args.output / 'tasks' / (task['config'] + '.json')
        with lock(path.with_suffix('.lock')):
            if path.exists():
                old = json.loads(path.read_text())
                if old['signature'] != sig or old['samples'] != task['evaluate_samples']:
                    raise ValueError(f'Incompatible old task {path}')
                print(json.dumps(dict(event='resume_skip', config=task['key'])), flush=True)
                continue
            if model is None:
                model = load_model(args, device)
            result = evaluate(model, args, task, device)
            atomic_json(path, dict(signature=sig, **result))
    if world == 1:
        if not finalize(args, tasks, sig):
            raise RuntimeError('Incomplete run')
    print(json.dumps(dict(event='rank_complete', rank=rank)), flush=True)
