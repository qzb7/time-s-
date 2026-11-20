"""Frozen endogenous backbone; staged shared-head / retrieval joint training."""
import argparse, json, os, time, math, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import zarr
import training_core as base
from load_replay_zarr import Toto2MultimodalZarrDataset
import replay_data as old
from cpm_features import forecast_cpm_hidden
from bank_sampler import RelationSampler as SixteenSourceSampler
from external_model import ExternalRAG


def resolve_device(local_rank, requested):
    """Pick an accelerator: NPU, then CUDA (under 'auto'), else CPU."""
    npu = None
    if requested in ("auto", "npu"):
        try:
            import torch_npu
            npu = torch_npu.npu
        except Exception:
            npu = None
    if npu is not None and npu.is_available():
        count = npu.device_count()
        if count < 1:
            raise RuntimeError("No NPU is visible to this rank")
        index = 0 if count == 1 else local_rank
        if index >= count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {count} NPUs are visible")
        device = torch.device("npu", index)
        npu.set_device(device)
        return device
    if requested == "npu":
        raise RuntimeError("NPU requested but unavailable; install torch_npu or choose another device")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        count = torch.cuda.device_count()
        index = 0 if count == 1 else local_rank
        if index >= count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {count} GPUs are visible")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        return device
    if requested == "cuda":
        raise RuntimeError("CUDA unavailable; activate your training environment first")
    return torch.device("cpu")


def peak_memory_mib(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "npu":
        import torch_npu
        return torch_npu.npu.max_memory_allocated() / 2**20
    return 0.0

class SourceDataset(Toto2MultimodalZarrDataset):
    def _read_chunk(self,path,left,right,order):
        g=zarr.open_group(str(path),mode='r')
        def texts(key):
            if key not in g:return ['']*(right-left)
            return [(v.decode() if isinstance(v,bytes) else str(v)).rstrip('\x00') for v in g[key][left:right]]
        terms,datasets=texts('source_term'),texts('source_dataset')
        for idx,row in zip(order,super()._read_chunk(path,left,right,order)):
            term,dataset=terms[int(idx)],datasets[int(idx)]
            if term.startswith('test_input_') and dataset.lower() in {'m4_yearly','m4_hourly'}:term='short'
            row.update(source_term=term,source_dataset=dataset,source_bank=path.parent.name)
            yield row

def make_loader(a, device=None):
    from replay_config import load_replay_config
    # pin_memory only helps CUDA copies; NPU has no cudaHostAlloc path and can error.
    pin = device is not None and device.type == 'cuda'
    cfg=load_replay_config(a.replay_config,a.cpm_max_span_patches,a.cpm_max_mask_probability) if a.replay_config else {
        'cpm_max_span_patches':a.cpm_max_span_patches, 'cpm_max_mask_probability':a.cpm_max_mask_probability}
    if a.replay_data_dir:
        paths=[str(Path(p)/'generation' if (Path(p)/'generation').is_dir() else Path(p)) for p in a.replay_data_dir]
        ds=SourceDataset(paths,seed=a.seed)
        return iter(DataLoader(ds,batch_size=a.old_batch_size,num_workers=0,pin_memory=pin,drop_last=True)),cfg
    def resolve(p):
        if isinstance(p, (list, tuple)):
            path, frac = p
            path = Path(path)
            return (str(path/'generation' if (path/'generation').is_dir() else path), frac)
        p=Path(p);return str(p/'generation' if (p/'generation').is_dir() else p)
    if 'test_input_data_dirs' in cfg:
        ds=old.TwoGroupReplayDataset(SourceDataset,
            test_dirs=[resolve(p) for p in cfg['test_input_data_dirs']],
            clean_dirs=[resolve(p) for p in cfg['clean_data_dirs']],
            batch_size=a.old_batch_size,test_ratio=cfg['test_input_ratio'],seed=a.seed)
        return iter(DataLoader(ds,batch_size=a.old_batch_size,num_workers=0,pin_memory=pin,drop_last=True)),cfg
    ds=old.TimeGiftCleanFixedHorizonDataset(SourceDataset,
        clean_dirs=[resolve(p) for p in cfg['clean_data_dirs']],
        earlier_context_dirs=[resolve(p) for p in cfg['earlier_context_data_dirs']],
        previous_targeted_dir=resolve(cfg['previous_targeted_data_dir']),
        new_targeted_dir=resolve(cfg['new_targeted_data_dir']),
        time_test_dir=resolve(cfg['time_test_data_dir']),batch_size=a.old_batch_size,
        source_ratios=(cfg['time_test_ratio'],cfg['gifteval_test_ratio'],cfg['clean_ratio']),
        gift_horizon_ratios=(cfg['gift_short_ratio'],cfg['gift_medium_ratio'],cfg['gift_long_ratio']),seed=a.seed)
    # This imported loader's actual iteration uses natural S/M/L, not quotas.
    return iter(DataLoader(ds,batch_size=a.old_batch_size,num_workers=0,pin_memory=pin,drop_last=True)),cfg

def schedule(step,total,peak,floor,start=0,warmup=500):
    if step<start:return 0.
    n=step-start+1
    if n<=warmup:return peak*n/warmup
    progress=min(1.,(n-warmup)/max(1,total-start-warmup))
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*progress))

def global_patch_loss(pred,wanted,mask,world):
    q=torch.arange(1,10,device=pred.device,dtype=pred.dtype).view(9,1,1,1,1)/10
    err=wanted[None]-pred
    point=torch.maximum(q*err,(q-1)*err).mean(0,dtype=torch.float32)
    s,n=base.masked_patch_sum_count(point,mask)
    count=n.detach().clone();logged=s.detach().clone()
    if world>1:dist.all_reduce(count);dist.all_reduce(logged)
    if count.item()==0:raise RuntimeError('No valid global future patches')
    return s*world/count,logged/count

def rag_loss(pred,wanted,mask,world):
    return global_patch_loss(pred,wanted[:,None],mask[:,None],world)

def cpm_loss(outputs,prepared,world):
    truth=torch.asinh((prepared['target']-outputs.loc)/outputs.scale).unflatten(-1,(256,32))[...,1:,:]
    valid=prepared['target_mask'].unflatten(-1,(256,32))[...,1:,:]&prepared['masked_patches'][...,1:,None]
    return global_patch_loss(outputs.quantiles[...,:-1,:],truth,valid,world)

@torch.no_grad()
def rag_features(model,batch):
    query,loc,scale=model.forecast_parallel_hidden(batch['target'][:,:8192,None].transpose(1,2),
        batch['target_mask'][:,:8192,None].transpose(1,2),batch['image'][:,None],batch['text'][:,None],
        horizon=192,forecast_patch_mask=batch['forecast_patch_mask'])
    target=torch.cat([model.normalize_known_target_patches(batch['candidate_future'][:,s:s+4],
        batch['candidate_future_mask'][:,s:s+4]) for s in range(0,batch['candidate_future'].shape[1],4)],dim=1)
    valid=batch['future_mask']&(torch.arange(192,device=query.device)[None]<batch['prediction_length'][:,None])
    wanted=torch.asinh((batch['future']-loc[:,0].flatten(-2))/scale[:,0].flatten(-2).clamp_min(1e-6)).unflatten(-1,(6,32))
    mask=valid.unflatten(-1,(6,32))&batch['forecast_patch_mask'][...,None]
    return query[:,0],target,wanted,mask

@torch.no_grad()
def old_features(model,batch,cfg,device):
    p=old.prepare_tail_first_batch(batch,device,patch_size=32,max_span_patches=cfg['cpm_max_span_patches'],max_mask_probability=cfg['cpm_max_mask_probability'])
    # Reproduce model.forward up to (but not through) output_head.
    x,loc,scale=model._time_embed(p['target'],p['target_mask'],p['cpm_mask'])
    n=x.shape[-2];visible=p['cpm_mask'].unflatten(-1,(n,32)).any(-1)
    groups=model._group_ids(p['target_mask'],p['cpm_mask'],p['series_ids'],n)
    h=model._encode(x,groups,batch['image_tokens'].to(device)[:,None],batch['text_tokens'].to(device)[:,None],visible)
    return h,loc,scale,p

def save(out,step,external,head,optim,a,device=None):
    ck=out/f'checkpoint-{step}';ck.mkdir(parents=True,exist_ok=False)
    torch.save(external.state_dict(),ck/'external_rag_state.pt')
    torch.save(head.state_dict(),ck/'shared_head_state.pt')
    accel_rng=None
    if device is not None and device.type=='cuda':
        accel_rng=torch.cuda.get_rng_state_all()
    elif device is not None and device.type=='npu':
        import torch_npu
        accel_rng=torch_npu.npu.get_rng_state()
    torch.save(dict(optimizer=optim.state_dict(),step=step,scheduler=dict(step=step,policy='two_group_cosine',args=vars(a)),
        torch_rng_state=torch.get_rng_state(),accelerator_rng_state=accel_rng),ck/'training_state.pt')
    (ck/'external_rag_config.json').write_text(json.dumps(vars(a),indent=2))
    (ck/'.checkpoint_complete').write_text('ok\n')

def main():
    from train_external import parse_args
    a=parse_args();a.protocol_version=11;a.training_version=11;a.shared_head_file='shared_head_state.pt'
    a.attention_prior='relation_encoding_K_only_nullK_zeroV';a.candidate_selection='GRW_meanaggregate1500_T1_times_mappedcos_top10_union24'
    a.query_normalization='original_cpm_no_backfill';a.sampling='16_distinct_sources_x32_without_replacement_when_possible'
    a.num_heads=16;a.loss_reduction='global_valid_masked_patch_mean'
    
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=resolve_device(local,a.device)
    if world>1:
        backend='nccl' if device.type=='cuda' else ('hccl' if device.type=='npu' else 'gloo')
        dist.init_process_group(backend)
    torch.manual_seed(a.seed);np.random.seed(a.seed+rank);random.seed(a.seed+rank)
    if device.type=='npu':
        import torch_npu
        torch_npu.npu.manual_seed_all(a.seed+rank)
    elif device.type=='cuda':
        torch.cuda.manual_seed_all(a.seed+rank)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model=base.EndoMultimodalPredictor.from_pretrained(a.internal_checkpoint,map_location='cpu').to(device).eval()
    model.requires_grad_(False);head=model.base_model.output_head
    import types
    model.forecast_parallel_hidden=types.MethodType(forecast_cpm_hidden,model)
    a.num_heads=int(model.base_model.config.num_heads)
    ext=ExternalRAG(d_model=model.d_model,heads=a.num_heads).to(device)
    ddp_device_ids=[local] if device.type in ('cuda','npu') else None
    ext_ddp=DDP(ext,device_ids=ddp_device_ids,broadcast_buffers=False) if world>1 else ext
    opt_kwargs=dict(weight_decay=a.weight_decay)
    if device.type=='cuda':
        opt_kwargs['fused']=True
    opt=torch.optim.AdamW([dict(params=ext.parameters(),lr=a.external_lr[0]),dict(params=head.parameters(),lr=0.)],**opt_kwargs)
    sampler=SixteenSourceSampler(a.bank_root,a.seed,rank)
    loader,cfg=make_loader(a,device)
    a.old_data_config=cfg;a.old_sampling=('custom_replay_dirs' if a.replay_data_dir else 'configured_test_input_clean' if 'test_input_data_dirs' in cfg else 'configured_TIME_GIFT_clean_natural');out=Path(a.output_dir)
    if rank==0:
        out.mkdir(parents=True,exist_ok=a.probe)
        (out/'training_args.json').write_text(json.dumps(vars(a),indent=2))
        print(json.dumps(dict(event='initialized',args=vars(a))),flush=True)
    if world>1:dist.barrier()
    params=list(ext.parameters())+list(head.parameters());head_runner=head
    steps=[a.stage1_steps-1,a.stage1_steps] if a.probe else range(a.stop_after_steps)
    initial_head={k:v.detach().clone() for k,v in head.state_dict().items()} if a.probe else None
    step_seconds_list = []
    data_seconds_total = 0.0
    total_samples = 0
    for step in steps:
        start=time.time();joint=step>=a.stage1_steps
        if joint and not next(head.parameters()).requires_grad:
            head.requires_grad_(True)
            head_runner=DDP(head,device_ids=ddp_device_ids,broadcast_buffers=False) if world>1 else head
            if rank==0:print(json.dumps(dict(event='head_unfrozen',step=step+1,optimizer_reset=False,backbone_frozen=True)),flush=True)
        opt.zero_grad(set_to_none=True)
        opt.param_groups[0]['lr']=schedule(step,a.max_steps,*a.external_lr,warmup=a.warmup_steps)
        opt.param_groups[1]['lr']=schedule(step,a.max_steps,*a.head_lr,start=a.stage1_steps,warmup=a.warmup_steps)
        logs={}
        if joint:
            h,loc,scale,prepared=old_features(model,next(loader),cfg,device)
            outputs=SimpleNamespace(quantiles=head_runner(h,q=None),loc=loc,scale=scale)
            loss,metric=cpm_loss(outputs,prepared,world)
            (a.loss_weights[0]*loss).backward();logs['loss_old']=float(metric)
            del h,loc,scale,prepared,outputs,loss
        t_s = time.time()
        b=base.to_device(sampler.sample(a.batch_size),device)
        data_seconds_total += time.time() - t_s
        total_samples += a.batch_size * world
        if step%50==0 or a.probe:print(json.dumps(dict(event='sampling',rank=rank,step=step+1,sources=sampler.last_report)),flush=True)
        h,target,wanted,mask=rag_features(model,b)
        if joint:
            loss,metric=rag_loss(head_runner(h[:,None],q=None),wanted,mask,world)
            (a.loss_weights[1]*loss).backward();logs['loss_rag_internal']=float(metric);del loss
        enhanced=ext_ddp(h,b['candidate_image'],b['candidate_text'],b['candidate_ts'],target,b['similarity'],b['source_weight'],b['temperature'],b['candidate_mask'],b['candidate_patch_mask'],b['forecast_patch_mask'])
        loss,metric=rag_loss(head_runner(enhanced[:,None],q=None),wanted,mask,world)
        ((a.loss_weights[2] if joint else 1.)*loss).backward();logs['loss_rag_external']=float(metric)
        if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('base_model.output_head.')):raise AssertionError('Frozen backbone received gradient')
        grad=torch.nn.utils.clip_grad_norm_(params,a.max_grad_norm,error_if_nonfinite=True)
        opt.step()
        step_seconds_list.append(time.time() - start)
        if a.probe:
            changed=any(not torch.equal(v,initial_head[k]) for k,v in head.state_dict().items())
            assert changed==joint,(changed,joint)
        if rank==0:
            print(json.dumps(dict(step=step+1,joint=joint,**logs,loss=(sum(w*logs[k] for w,k in zip(a.loss_weights,['loss_old','loss_rag_internal','loss_rag_external']))) if joint else logs['loss_rag_external'],grad_norm=float(grad),external_lr=opt.param_groups[0]['lr'],head_lr=opt.param_groups[1]['lr'],seconds=time.time()-start,peak_mib=peak_memory_mib(device))),flush=True)
        if (step+1)%a.save_steps==0 or step+1==a.stop_after_steps or a.probe:
            if world>1:dist.barrier()
            if rank==0:save(out,step+1,ext,head,opt,a,device)
            if world>1:dist.barrier()
        del b,h,target,wanted,mask,enhanced,loss
    steady = step_seconds_list[1:] if len(step_seconds_list) > 1 else []
    report = dict(
        benchmark='external_training', device=str(device), backend=device.type,
        world_size=world, batch_size=a.batch_size, grad_accum_steps=1,
        total_steps=len(step_seconds_list), total_samples=total_samples,
        avg_step_seconds=round(sum(step_seconds_list)/len(step_seconds_list), 3) if step_seconds_list else None,
        avg_step_seconds_excl_first=round(sum(steady)/len(steady), 3) if steady else None,
        throughput_samples_per_sec=round(total_samples/sum(step_seconds_list), 2) if step_seconds_list and sum(step_seconds_list) > 0 else None,
        total_data_seconds=round(data_seconds_total, 3),
        first_step_seconds=round(step_seconds_list[0], 3) if step_seconds_list else None,
        note='step = forward+backward+optimizer+sampling; data_seconds = sampler.sample only (joint replay load not separated); first step includes NPU graph compile',
    )
    out.mkdir(parents=True, exist_ok=True)
    name = 'training_speed.json' if world == 1 else f'training_speed.rank-{rank}.json'
    (out / name).write_text(json.dumps(report, indent=2))
    if a.probe and rank==0:print('JOINT_STAGE_TRANSITION_PROBE_OK',flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
