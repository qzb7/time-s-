"""Plan independently sized GIFT-Eval/TIME official test-history views."""
import argparse
import json
import os
from pathlib import Path
import random

from test_inputs import ROOT,CROP_LENGTHS,view_counts,dataset_api,context_order


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gifteval-dir',type=Path,required=True)
    p.add_argument('--time-dir',type=Path,required=True)
    p.add_argument('--gifteval-test-count',type=int,default=0)
    p.add_argument('--time-test-count',type=int,default=0)
    p.add_argument('--seed',type=int,default=20260902)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.gifteval_test_count,a.time_test_count) < -1:
        p.error('Test count:0 skip,-1 all,positive total across this benchmark')
    if a.output.exists():
        raise FileExistsError('Choose a new output directory; never overwrite a test plan')
    GiftDataset,TimeDataset=dataset_api()
    families=[]
    excluded=[]
    if a.gifteval_test_count:
        if not a.gifteval_dir.is_dir():
            raise FileNotFoundError(a.gifteval_dir)
        os.environ['GIFT_EVAL']=str(a.gifteval_dir.resolve())
        catalogue=json.loads((ROOT/'gifteval_test_catalogue.json').read_text())['configs']
        tasks=[]
        for item in catalogue:
            if not (a.gifteval_dir/item['dataset']).is_dir():
                excluded.append(dict(benchmark='gifteval',dataset=item['dataset'],term=item['term'],reason='not_installed'))
                continue
            print(f"[prepare] gifteval {item['dataset']}/{item['term']}: loading ...", flush=True)
            ds=GiftDataset(name=item['dataset'],term=item['term'],to_univariate=False)
            print(f"[prepare] gifteval {item['dataset']}/{item['term']}: done", flush=True)
            tasks.append(dict(benchmark='gifteval',dataset=item['dataset'],term=item['term'],
                data_root=str(a.gifteval_dir.resolve()),num_variates=ds.target_dim,
                horizon=ds.prediction_length,available=len(ds.hf_dataset)*ds.target_dim*ds.windows,
                fingerprint=ds.hf_dataset._fingerprint,source_type=3))
        families.append(('gifteval',a.gifteval_test_count,tasks))
    if a.time_test_count:
        if not a.time_dir.is_dir():
            raise FileNotFoundError(a.time_dir)
        from timebench.evaluation.data import load_dataset_config,get_dataset_settings
        config=load_dataset_config(ROOT/'vendor/time_src/timebench/config/datasets.yaml')
        tasks=[]
        for name,setting in sorted(config['datasets'].items()):
            if not (a.time_dir/name).is_dir():
                excluded.append(dict(benchmark='time',dataset=name,reason='not_installed'))
                continue
            for term in ('short','medium','long'):
                if term not in setting:
                    continue
                settings=get_dataset_settings(name,term,config)
                print(f"[prepare] time {name}/{term}: loading ...", flush=True)
                ds=TimeDataset(name=name,term=term,to_univariate=False,
                    storage_path=a.time_dir,**settings)
                print(f"[prepare] time {name}/{term}: done", flush=True)
                tasks.append(dict(benchmark='time',dataset=name,term=term,
                    data_root=str(a.time_dir.resolve()),num_variates=ds.target_dim,
                    horizon=ds.prediction_length,available=len(ds.hf_dataset)*ds.target_dim*ds.windows,
                    fingerprint=ds.hf_dataset._fingerprint,settings=settings,source_type=5))
        families.append(('time',a.time_test_count,tasks))
    output=[]
    summaries={}
    offset=0
    for family,requested,tasks in families:
        if not tasks or not sum(t['available'] for t in tasks):
            raise ValueError(f'No installed official test input for requested benchmark:{family}')
        random.Random(a.seed+ (3 if family=='gifteval' else 5)).shuffle(tasks)
        allocations=view_counts(requested,[t['available'] for t in tasks])
        summaries[family]=dict(requested=requested,actual=sum(allocations),
            available_unique_windows=sum(t['available'] for t in tasks),installed_settings=len(tasks),
            settings_with_samples=sum(n>0 for n in allocations),
            repeated_views=sum(max(0,n-t['available']) for t,n in zip(tasks,allocations)))
        for task,count in zip(tasks,allocations):
            key=family+'/'+task['dataset']+'/'+task['term']
            indices=context_order(task['available'],count,a.seed,key)
            output.append(dict(task,key=key,selected=count,global_offset=offset,context_indices=indices))
            offset+=count
    plan=dict(version=3,seed=a.seed,total=offset,benchmarks=summaries,tasks=output,
        excluded=excluded,labels_accessed=False,intended_use='official_test_history_views',
        policy='balanced setting quotas; read test_data.input ONLY; cycle shuffled contexts with tail crops, like TIME targeted200k; no future labels; adaptation using test histories is transductive, not strict zero-shot',
        crop_lengths=list(CROP_LENGTHS),sequence_length=8192,patch_size=32)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp.json')
    temp.write_text(json.dumps(plan,indent=2))
    temp.replace(a.output)
    print(json.dumps(dict(total=offset,benchmarks=summaries,excluded_settings=len(excluded),plan=str(a.output)),indent=2),flush=True)


if __name__=='__main__':
    main()
