"""Run one attack and source role in isolated server outputs; no model fitting."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from adv_mixed_data import PROJECT,NAMES,ROOT,digest,read,save,project_path
import adv_mixed_run as worker
from adv_mixed_algorithms import METHODS

def main(role,method,dataset=None):
    manifest=read(PROJECT/'attack_generation_inputs.json')
    for entry in manifest['files']:
        path=project_path(entry['path'])
        assert path.is_file() and digest(path)==entry['sha256'],f'Missing/changed attack input: {path}'
    for relative in manifest['requires_existing']:assert project_path(relative).is_file(),f'Missing base training input: {relative}'
    assert digest(PROJECT/'artifacts/mixed_training/records.npy')==read(ROOT/'configuration.json')['mixed_sha256']
    output=PROJECT/'artifacts/server_attack_generation'/role
    output.mkdir(parents=True,exist_ok=True)
    lane=next(k for k,v in worker.LANES.items() if method in v)
    worker.LANES[lane]=[method]
    lock=output/f'{method}.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        save(output/f'{method}_run.json',dict(status='running',role=role,method=method,
            origin='Fresh server run; resumes its own saved shards, not workstation cursors',
            inputs='Frozen original baseline, proxies and GAN; no defense models or new fitting'))
        if role=='train':worker.main(lane,output_base=output)
        else:
            pending=[dataset] if dataset else list(NAMES)
            while pending:
                for name in pending.copy():
                    if worker.main(lane,test_dataset=name,one_cycle=True,heldout_role=role,output_base=output):pending.remove(name)
                    gc.collect()
        save(output/f'{method}_run.json',dict(status='generated',role=role,method=method,
            full_quality_audit_pending=True,output=str(output.relative_to(PROJECT))))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','validation','test'],required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--dataset',choices=NAMES)
    args=p.parse_args()
    if args.role=='train' and args.dataset:p.error('Mixed training keeps all four sources; --dataset applies only to held-out roles')
    main(args.role,args.method,args.dataset)
