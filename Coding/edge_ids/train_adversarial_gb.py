"""Independent GB defenses on fixed, training-only augmented snapshots."""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
import shutil
import os
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy import sparse
from augmented_training import PROJECT,ATTACKS,NAMES,write,batches

OUT=PROJECT/'artifacts/adversarial_defenses'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dense_native(values,codes,xp=np):
    """Dense GPU-compatible namespace; missing blocks stay NaN, active zeros stay zero."""
    native=xp.asarray(values,dtype=xp.float32)
    result=xp.full((len(values),157),xp.nan,dtype=xp.float32)
    for code,(width,offset) in enumerate(zip([64,25,39,25],[0,64,89,128])):
        selected=np.flatnonzero(codes==code)
        if not len(selected):continue
        selected=xp.asarray(selected)
        result[selected,offset:offset+width]=native[selected,:width]
        result[selected,153+code]=1
    return result

class Iterator(xgb.DataIter):
    def __init__(self,attack,directory,sha,matrix_mode=None):
        self.attack=attack;self.sha=sha
        self.cuda=os.environ.get('EDGE_IDS_DEVICE','cpu')=='cuda'
        mode=matrix_mode or os.environ.get('EDGE_IDS_MATRIX','external')
        assert mode in ['external','memory']
        prefix=str(directory/'cache') if mode=='external' else None
        super().__init__(cache_prefix=prefix,release_data=True,on_host=self.cuda)
        self.reset()
    def reset(self):
        assert digest(PROJECT/'artifacts/augmented_training'/self.attack/'manifest.json')==self.sha,'Snapshot changed during training'
        self.stream=iter(batches(self.attack,16384));self.keep=None
    def next(self,input_data):
        try:x,y,codes,rows,is_adv=next(self.stream)
        except StopIteration:return False
        if self.cuda:
            import cupy as cp
            self.keep=dense_native(x,codes,cp)
            self.labels=cp.asarray(y)
            input_data(data=self.keep,label=self.labels)
            return True
        # Preserve original native namespace: inactive blocks absent, observed zero explicit.
        widths=[64,25,39,25];offsets=[0,64,89,128,153]
        counts=np.array([widths[int(c)]+1 for c in codes],np.int32)
        pointers=np.r_[0,np.cumsum(counts)].astype(np.int32)
        values=np.empty(int(pointers[-1]),np.float32);indices=np.empty(len(values),np.int32)
        for code,width in enumerate(widths):
            selected=np.flatnonzero(codes==code)
            if not len(selected):continue
            positions=pointers[selected,None]+np.arange(width+1)
            values[positions[:,:width]]=x[selected,:width];values[positions[:,width]]=1
            indices[positions]=np.r_[np.arange(offsets[code],offsets[code+1]),153+code]
        self.keep=sparse.csr_matrix((values,indices,pointers),shape=(len(y),157))
        self.labels=y
        input_data(data=self.keep,label=self.labels)
        return True

class Progress(xgb.callback.TrainingCallback):
    def __init__(self,directory,base,history):self.directory=directory;self.base=base;self.history=history
    def after_iteration(self,model,epoch,evals_log):
        step=self.base+epoch+1
        self.history.append(dict(round=step,train_logloss=float(evals_log['train']['logloss'][-1])))
        write(self.directory/'progress.json',dict(stage='boosting',round=step,total_rounds=300))
        if step%25==0:
            model.save_model(self.directory/'partial.ubj')
            write(self.directory/'history.json',self.history)
        return False

def train(attack):
    directory=OUT/attack;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return
    source=PROJECT/'artifacts/augmented_training'/attack/'manifest.json';sha=digest(source)
    manifest=json.loads(source.read_text());assert manifest['split']=='train'
    snapshot=directory/'training_manifest.json'
    if snapshot.exists():assert digest(snapshot)==sha,'Existing defense uses another snapshot; use a separate run'
    else:shutil.copyfile(source,snapshot)
    baseline=json.loads((PROJECT/'artifacts/expanded_classical/GB/configuration.json').read_text())
    parameters=dict(baseline['parameters']);parameters['device']=os.environ.get('EDGE_IDS_DEVICE','cpu');parameters['nthread']=int(os.environ.get('EDGE_IDS_THREADS','2'))
    if parameters['device']=='cuda':
        import cupy as cp
        gpu_pool=cp.cuda.MemoryAsyncPool()
        cp.cuda.set_allocator(gpu_pool.malloc)
    assert parameters['nthread']>0
    matrix_mode=os.environ.get('EDGE_IDS_MATRIX','external')
    assert matrix_mode in ['external','memory']
    write(directory/'configuration.json',dict(attack=attack,parameters=parameters,boosting_rounds=300,
        training_rows=manifest['total_rows'],snapshot_sha256=sha,source_status=manifest['status'],
        provisional=manifest['status']!='complete',initialization='Independent model from scratch, not appended baseline trees',
        validation_rule='Fixed 300 rounds, no early stopping or validation/test selection',
        optimizer='Gradient-boosted trees; no neural optimizer',learning_rate=parameters['eta'],dropout='Not applicable',
        memory=f"{parameters['device']} {matrix_mode} histogram; same baseline tree hyperparameters",
        labels='Original labels; unchanged and failed attack candidates included',fitting_splits=['train']))
    write(directory/'progress.json',dict(stage='training_quantile_construction',rows=manifest['total_rows']))
    cache_root=os.environ.get('EDGE_IDS_CACHE_DIR')
    cache_directory=Path(cache_root).expanduser().resolve()/attack if cache_root else directory
    cache_directory.mkdir(parents=True,exist_ok=True)
    write(directory/'runtime.json',dict(threads=parameters['nthread'],matrix=matrix_mode,device=parameters['device'],cache_directory=str(cache_directory),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    started=time.time();iterator=Iterator(attack,cache_directory,sha,matrix_mode=matrix_mode)
    matrix_class=xgb.ExtMemQuantileDMatrix if matrix_mode=='external' else xgb.QuantileDMatrix
    training=matrix_class(iterator,max_bin=128,nthread=parameters['nthread'])
    assert training.num_row()==manifest['total_rows'] and training.num_col()==157
    resume=None;base=0;history=[]
    if (directory/'partial.ubj').exists():
        resume=xgb.Booster();resume.load_model(directory/'partial.ubj');base=resume.num_boosted_rounds()
        if (directory/'history.json').exists():history=[v for v in json.loads((directory/'history.json').read_text()) if v['round']<=base]
    callback=Progress(directory,base,history)
    model=xgb.train(parameters,training,num_boost_round=300-base,evals=[(training,'train')],
        xgb_model=resume,callbacks=[callback],verbose_eval=25)
    assert model.num_boosted_rounds()==300
    model.save_model(directory/'model.ubj');write(directory/'history.json',callback.history)
    write(directory/'complete.json',dict(status='trained',attack=attack,rounds=300,training_rows=manifest['total_rows'],
        provisional=manifest['status']!='complete',model_sha256=digest(directory/'model.ubj'),seconds=time.time()-started,
        evaluation_performed=False))
    del training,iterator;gc.collect()

def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'scheduler.lock'
    with lock.open('x') as f:f.write(str(__import__('os').getpid()))
    results={}
    try:
        for attack in ATTACKS:
            write(OUT/'status.json',dict(stage='training',current_attack=attack,results=results))
            with (OUT/f'{attack}.log').open('a') as log:
                result=subprocess.run([sys.executable,'-u',__file__,'--attack',attack],stdout=log,stderr=subprocess.STDOUT)
            results[attack]='trained' if result.returncode==0 else 'failed; see log'
        write(OUT/'status.json',dict(stage='finished' if all(v=='trained' for v in results.values()) else 'finished_with_failures',results=results))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attack',choices=ATTACKS)
    parser.add_argument('--threads',type=int,default=int(os.environ.get('EDGE_IDS_THREADS','2')))
    parser.add_argument('--matrix',choices=['external','memory'],default=os.environ.get('EDGE_IDS_MATRIX','external'))
    parser.add_argument('--cache-dir',default=os.environ.get('EDGE_IDS_CACHE_DIR'))
    parser.add_argument('--device',choices=['cpu','cuda'],default=os.environ.get('EDGE_IDS_DEVICE','cpu'))
    args=parser.parse_args();assert args.threads>0
    os.environ['EDGE_IDS_THREADS']=str(args.threads);os.environ['EDGE_IDS_MATRIX']=args.matrix
    os.environ['EDGE_IDS_DEVICE']=args.device
    if args.cache_dir:os.environ['EDGE_IDS_CACHE_DIR']=args.cache_dir
    if args.attack:train(args.attack)
    else:launch()
