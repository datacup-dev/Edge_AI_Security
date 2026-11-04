"""One independent full mixed-training output per attack, both true classes."""
import argparse
import random
import time
import traceback
import numpy as np
import torch
from torch import nn
from adv_mixed_data import ROOT,PROJECT,NAMES,COUNT,EPS,Data,Target,read,save,replace,proxy
from adv_mixed_prepare import Generator
from adv_mixed_algorithms import METHODS,PARAMETERS,GRADIENT,SCORE,QUERY,gradient,scores,queries

LANES={'gradient':GRADIENT,'score':SCORE,'Boundary':['Boundary'],'HopSkipJump':['HopSkipJump'],'ZOO':['ZOO'],'GAN':['GAN']}


def empty_state(total=COUNT):
    return dict(done=0,total=total,by_class={str(y):dict(done=0,eligible=0,success=0,changed=0,queries=0,capped=0,initialization_failed=0,l2_sum=0.,linf_sum=0.) for y in [0,1]},seconds=0.)


def main(lane,test_dataset=None,one_cycle=False,heldout_role='test',output_base=None):
    assert read(ROOT/'preflight.json')['status']=='passed'
    torch.set_num_threads(2);data=Data();target=Target(data);config=read(ROOT/'configuration.json');assert target.hash==config['baseline_sha256']
    assert config['epsilon_standardized_linf']==EPS
    assert heldout_role in ['test','validation']
    input_root=ROOT if test_dataset is None else PROJECT/'artifacts'/f'separate_{heldout_role}_attacks'
    output_root=input_root if output_base is None else output_base
    if test_dataset is not None:
        data.records=np.load(input_root/'cohorts'/f'{test_dataset}.npy',mmap_mode='r')
    count=len(data.records)
    probability_path=ROOT/'clean_probability.npy' if test_dataset is None else input_root/'cohorts'/f'{test_dataset}_probability.npy'
    probabilities=np.load(probability_path,mmap_mode='r');proxies={};generators=None
    batch_size=4096 if lane in ['gradient','GAN'] else 512 if lane=='score' else 64
    if lane=='gradient':proxies={True:proxy(data,True),False:proxy(data,False)}
    if lane=='GAN':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');generators=nn.ModuleList(Generator(f) for f in data.widths).to(device)
        generators.load_state_dict(torch.load(ROOT/'GAN_model/model.pt',map_location=device,weights_only=True)['state_dict']);generators.eval()
    tasks=[]
    for method in LANES[lane]:
        directory=output_root/method if test_dataset is None else output_root/method/test_dataset
        (directory/'shards').mkdir(parents=True,exist_ok=True)
        method_record=dict(method=method,parameters=PARAMETERS[method],epsilon=EPS,processing_batch=batch_size,
            scope='All mixed training records, both classes',labels='Original labels retained; target=1-label',mixed_sha256=config['mixed_sha256'],baseline_sha256=target.hash)
        if test_dataset is not None:method_record.update(scope=f'All original {heldout_role} records, both classes',dataset=test_dataset,source_split=heldout_role,fitting_performed=False)
        if (directory/'parameters.json').exists():assert read(directory/'parameters.json')==method_record,'Changed settings require a separate run directory'
        else:save(directory/'parameters.json',method_record)
        state=read(directory/'cursor.json') if (directory/'cursor.json').exists() else empty_state(count);assert state['total']==count
        tasks.append(dict(method=method,directory=directory,state=state))
    while True:
        active=[task for task in tasks if task['state']['done']<count]
        if not active:break
        for task in active:
            state=task['state'];method=task['method'];start=state['done'];end=min(start+batch_size,count)
            records=data.records[start:end];y=np.asarray(records['label']).copy();x=data.compact(records);adv=x.copy()
            clean=np.asarray(probabilities[start:end]).copy();eligible=(clean>=.5)==y
            q=np.zeros(len(y),np.int64);capped=np.zeros(len(y),bool);init_failed=np.zeros(len(y),bool)
            seed=42+METHODS.index(method)*100_000_000+start
            if test_dataset is not None:seed+=(2_000_000_000 if heldout_role=='test' else 3_000_000_000)+NAMES.index(test_dataset)*50_000_000
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);rng=np.random.default_rng(seed);started=time.perf_counter()
            worker_name=lane if test_dataset is None else lane+'_'+test_dataset
            save(output_root/'workers'/f'{worker_name}.json',dict(status='running',method=method,current_start=start,current_end=end,total=count,completed_records=sum(t['state']['done'] for t in tasks)))
            for code in range(4):
                selected=np.flatnonzero((records['dataset_id']==code)&eligible)
                if not len(selected):continue
                width=data.widths[code];inputs=x[selected,:width];truth=y[selected];rule=data.rules[code];before=target.queries
                if method in GRADIENT:
                    model,device=proxies[method!='Transfer'];generated=gradient(method,inputs,truth,code,model,device,rule,rng)
                elif method in SCORE:
                    generated=scores(method,inputs,truth,code,target,rule,rng);used=target.queries-before
                    assert used%len(selected)==0;q[selected]=used//len(selected)
                elif method in QUERY:
                    generated,q[selected],capped[selected],init_failed[selected]=queries(method,inputs,truth,code,target,rule,rng)
                else:
                    with torch.inference_mode():
                        tensor=torch.as_tensor(inputs,device=device);target_labels=torch.as_tensor((1-truth).astype(np.int64),device=device)
                        generated=rule.tensor(tensor+generators[code](tensor,target_labels),tensor).cpu().numpy()
                generated=rule.project(generated,inputs)
                assert np.isfinite(generated).all() and np.array_equal(generated[:,~rule.mask],inputs[:,~rule.mask])
                adv[selected,:width]=generated
            p=target.compact(adv,records);diff=np.where(np.isfinite(x),adv.astype(np.float64)-x.astype(np.float64),0.)
            assert np.array_equal(np.isnan(x),np.isnan(adv))
            l0=(diff!=0).sum(1);l2=np.linalg.norm(diff,axis=1);linf=np.abs(diff).max(1);changed=l0>0
            success=eligible&changed&((p>=.5)!=y)
            assert np.all(linf<=EPS+1e-7)
            path=task['directory']/'shards'/f'{start:012d}.npz';temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,x_adversarial_native=adv,mixed_position=np.arange(start,end,dtype=np.int64) if test_dataset is None else np.full(len(y),-1,np.int64),
                    cohort_position=np.arange(start,end,dtype=np.uint32),source_dataset_id=records['dataset_id'],source_row=records['source_row'],labels=y,target_labels=(1-y).astype(np.uint8),
                    attack=np.array(method),source_split=np.array('train' if test_dataset is None else heldout_role),generation_seed=np.array(seed,np.int64),
                    clean_probability=clean,adversarial_probability=p,eligible=eligible,attempted=eligible,changed=changed,success=success,
                    l0=l0,l2=l2,linf=linf,query_counts=q,query_budget_exhausted=capped,initialization_failed=init_failed)
            replace(temporary,path)
            for label in [0,1]:
                selected=y==label;entry=state['by_class'][str(label)]
                for field,value in [('done',int(selected.sum())),('eligible',int(eligible[selected].sum())),('success',int(success[selected].sum())),('changed',int(changed[selected].sum())),
                    ('queries',int(q[selected].sum())),('capped',int(capped[selected].sum())),('initialization_failed',int(init_failed[selected].sum())),('l2_sum',float(l2[selected].sum())),('linf_sum',float(linf[selected].sum()))]:entry[field]+=value
            state['done']=end;state['seconds']+=time.perf_counter()-started;save(task['directory']/'cursor.json',state)
            if end==count:save(task['directory']/'complete.json',dict(status='complete',**state))
            print(method,end,'/',count,'success_by_class',[state['by_class'][str(c)]['success'] for c in [0,1]],flush=True)
        if one_cycle:break
    finished=all(t['state']['done']==count for t in tasks)
    worker_name=lane if test_dataset is None else lane+'_'+test_dataset
    save(output_root/'workers'/f'{worker_name}.json',dict(status='complete' if finished else 'running',completed_records=sum(t['state']['done'] for t in tasks)))
    return finished


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lane',choices=LANES,required=True);lane=parser.parse_args().lane
    try:main(lane)
    except Exception:
        save(ROOT/'workers'/f'{lane}_failure.json',dict(status='failed',traceback=traceback.format_exc()));raise
