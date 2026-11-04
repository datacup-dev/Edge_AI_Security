"""Separate held-out test attacks using frozen training-only components."""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
import traceback
import numpy as np
from adv_mixed_data import PROJECT,ROOT,NAMES,Data,Target,read,save,replace,digest,project_path
from adv_mixed_algorithms import METHODS
from adv_mixed_run import LANES,main as generate

TEST=PROJECT/'artifacts/separate_test_attacks'
ROLE='test'
SPLIT_CODE=2
ENTRY='adv_test_control.py'
REPORT_FILE='TEST_ATTACK_REPORT.md'
PROTOCOL_FILE='TEST_ATTACKS.md'


def prepare():
    (TEST/'cohorts').mkdir(parents=True,exist_ok=True);data=Data();target=Target(data,2)
    training=read(ROOT/'configuration.json');assert target.hash==training['baseline_sha256']
    components={p.relative_to(PROJECT).as_posix():digest(p) for p in [ROOT/'surrogate/model.pt',ROOT/'GAN_model/model.pt',PROJECT/'artifacts/adapted_experiment/MLP/best.pt']}
    cohorts=[];consistency=[]
    for code,name in enumerate(NAMES):
        native=PROJECT/'artifacts/encoded_inputs'/name;split=np.load(native/'split.npy',mmap_mode='r');labels=np.load(native/'labels.npy',mmap_mode='r')
        rows=np.flatnonzero(split==SPLIT_CODE)
        if ROLE=='validation':
            sibling=PROJECT/'artifacts/separate_test_attacks/cohorts'/f'{name}.npy'
            if sibling.exists():assert np.all(split[np.load(sibling,mmap_mode='r')['source_row']]==2)
        assert not np.intersect1d(rows,data.records[data.records['dataset_id']==code]['source_row']).size
        records=np.empty(len(rows),dtype=data.records.dtype);records['dataset_id']=code;records['source_row']=rows;records['label']=labels[rows]
        np.save(TEST/'cohorts'/f'{name}.npy',records)
        output=TEST/'cohorts'/f'{name}_probability.npy'
        if not (TEST/'cohorts'/f'{name}_complete.json').exists():
            p=np.lib.format.open_memmap(output,mode='w+',dtype=np.float32,shape=(len(rows),))
            for start in range(0,len(records),16384):
                batch=records[start:start+16384];p[start:start+len(batch)]=target.compact(data.compact(batch),batch)
            p.flush();save(TEST/'cohorts'/f'{name}_complete.json',dict(rows=len(rows),baseline_sha256=target.hash))
        current=np.load(output,mmap_mode='r')
        if ROLE=='test':
            old=PROJECT/'artifacts/expanded_classical/GB/test'/name
            assert np.array_equal(np.load(old/'source_rows.npy'),rows)
            with np.load(old/'predictions.npz') as reference:old_probability=reference['probability']
            difference=np.abs(current-old_probability)
            assert np.allclose(current,old_probability,rtol=0,atol=2e-6) and np.array_equal(current>=.5,old_probability>=.5)
            consistency.append(dict(dataset=name,maximum_probability_difference=float(difference.max()),absolute_tolerance=2e-6,decision_differences=0))
        else:
            import xgboost as xgb
            from classical_data import sparse_batch
            maximum=0.
            for start in range(0,len(records),16384):
                batch=records[start:start+16384];native_values=data.local(batch,code)
                reference=target.model.predict(xgb.DMatrix(sparse_batch(native_values,code,data.offsets,157),nthread=2))
                observed=current[start:start+len(batch)];maximum=max(maximum,float(np.abs(reference-observed).max()))
                assert np.allclose(reference,observed,rtol=0,atol=2e-6) and np.array_equal(reference>=.5,observed>=.5)
            consistency.append(dict(dataset=name,reference='Independent original sparse GB adapter',maximum_probability_difference=maximum,absolute_tolerance=2e-6,decision_differences=0))
        cohorts.append(dict(dataset=name,rows=len(records),benign=int((records['label']==0).sum()),attack=int((records['label']==1).sum())))
    save(TEST/'configuration.json',dict(scope=f'All original {ROLE} records, both classes, four separate datasets',source_split=ROLE,split_code=SPLIT_CODE,cohorts=cohorts,baseline_sha256=target.hash,
        frozen_component_hashes=components,training_outputs=str(ROOT.relative_to(PROJECT)),heldout_records_used_for_fitting=0,epsilon=training['epsilon_standardized_linf'],clean_prediction_consistency=consistency))
    print('VERIFIED SEPARATE',ROLE.upper(),'COHORTS',cohorts,flush=True)


def report():
    config=read(TEST/'configuration.json');rows=[];done=complete=0
    for method in METHODS:
        for cohort in config['cohorts']:
            path=TEST/method/cohort['dataset'];state=read(path/'cursor.json') if (path/'cursor.json').exists() else None
            completed=(path/'complete.json').exists();complete+=completed;done+=state['done'] if state else 0
            for label in [0,1]:
                item=state['by_class'][str(label)] if state else dict(done=0,eligible=0,success=0,queries=0,capped=0,initialization_failed=0)
                rows.append(dict(method=method,dataset=cohort['dataset'],source_label=label,target_label=1-label,generated=item['done'],
                    required=cohort['benign' if label==0 else 'attack'],eligible=item['eligible'],successful=item['success'],
                    success_rate=item['success']/item['eligible'] if item['eligible'] else None,queries=item['queries'],query_cap_exhausted=item['capped'],initialization_failed=item['initialization_failed'],complete=completed))
    with (TEST/'progress.tmp.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    replace(TEST/'progress.tmp.csv',TEST/'progress.csv')
    failures={p.stem:read(p) for p in (TEST/'workers').glob('*failure.json')} if (TEST/'workers').exists() else {}
    total=sum(c['rows'] for c in config['cohorts'])*15
    status=dict(status='failed_worker' if failures else 'complete' if complete==60 else 'running',generated_records=done,required_records=total,complete_jobs=complete,total_jobs=60,
        both_classes=True,source_split=ROLE,training_heldout_outputs_separate=True,heldout_records_used_for_fitting=0,failures=failures)
    save(TEST/'status.json',status)
    prefix=str(TEST.relative_to(PROJECT)).replace('\\','/')
    lines=[f'# Separate adversarial {ROLE} outputs','',f"Status: **{status['status']}**; {done:,}/{total:,} candidates saved; {complete}/60 attack/dataset jobs complete.",'',
        f'Each of fifteen methods processes all records from all four original {ROLE} sets. Outputs remain separate from training and other held-out outputs, and separate by dataset. Original labels are retained, targets are opposite labels, and no {ROLE} data fit or tune the proxy or GAN.','',
        f'[Process and parameters]({PROTOCOL_FILE}). [Live per-class metrics]({prefix}/progress.csv). [Source provenance]({prefix}/configuration.json).','',
        'Saved candidates include failed, unchanged and initially misclassified cases. New success excludes existing baseline errors. Partial rates cannot rank final attack strength.','',
        '| Attack | Dataset | True → target | Saved / required | New successes / eligible | Complete |','|---|---|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['method']} | {r['dataset']} | {r['source_label']} → {r['target_label']} | {r['generated']:,} / {r['required']:,} | {r['successful']:,} / {r['eligible']:,} | {r['complete']} |")
    (PROJECT/REPORT_FILE).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(ROLE.upper(),'ATTACKS',done,'/',total,'complete jobs',complete,'/60',flush=True)
    path=PROJECT/'RUN_STATUS.md';old=path.read_text(encoding='utf-8')
    section='\n'.join([f'<!-- {ROLE}-attacks-start -->',f'## Separate adversarial {ROLE} generation','',
        f"Status: **{status['status']}**; {done:,}/{total:,} candidates; {complete}/60 jobs complete.",
        f'All four original {ROLE} sets, both classes. Frozen training-fitted attack components; no held-out fitting.',
        f'[Live report]({REPORT_FILE}). [Protocol]({PROTOCOL_FILE}).',f'<!-- {ROLE}-attacks-end -->',''])
    old=re.sub(rf'<!-- {ROLE}-attacks-start -->.*?<!-- {ROLE}-attacks-end -->\n?',lambda _:section,old,flags=re.S) if f'<!-- {ROLE}-attacks-start -->' in old else old.replace('# Run status\n','# Run status\n\n'+section,1)
    path.write_text(old,encoding='utf-8')


def worker(lane):
    config=read(TEST/'configuration.json')
    assert all(digest(project_path(path))==expected for path,expected in config['frozen_component_hashes'].items())
    pending=set(NAMES)
    while pending:
        for name in NAMES:
            if name not in pending:continue
            try:
                if generate(lane,test_dataset=name,one_cycle=True,heldout_role=ROLE):pending.remove(name)
            except (OSError,MemoryError) as error:
                if not isinstance(error,MemoryError) and getattr(error,'winerror',None)!=1455:raise
                import gc
                gc.collect()
                print('Paging memory temporarily exhausted; resuming saved cursor after 30 seconds',flush=True)
                time.sleep(30)


def audit(partial=False):
    data=Data();target=Target(data);config=read(TEST/'configuration.json');checked=[]
    assert target.hash==config['baseline_sha256']
    assert all(digest(project_path(path))==expected for path,expected in config['frozen_component_hashes'].items())
    for code,name in enumerate(NAMES):
        cohort=np.load(TEST/'cohorts'/f'{name}.npy',mmap_mode='r');native=PROJECT/'artifacts/encoded_inputs'/name
        split=np.load(native/'split.npy',mmap_mode='r');labels=np.load(native/'labels.npy',mmap_mode='r')
        assert np.array_equal(cohort['source_row'],np.flatnonzero(split==SPLIT_CODE))
        assert np.all(cohort['dataset_id']==code) and np.array_equal(cohort['label'],labels[cohort['source_row']])
    for method in METHODS:
        for name in NAMES:
            directory=TEST/method/name
            if not (directory/'cursor.json').exists():
                if partial:continue
                raise AssertionError(f'Incomplete {method}/{name}')
            state=read(directory/'cursor.json');records=np.load(TEST/'cohorts'/f'{name}.npy',mmap_mode='r')
            if not partial:assert (directory/'complete.json').exists() and state['done']==len(records)
            files=[p for p in sorted((directory/'shards').glob('*.npz')) if int(p.stem)<state['done']]
            if partial:files=files[:1]
            count=0
            for file in files:
                start=int(file.stem)
                if not partial:assert start==count
                with np.load(file) as z:
                    positions=z['cohort_position'];source=records[positions];clean=data.compact(source);adv=z['x_adversarial_native'];y=source['label']
                    assert np.array_equal(positions,np.arange(start,start+len(positions)))
                    assert str(z['source_split'])==ROLE and str(z['attack'])==method and np.all(z['mixed_position']==-1)
                    assert np.array_equal(z['source_row'],source['source_row']) and np.array_equal(z['source_dataset_id'],source['dataset_id'])
                    assert np.array_equal(z['labels'],y) and np.array_equal(z['target_labels'],1-y)
                    assert np.array_equal(np.isnan(clean),np.isnan(adv))
                    code=NAMES.index(name);width=data.widths[code]
                    assert np.array_equal(adv[:,:width],data.rules[code].project(adv[:,:width],clean[:,:width]))
                    diff=np.where(np.isfinite(clean),adv.astype(np.float64)-clean,0.)
                    assert np.all(np.abs(diff).max(1)<=.3+1e-7)
                    eligible=(z['clean_probability']>=.5)==y;changed=np.any(diff!=0,axis=1)
                    assert np.array_equal(z['success'],eligible&changed&((z['adversarial_probability']>=.5)!=y))
                    samples=np.array([np.flatnonzero(y==c)[0] for c in [0,1] if np.any(y==c)])
                    assert np.allclose(target.compact(adv[samples],source[samples]),z['adversarial_probability'][samples],rtol=0,atol=1e-6)
                    count+=len(source)
            if not partial:assert count==len(records)
            checked.append(dict(method=method,dataset=name,verified_records=count,status='passed'))
    if not partial:assert len(checked)==60
    save(TEST/('partial_audit.json' if partial else 'full_audit.json'),dict(status='passed',source_split=ROLE,full_generation_verified=not partial,checked=checked,frozen_components_unchanged=True,train_test_separate=True,training_heldout_separate=True))
    print(ROLE.upper(),'OUTPUT AUDIT PASSED',len(checked),'jobs; full generation verified',not partial,flush=True)


def launch():
    lock=TEST/'supervisor.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    logs=TEST/'logs';logs.mkdir(exist_ok=True);children=[]
    try:
        for lane in LANES:
            stream=(logs/f'{lane}.log').open('a',encoding='utf-8')
            process=subprocess.Popen([sys.executable,'-u',str(PROJECT/ENTRY),'worker','--lane',lane],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            children.append((lane,process,stream));print('STARTED TEST WORKER',lane,process.pid,flush=True)
        save(TEST/'supervisor.json',dict(pid=os.getpid(),workers={lane:p.pid for lane,p,_ in children}))
        while any(p.poll() is None for _,p,_ in children):
            subprocess.run([sys.executable,str(PROJECT/ENTRY),'report'],cwd=PROJECT);time.sleep(30)
        failures={lane:p.returncode for lane,p,_ in children if p.returncode};report()
        if failures:raise RuntimeError(failures)
        audit()
    finally:
        for _,_,stream in children:stream.close()
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','worker','report','audit']);parser.add_argument('--lane',choices=LANES);parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    try:
        if args.action=='worker':worker(args.lane)
        elif args.action=='audit':audit(args.partial)
        else:{'prepare':prepare,'launch':launch,'report':report}[args.action]()
    except Exception:
        if args.action=='worker':save(TEST/'workers'/f'{args.lane}_failure.json',dict(traceback=traceback.format_exc()))
        raise
