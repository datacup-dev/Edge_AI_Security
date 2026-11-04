"""Actual saved-shard audit; all output methods reference the full mixed index."""
import argparse
import numpy as np
from adv_mixed_data import PROJECT,ROOT,NAMES,COUNT,EPS,Data,Target,read,save,digest
from adv_mixed_algorithms import METHODS,QUERY


def main(partial=False):
    data=Data();target=Target(data);config=read(ROOT/'configuration.json')
    assert target.hash==config['baseline_sha256'] and digest(data.mixed_path)==config['mixed_sha256']
    assert len(data.records)==COUNT
    for code,name in enumerate(NAMES):
        directory=PROJECT/'artifacts/encoded_inputs'/name
        split=np.load(directory/'split.npy',mmap_mode='r');labels=np.load(directory/'labels.npy',mmap_mode='r')
        records=data.records[data.records['dataset_id']==code]
        assert np.array_equal(np.sort(records['source_row']),np.flatnonzero(split==0))
        assert np.array_equal(records['label'],labels[records['source_row']])
    teacher=np.load(ROOT/'clean_probability.npy',mmap_mode='r');checked=[]
    for method in METHODS:
        directory=ROOT/method
        if not (directory/'cursor.json').exists():
            if partial:continue
            raise AssertionError(f'No completed chunks for {method}')
        state=read(directory/'cursor.json');done=state['done']
        if not partial:assert (directory/'complete.json').exists() and done==COUNT
        files=[f for f in sorted((directory/'shards').glob('*.npz')) if int(f.stem)<done]
        if partial:files=files[:1]
        count=0;by_class={str(y):dict(done=0,eligible=0,success=0,changed=0,queries=0,capped=0,initialization_failed=0) for y in [0,1]}
        for file in files:
            start=int(file.stem)
            if not partial:assert start==count
            with np.load(file) as z:
                positions=z['mixed_position'];assert np.array_equal(positions,np.arange(start,start+len(positions),dtype=np.uint32))
                records=data.records[positions];original=data.compact(records);adv=z['x_adversarial_native'];labels=records['label']
                assert str(z['attack'])==method and str(z['source_split'])=='train'
                assert adv.shape==original.shape==(len(records),64)
                assert np.array_equal(z['labels'],labels) and np.array_equal(z['target_labels'],1-labels)
                assert np.array_equal(np.isnan(original),np.isnan(adv))
                for code in range(4):
                    selected=np.flatnonzero(records['dataset_id']==code);f=data.widths[code];rule=data.rules[code]
                    assert np.array_equal(adv[selected,:f],rule.project(adv[selected,:f],original[selected,:f]))
                diff=np.where(np.isfinite(original),adv.astype(np.float64)-original.astype(np.float64),0.)
                assert np.all(np.abs(diff).max(1)<=EPS+1e-7)
                assert np.array_equal(z['l0'],(diff!=0).sum(1)) and np.allclose(z['l2'],np.linalg.norm(diff,axis=1))
                assert np.array_equal(z['linf'],np.abs(diff).max(1))
                assert np.array_equal(z['clean_probability'],teacher[positions])
                eligible=(teacher[positions]>=.5)==labels;changed=np.any(diff!=0,axis=1);success=eligible&changed&((z['adversarial_probability']>=.5)!=labels)
                assert np.array_equal(z['eligible'],eligible) and np.array_equal(z['attempted'],eligible)
                assert np.array_equal(z['changed'],changed) and np.array_equal(z['success'],success)
                samples=np.array([np.flatnonzero(labels==y)[0] for y in [0,1] if np.any(labels==y)],dtype=np.int64)
                assert np.allclose(target.compact(adv[samples],records[samples]),z['adversarial_probability'][samples],rtol=0,atol=1e-6)
                if method in QUERY:assert np.all(z['query_counts']<=4096)
                for y in [0,1]:
                    mask=labels==y;entry=by_class[str(y)]
                    for field,value in [('done',int(mask.sum())),('eligible',int(eligible[mask].sum())),('success',int(success[mask].sum())),('changed',int(changed[mask].sum())),
                        ('queries',int(z['query_counts'][mask].sum())),('capped',int(z['query_budget_exhausted'][mask].sum())),('initialization_failed',int(z['initialization_failed'][mask].sum()))]:entry[field]+=value
                count+=len(records)
        if not partial:
            assert count==COUNT
            for y in ['0','1']:
                for field,value in by_class[y].items():assert state['by_class'][y][field]==value
        checked.append(dict(method=method,verified_rows=count,classes_present=[int(y) for y in ['0','1'] if by_class[y]['done']],status='passed'))
    if not partial:assert len(checked)==15
    save(ROOT/('partial_audit.json' if partial else 'full_audit.json'),dict(status='passed',full_generation_verified=not partial,checked=checked,
        mixed_training_exact=True,both_classes_verified=True,baseline_unchanged=True,heldout_records_in_outputs=False,
        scope='First saved shard per available method' if partial else 'Every shard for every method'))
    print('SAVED OUTPUT AUDIT PASSED',len(checked),'methods; full generation verified:',not partial,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');main(parser.parse_args().partial)
