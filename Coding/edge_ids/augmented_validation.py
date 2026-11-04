"""Per-attack, per-dataset clean + adversarial validation snapshots. No fitting."""
import json
import numpy as np
from augmented_training import PROJECT,ATTACKS,NAMES,write

SOURCE=PROJECT/'artifacts/separate_validation_attacks'
OUT=PROJECT/'artifacts/augmented_validation'

def check(z,attack,code,records,start,count):
    r=records[start:start+count]
    assert str(z['source_split'])=='validation' and str(z['attack'])==attack
    assert np.all(z['mixed_position']==-1)
    assert np.array_equal(z['cohort_position'],np.arange(start,start+count))
    assert np.array_equal(z['labels'],r['label'])
    assert np.array_equal(z['source_row'],r['source_row'])
    assert np.all(z['source_dataset_id']==code)

def build():
    summary=[]
    for code,name in enumerate(NAMES):
        cohort=SOURCE/'cohorts'/f'{name}.npy'
        records=np.load(cohort,mmap_mode='r')
        native=PROJECT/'artifacts/encoded_inputs'/name
        split=np.load(native/'split.npy',mmap_mode='r')
        labels=np.load(native/'labels.npy',mmap_mode='r')
        assert np.array_equal(records['source_row'],np.flatnonzero(split==1))
        assert np.all(records['dataset_id']==code)
        assert np.array_equal(records['label'],labels[records['source_row']])
        for attack in ATTACKS:
            directory=SOURCE/attack/name
            state=json.loads((directory/'cursor.json').read_text()) if (directory/'cursor.json').exists() else {'done':0}
            committed=int(state['done']);assert 0<=committed<=len(records)
            files=[p for p in sorted((directory/'shards').glob('*.npz')) if int(p.stem)<committed]
            params=json.loads((directory/'parameters.json').read_text()) if files else {}
            if files:assert params['source_split']=='validation' and params['dataset']==name
            chunk=int(params.get('processing_batch',4096));end=0;shards=[]
            for i,p in enumerate(files):
                start=int(p.stem);assert start==end
                count=min(chunk,committed-start)
                if i in (0,len(files)-1):
                    with np.load(p) as z:
                        assert len(z['labels'])==count
                        check(z,attack,code,records,start,count)
                shards.append(dict(path=p.relative_to(PROJECT).as_posix(),start=start,rows=count))
                end+=count
            assert end==committed
            m=dict(attack=attack,dataset=name,split='validation',representation='zero-copy concatenation',
                clean_records=cohort.relative_to(PROJECT).as_posix(),clean_rows=len(records),
                adversarial_rows=end,total_rows=len(records)+end,expected_full_rows=2*len(records),
                status='complete' if end==len(records) else 'partial',shards=shards,
                labels='Original benign/attack labels, not target labels',
                inclusion='All committed candidates, including unchanged and unsuccessful attempts',
                verification='Every clean split membership and shard range; boundary shard provenance; every shard checked when read')
            write(OUT/attack/name/'manifest.json',m)
            summary.append({k:m[k] for k in ['attack','dataset','clean_rows','adversarial_rows','total_rows','status']})
    write(OUT/'status.json',summary)
    lines=['# Combined clean and adversarial validation datasets','',
        'Fifteen attacks times four datasets form 60 separate evaluation datasets. Each contains the complete original clean validation set followed by the committed validation candidates for that attack and dataset. Training and test records are excluded. Original class labels are retained, and failed or unchanged candidates remain included. The four native schemas remain separate.','',
        'These are fixed zero-copy snapshots, not duplicated feature files. Rerun `augmented_validation.py` to include newly generated shards. Partial generation remains explicitly marked. This step performs no training, model selection or evaluation.','',
        'Verification checks all clean cohort rows against the original split code 1 and labels, every committed shard range for contiguous coverage, and the first/last shard provenance. It is not an exhaustive new attack-quality audit. The batch reader checks provenance for every shard as it is consumed.','',
        'Use `from augmented_validation import batches`, then `batches(attack, dataset, batch_size=4096)`. It yields `(native_features, labels, dataset_ids, source_rows, is_adversarial)`. Features have 64 columns with native features first and NaN padding. `is_adversarial` identifies a generated candidate, not necessarily a successful attack. For valid evaluation, report clean and adversarial metrics separately using that flag as well as any combined metrics.','',
        '| Attack | Dataset | Clean | Adversarial | Combined | Status |','|---|---|---:|---:|---:|---|']
    lines += [f"| {r['attack']} | {r['dataset']} | {r['clean_rows']:,} | {r['adversarial_rows']:,} | {r['total_rows']:,} | {r['status']} |" for r in summary]
    (PROJECT/'AUGMENTED_VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Created',len(summary),'combined validation snapshots;',sum(r['status']=='complete' for r in summary),'complete;',sum(r['adversarial_rows'] for r in summary),'adversarial candidates',flush=True)

def batches(attack,dataset,batch_size=4096,*,include_clean=True):
    assert attack in ATTACKS and dataset in NAMES and batch_size>0
    m=json.loads((OUT/attack/dataset/'manifest.json').read_text());assert m['split']=='validation'
    records=np.load(PROJECT/m['clean_records'],mmap_mode='r');code=NAMES.index(dataset)
    if include_clean:
        f=np.load(PROJECT/'artifacts/encoded_inputs'/dataset/'features.npy',mmap_mode='r')
        for start in range(0,len(records),batch_size):
            r=records[start:start+batch_size];x=np.full((len(r),64),np.nan,np.float32)
            x[:,:f.shape[1]]=f[r['source_row']]
            yield x,r['label'].copy(),r['dataset_id'].copy(),r['source_row'].copy(),np.zeros(len(r),bool)
    for shard in m['shards']:
        with np.load(PROJECT/shard['path']) as z:
            check(z,attack,code,records,shard['start'],shard['rows'])
            x=z['x_adversarial_native'];assert x.shape==(shard['rows'],64)
            r=records[shard['start']:shard['start']+shard['rows']]
            for start in range(0,len(r),batch_size):
                sl=slice(start,start+batch_size)
                yield x[sl],r['label'][sl].copy(),r['dataset_id'][sl].copy(),r['source_row'][sl].copy(),np.ones(len(r[sl]),bool)

if __name__=='__main__':build()

