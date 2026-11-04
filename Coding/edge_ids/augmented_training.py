"""Create immutable, zero-copy clean+attack training snapshots and batch readers.

Run again to include newly committed attack shards. No validation/test inputs.
"""
import json
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parent
ATTACKS=['FGSM','CW','PGD','JSMA','DeepFool','BIM','MIM','Transfer','Boundary','HopSkipJump','ZOO','NES','SPSA','Square','GAN']
NAMES=['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']
OUT=PROJECT/'artifacts/augmented_training'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)

def build():
    clean_path=PROJECT/'artifacts/mixed_training/records.npy'
    records=np.load(clean_path,mmap_mode='r')
    splits=[np.load(PROJECT/'artifacts/encoded_inputs'/n/'split.npy',mmap_mode='r') for n in NAMES]
    for code in range(4):
        rows=records['source_row'][records['dataset_id']==code]
        assert np.all(splits[code][rows]==0),'Non-training clean record'
    rows=[]
    for attack in ATTACKS:
        source=PROJECT/'artifacts/mixed_training_attacks'/attack
        state=json.loads((source/'cursor.json').read_text()) if (source/'cursor.json').exists() else {'done':0}
        committed=int(state['done']);end=0;shards=[]
        files=[f for f in sorted((source/'shards').glob('*.npz')) if int(f.stem)<committed]
        parameters=json.loads((source/'parameters.json').read_text()) if files else {}
        chunk=int(parameters.get('processing_batch',4096))
        for index,file in enumerate(files):
            start=int(file.stem)
            if start>=committed:break
            assert start==end,(attack,start,end)
            count=min(chunk,committed-start)
            # Check every filename/range, and inspect boundary shard provenance.
            # The reader checks labels on every shard when it is consumed.
            if index in (0,len(files)-1):
              with np.load(file) as z:
                positions=z['mixed_position'];count=len(positions)
                assert count and start+count<=committed
                assert np.array_equal(positions,np.arange(start,start+count))
                assert np.array_equal(z['labels'],records['label'][start:start+count])
                if 'source_split' in z:assert str(z['source_split'])=='train'
                if 'attack' in z:assert str(z['attack'])==attack
                if 'source_row' in z:assert np.array_equal(z['source_row'],records['source_row'][start:start+count])
            shards.append({'path':file.relative_to(PROJECT).as_posix(),'start':start,'rows':count})
            end+=count
        assert end==committed,(attack,end,committed)
        manifest=dict(attack=attack,representation='zero-copy concatenation',
            clean_records=clean_path.relative_to(PROJECT).as_posix(),clean_rows=len(records),
            adversarial_rows=end,total_rows=len(records)+end,expected_full_rows=2*len(records),
            status='complete' if end==len(records) else 'partial',
            verification='All clean split memberships and shard ranges; first/last shard provenance; labels rechecked on consumption',
            labels='Original binary labels, never attack target labels',
            inclusion='All committed candidates, including unchanged and unsuccessful attempts',
            split='train',shards=shards)
        write(OUT/attack/'manifest.json',manifest)
        rows.append({k:manifest[k] for k in ['attack','clean_rows','adversarial_rows','total_rows','status']})
        print(rows[-1],flush=True)
    write(OUT/'status.json',rows)
    lines=['# Augmented training datasets','',
        'Each dataset concatenates all original mixed training records with one attack’s committed training candidates. Validation and test records are excluded. These are zero-copy datasets; the batch reader resolves original feature arrays and saved adversarial shards without duplicating them. Labels retain their original class. Unchanged/failed candidates are included and therefore can duplicate clean examples.','',
        'Snapshots are fixed at build time. Rerun `augmented_training.py` as generation advances. A partial dataset must not be presented as a completed full-pool experiment. No defended model has been trained by this step.','',
        '| Attack | Clean | Adversarial | Combined | Status |','|---|---:|---:|---:|---|']
    lines += [f"| {r['attack']} | {r['clean_rows']:,} | {r['adversarial_rows']:,} | {r['total_rows']:,} | {r['status']} |" for r in rows]
    lines += ['', 'Use `batches(attack, batch_size=4096)` from `augmented_training.py`. It yields `(native_features, labels, dataset_ids, source_rows, is_adversarial)`; native features occupy the first F columns with NaN padding to 64. Dataset identity selects the native encoder. Iteration is clean then adversarial; training code must add reproducible shuffling, using only training records.']
    (PROJECT/'AUGMENTED_TRAINING.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def batches(attack,batch_size=4096):
    assert attack in ATTACKS and batch_size>0
    m=json.loads((OUT/attack/'manifest.json').read_text())
    records=np.load(PROJECT/m['clean_records'],mmap_mode='r')
    features=[np.load(PROJECT/'artifacts/encoded_inputs'/n/'features.npy',mmap_mode='r') for n in NAMES]
    for start in range(0,len(records),batch_size):
        r=records[start:start+batch_size];x=np.full((len(r),64),np.nan,np.float32)
        for code,f in enumerate(features):
            selected=r['dataset_id']==code
            x[selected,:f.shape[1]]=f[r['source_row'][selected]]
        yield x,r['label'].copy(),r['dataset_id'].copy(),r['source_row'].copy(),np.zeros(len(r),bool)
    for shard in m['shards']:
        with np.load(PROJECT/shard['path']) as z:
            x=z['x_adversarial_native'];positions=z['mixed_position'];r=records[positions]
            assert np.array_equal(z['labels'],r['label'])
            for start in range(0,len(r),batch_size):
                sl=slice(start,start+batch_size)
                yield x[sl],r['label'][sl].copy(),r['dataset_id'][sl].copy(),r['source_row'][sl].copy(),np.ones(len(r[sl]),bool)

if __name__=='__main__':build()
