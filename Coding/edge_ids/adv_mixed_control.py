"""Report, supervise and preview fifteen independent both-class attack datasets."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adv_mixed_data import PROJECT,ROOT,COUNT,read,save,replace
from adv_mixed_algorithms import METHODS,PARAMETERS
from adv_mixed_run import LANES


def report():
    rows=[];saved=complete=0
    for method in METHODS:
        directory=ROOT/method;cursor=directory/'cursor.json'
        state=read(cursor) if cursor.exists() else None;done=state['done'] if state else 0;saved+=done
        complete+=int((directory/'complete.json').exists())
        for label,total in [(0,2290435),(1,32740311)]:
            values=state['by_class'][str(label)] if state else dict(done=0,eligible=0,success=0,changed=0,queries=0,capped=0,initialization_failed=0,l2_sum=0.,linf_sum=0.)
            rows.append(dict(method=method,source_label=label,target_label=1-label,generated=values['done'],required=total,
                eligible=values['eligible'],successful=values['success'],success_rate=values['success']/values['eligible'] if values['eligible'] else None,
                changed=values['changed'],queries=values['queries'],query_cap_exhausted=values['capped'],initialization_failed=values['initialization_failed'],
                mean_l2=values['l2_sum']/values['done'] if values['done'] else None,mean_linf=values['linf_sum']/values['done'] if values['done'] else None))
    with (ROOT/'progress.tmp.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    replace(ROOT/'progress.tmp.csv',ROOT/'progress.csv')
    failures={p.stem:read(p) for p in (ROOT/'workers').glob('*failure.json')} if (ROOT/'workers').exists() else {}
    preparation=read(ROOT/'preparation_progress.json') if (ROOT/'preparation_progress.json').exists() else {'stage':'not_started'}
    status=dict(status='failed_worker' if failures else 'complete' if complete==15 else 'running',preparation_stage=preparation['stage'],generated_records=saved,required_records=COUNT*15,
        complete_methods=complete,total_methods=15,rows_per_method=COUNT,benign_per_method=2290435,attack_per_method=32740311,
        both_classes=True,training_only=True,heldout_outputs=False,failures=failures)
    save(ROOT/'status.json',status);save(ROOT/'attack_parameters.json',PARAMETERS)
    fig,axes=plt.subplots(1,2,figsize=(13,6));positions=np.arange(len(METHODS));coverage=[]
    for method in METHODS:coverage.append(sum(r['generated'] for r in rows if r['method']==method)/COUNT)
    axes[0].barh(METHODS,coverage);axes[0].set(xlim=(0,1),title='Full mixed-training coverage',xlabel='Fraction processed')
    for label,offset,color in [(0,-.18,'tab:blue'),(1,.18,'tab:orange')]:
        values=[next(r['success_rate'] or 0 for r in rows if r['method']==m and r['source_label']==label) for m in METHODS]
        axes[1].barh(positions+offset,values,height=.35,color=color,label='benign → attack' if label==0 else 'attack → benign')
    axes[1].set(yticks=positions,yticklabels=METHODS,xlim=(0,1),title='Partial training-only ASR',xlabel='New successful flips / eligible processed');axes[1].legend()
    fig.tight_layout();fig.savefig(ROOT/'progress.tmp.png',dpi=150);plt.close(fig);replace(ROOT/'progress.tmp.png',ROOT/'progress.png')
    prefix='artifacts/mixed_training_attacks/'
    text=['# Separate attacks on all mixed-training records','',f"Status: **{status['status']}**. {saved:,}/{COUNT*15:,} records saved; {complete}/15 attack datasets complete.",'',
        'Every method processes the same 35,030,746 original mixed-training records, including 2,290,435 benign and 32,740,311 attack records. No validation/test records or final subsampling are included. Each method has its own folder; candidates retain their original label and target the opposite class.','',
        '[Full process and settings](MIXED_TRAINING_ATTACKS.md). [Live per-class metrics]('+prefix+'progress.csv). [Status]('+prefix+'status.json).','',
        'Saved records include unsuccessful, unchanged and initially misclassified cases. Success is a new opposite-class GB prediction on a changed record GB originally classified correctly. Partial rates describe different prefixes and cannot rank final attack strength. These are training-set vulnerability results, not held-out robustness.','',
        '![Live coverage and per-class ASR]('+prefix+'progress.png)','','| Attack | Source → target | Saved / requested | New successes / eligible | Seed failures | Query cap exhausted |','|---|---|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['method']} | {r['source_label']} → {r['target_label']} | {r['generated']:,} / {r['required']:,} | {r['successful']:,} / {r['eligible']:,} | {r['initialization_failed']:,} | {r['query_cap_exhausted']:,} |")
    if failures:text+=['','## Worker errors','',json.dumps(failures,indent=2)]
    (PROJECT/'MIXED_TRAINING_ATTACK_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    path=PROJECT/'RUN_STATUS.md';old=path.read_text(encoding='utf-8')
    section='\n'.join(['<!-- both-class-attacks-start -->','## Both-class mixed-training adversarial generation','',
        f"Status: **{status['status']}**; saved {saved:,}/{COUNT*15:,} records; {complete}/15 methods complete.",
        'All original mixed-training records, both classes. Fifteen separate outputs. No held-out outputs.',
        '[Live report](MIXED_TRAINING_ATTACK_REPORT.md). [Process and parameters](MIXED_TRAINING_ATTACKS.md).','<!-- both-class-attacks-end -->',''])
    old=re.sub(r'<!-- both-class-attacks-start -->.*?<!-- both-class-attacks-end -->\n?',lambda _:section,old,flags=re.S) if '<!-- both-class-attacks-start -->' in old else old.replace('# Run status\n','# Run status\n\n'+section,1)
    path.write_text(old,encoding='utf-8');save(ROOT/'source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROJECT.glob('adv_mixed_*.py')})
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()};save(ROOT/'environment_versions.json',versions)
    (PROJECT/'requirements-mixed-attacks-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu128\n'+'\n'.join(f'{n}=={v}' for n,v in sorted(versions.items(),key=lambda p:p[0].lower()))+'\n',encoding='utf-8')
    print('FULL MIXED ATTACKS',saved,'/',COUNT*15,'complete methods',complete,'/15',flush=True)


def start():
    wait_lock=ROOT/'start.lock';fd=os.open(wait_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        while not (ROOT/'GAN_model/complete.json').exists():
            if (ROOT/'preparation_failure.json').exists():raise RuntimeError(read(ROOT/'preparation_failure.json'))
            report();time.sleep(30)
        subprocess.run([sys.executable,'-u',str(PROJECT/'adv_mixed_check.py')],cwd=PROJECT,check=True)
        launch()
    finally:wait_lock.unlink(missing_ok=True)


def launch():
    assert read(ROOT/'preflight.json')['status']=='passed'
    lock=ROOT/'supervisor.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True);children=[]
    try:
        for lane in LANES:
            failure=ROOT/'workers'/f'{lane}_failure.json'
            if failure.exists():failure.replace(logs/f'{lane}_failure_{time.time_ns()}.json')
            stream=(logs/f'{lane}.log').open('a',encoding='utf-8')
            process=subprocess.Popen([sys.executable,'-u',str(PROJECT/'adv_mixed_run.py'),'--lane',lane],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            children.append((lane,process,stream));print('STARTED',lane,'PID',process.pid,flush=True)
        save(ROOT/'supervisor.json',dict(status='running',pid=os.getpid(),workers={lane:p.pid for lane,p,_ in children}))
        while any(p.poll() is None for _,p,_ in children):
            result=subprocess.run([sys.executable,str(PROJECT/'adv_mixed_control.py'),'report'],cwd=PROJECT)
            if result.returncode:print('Report refresh failed; workers continue and report will retry',flush=True)
            time.sleep(30)
        failures={lane:p.returncode for lane,p,_ in children if p.returncode};report()
        save(ROOT/'supervisor.json',dict(status='failed' if failures else 'complete',failures=failures))
        if failures:raise RuntimeError(failures)
        subprocess.run([sys.executable,str(PROJECT/'adv_mixed_audit.py')],cwd=PROJECT,check=True)
        preview()
    finally:
        for _,_,stream in children:stream.close()
        lock.unlink(missing_ok=True)


def preview():
    import nbformat
    from nbclient import NotebookClient
    path=PROJECT/'preview.ipynb';book=nbformat.read(path,as_version=4)
    book.cells=[c for c in book.cells if not c.get('id','').startswith('both-class-attacks-')]
    cells=[nbformat.v4.new_markdown_cell('# Separate attack datasets: both classes of the mixed training set\n\n'
        'Every method processes all 35,030,746 original mixed-training records. Labels are preserved and targets are opposite labels. '
        'Outputs are independent folders, not a pooled adversarial set. '
        '[Process and exact settings](MIXED_TRAINING_ATTACKS.md). [Live results](MIXED_TRAINING_ATTACK_REPORT.md). '
        'Re-execute the following cells for current status. Training-only ASR is not held-out robustness.',id='both-class-attacks-heading'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "attack_project = Path.cwd()\nif not (attack_project / 'artifacts').exists(): attack_project = attack_project / 'Coding' / 'edge_ids'\n"
            "attack_root = attack_project / 'artifacts' / 'mixed_training_attacks'\n"
            "display(json.loads((attack_root / 'status.json').read_text()))\n"
            "display(pd.read_csv(attack_root / 'progress.csv'))\n"
            "display(Image(filename=str(attack_root / 'progress.png')))\n"
            "for check in ['partial_audit.json','full_audit.json']:\n"
            "    if (attack_root / check).exists(): display(json.loads((attack_root / check).read_text()))",id='both-class-attacks-progress'),
        nbformat.v4.new_code_cell("chosen_attack = 'FGSM'\n"
            "files = sorted((attack_root / chosen_attack / 'shards').glob('*.npz'))\n"
            "mixed_records = np.load(attack_project / 'artifacts' / 'mixed_training' / 'records.npy',mmap_mode='r')\n"
            "if files:\n"
            "    with np.load(files[0]) as z:\n"
            "        display(pd.Series(z['labels']).value_counts().rename(index={0:'benign',1:'attack'}).rename('Both-class shard'))\n"
            "        for truth in [0,1]:\n"
            "            choices = np.flatnonzero((z['labels']==truth)&z['success'])\n"
            "            if not len(choices): choices = np.flatnonzero(z['labels']==truth)\n"
            "            if not len(choices): continue\n"
            "            i=int(choices[0]);position=int(z['mixed_position'][i]);source=mixed_records[position]\n"
            "            display({'source_dataset':int(source['dataset_id']),'source_row':int(source['source_row']), **{k:z[k][i].item() for k in ['mixed_position','labels','target_labels','eligible','changed','success','clean_probability','adversarial_probability','l0','l2','linf','query_counts']}})\n"
            "else:\n    print('First shard pending.')",id='both-class-attacks-examples')]
    fragment=nbformat.v4.new_notebook(cells=cells,metadata=book.metadata.copy())
    NotebookClient(fragment,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(PROJECT)}}).execute()
    book.cells.extend(fragment.cells);nbformat.validate(book);nbformat.write(book,path);print('Both-class attack preview executed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','launch','report','preview']);action=parser.parse_args().action
    {'start':start,'launch':launch,'report':report,'preview':preview}[action]()
