"""Aggregate expanded results without using test metrics for baseline choice."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extended_registry import PROJECT,choose_overall,read_json,NEW_NEURAL,NEW_CLASSICAL


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def graph_figure(path):
    from expanded_models import GRAPH_EDGES
    from adapted_models import GROUPS
    positions=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])
    fig,axis=plt.subplots(figsize=(8,7))
    for a,b in GRAPH_EDGES:
        axis.plot(positions[[a,b],0],positions[[a,b],1],color='gray',alpha=.6,zorder=1)
    axis.scatter(positions[:,0],positions[:,1],s=1000,c=np.arange(8),cmap='tab10',zorder=2)
    for i,(x,y) in enumerate(positions):
        axis.text(x,y,str(i),ha='center',va='center',color='white',weight='bold')
        axis.text(x*1.32,y*1.32,GROUPS[i].replace('_','\n'),ha='center',va='center',fontsize=9)
    axis.set(xlim=(-1.8,1.8),ylim=(-1.6,1.6),title='GNN feature-relation prior: not a network topology')
    axis.axis('off');fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    output=PROJECT/'artifacts/extended_report';output.mkdir(exist_ok=True)
    # This function reads selection records and artifact sizes, never test metrics.
    ranking=choose_overall()
    if ranking['status']!='incomplete':
        pointer=PROJECT/'artifacts/overall_baseline.json'
        prior=read_json(pointer)
        if prior and prior.get('status')=='verified_complete' and prior['selected']['model']==ranking['selected']['model']:
            ranking['status']='verified_complete'
            ranking['verification']=prior.get('verification','artifacts/extended_report/verification.json')
        pointer.write_text(json.dumps(ranking,indent=2),encoding='utf-8')
        candidates=ranking['new_candidates']+[ranking['previous_MLP']]
        pending=[]
    else:
        candidates=ranking['completed_candidates']+[ranking['incumbent']]
        pending=ranking['pending']
    (output/'selection_status.json').write_text(json.dumps(ranking,indent=2),encoding='utf-8')
    ordered=sorted(candidates,key=lambda r:(r['validation_macro_f1'],r['validation_ap']),reverse=True)
    write_csv(output/'validation_comparison.csv',[{k:c.get(k) for k in ['model','family','validation_macro_f1','validation_ap','artifact_bytes','parameters','epochs_run','fit_seconds']} for c in ordered])
    names=['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID'];rows=[]
    for c in ordered:
        for name in names:
            path=PROJECT/c['test_directory']/name/'metrics.json';metric=read_json(path)
            if metric:rows.append(dict(model=c['model'],dataset=name,**metric))
    write_csv(output/'all_test_metrics.csv',rows)
    if rows:
        values=np.array([[next((r['f1_macro'] for r in rows if r['model']==c['model'] and r['dataset']==n),np.nan) for n in names] for c in ordered])
        fig,axis=plt.subplots(figsize=(10,max(4,len(ordered)*.45)))
        image=axis.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i in range(len(ordered)):
            for j in range(4):axis.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='white' if values[i,j]>.7 else 'black',fontsize=9)
        axis.set(xticks=range(4),xticklabels=names,yticks=range(len(ordered)),yticklabels=[c['model'] for c in ordered],
                 title='Separate test macro-F1; row order uses validation only')
        fig.colorbar(image,ax=axis,label='Macro-F1');fig.tight_layout();fig.savefig(output/'test_macro_f1.png',dpi=160);plt.close(fig)
    graph_figure(output/'gnn_feature_graph.png')
    status='Pending: '+', '.join(pending) if pending else 'All 11 new candidates completed; baseline selection uses validation only. Artifact status: '+ranking['status']+'.'
    lines=['# Extended mixed-domain model comparison','',status,'',
           'All models use the original four native train/validation/test memberships. The real training population has 35,030,746 rows; natural domain proportions are preserved. This extends the authorized dataset-specific-encoder study, not the paper\'s common-flow representation.','',
           'User choices: exact KNN and a degree-2 polynomial kernel SVM. KNN has no approximate-search setting; the kernel map has no Nystroem/random-feature approximation. The SVM uses finite stochastic primal optimization, not a global-optimum certificate.','',
           '[Protocol and planned settings](EXTENDED_EXPERIMENT_PLAN.md). [Full expanded methods and parameters](EXTENDED_EXPERIMENT_METHOD.md). [All recorded test metrics](artifacts/extended_report/all_test_metrics.csv). [Validation comparison](artifacts/extended_report/validation_comparison.csv).','',
           '## Validation ranking','',
           '| Model | Mean validation macro-F1 | Mean validation AP | Checkpoint/reference bytes |','|---|---:|---:|---:|']
    for c in ordered:lines.append(f"| {c['model']} | {c['validation_macro_f1']:.6f} | {c['validation_ap']:.6f} | {c['artifact_bytes']:,} |")
    if not pending:
        lines+=['',f"Best new candidate: **{ranking['best_new']['model']}**. Overall baseline compared with the prior MLP: **{ranking['selected']['model']}**.",'',
                'The old MLP checkpoint/results are preserved. Overall selection compares validation pairs; new test results do not select the winner. These test datasets were already inspected in the prior stage, so this is a follow-on benchmark on reused holdouts.','']
    lines+=['','## Separate held-out test results','','| Model | Dataset | Macro-F1 | Accuracy | Attack recall | FPR | ROC-AUC | AP |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['dataset']} | {r['f1_macro']:.6f} | {r['accuracy']:.6f} | {r['recall_attack']:.6f} | {r['false_positive_rate']:.6f} | {r['roc_auc']:.6f} | {r['average_precision']:.6f} |")
    lines+=['','![Test comparison](artifacts/extended_report/test_macro_f1.png)','','## Architecture meaning and limitations','',
            '- Autoencoder: supervised bottleneck classifier with a weighted native-feature reconstruction objective; not benign-only anomaly detection.',
            '- GNN: message passing over eight feature-group nodes and a predefined relation graph; no inferred device or flow topology.',
            '- Transformer: attention over feature groups, not a chronological packet sequence.',
            '- GAN: conditional generator and benign/attack/fake discriminator. Binary classification conditions on the real classes; generated vectors are not adversarial network attacks.',
            '- LR and polynomial SVM: full native-coordinate models, optimized with Adam and validation early stopping. Finite iterations do not certify optimum convergence.',
            '- DT/RF: sparse zero-coded native namespace. GB: unavailable blocks are missing, but observed native zero values remain explicit. These are comparisons of complete pipelines, not a pure classifier-only ablation.',
            '- NB: Gaussian independence model over full native blocks; weak performance is retained rather than omitted. Its implementation matched a dense sklearn reference in synthetic checks.',
            '- SVM sigmoid scores are uncalibrated. ROC/AP use saved raw margins; log loss/Brier use the stated probability link. Numerical probability precision and clipping can differ by pipeline.',
            '- KNN keeps the full reference population. Constant-coordinate removal has an exact query-distance offset; it is not a PCA or sample reduction.',
            '- Random row splits do not isolate duplicate records, captures, devices or time. Native observation units differ. No unseen-domain transfer, adversarial robustness or end-to-end edge timing is established.','',
            '![Feature graph](artifacts/extended_report/gnn_feature_graph.png)','','## All confusion and ROC/PR figures','','| Model | Dataset | Confusion counts and normalized rates | ROC/PR |','|---|---|---|---|']
    for c in ordered:
        for n in names:
            folder=f"{c['test_directory']}/{n}"
            if (PROJECT/folder/'metrics.json').exists():lines.append(f"| {c['model']} | {n} | [Confusion]({folder}/confusion_matrix.png) | [ROC/PR]({folder}/roc_pr.png) |")
    lines+=['','## Training histories and reproducibility','','[Detailed settings and methods](EXTENDED_EXPERIMENT_METHOD.md). [Executed preview notebook](preview.ipynb).','']
    for c in ordered:
        folder=f"artifacts/{c['family']}/{c['model']}"
        curve=PROJECT/folder/'learning_curves.png'
        if curve.exists():lines.append(f"- {c['model']}: [learning curves]({folder}/learning_curves.png), [recorded history]({folder}/history.json).")
    for file,label in [('verification.json','Full artifact audit'),('loader_verification.json','Fresh loader consistency check')]:
        if (output/file).exists():lines+=['',f'[{label}](artifacts/extended_report/{file}).']
    if (PROJECT/'artifacts/expanded_classical/KNN/worker_benchmark.json').exists():
        lines+=['','[KNN CPU worker trial](artifacts/expanded_classical/KNN/worker_benchmark.json). [Index-layout decision](artifacts/knn_index_development/index_layout_decision.json).']
    (PROJECT/'EXTENDED_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    current=['## Expanded comparison','',status,'',
             '[Expanded report](EXTENDED_EXPERIMENT_REPORT.md). [Methods and parameters](EXTENDED_EXPERIMENT_METHOD.md).',
             f"Completed new candidates: {len(candidates)-1}/11; recorded new test evaluations: {sum(r['model']!='MLP' for r in rows)}/44.",'']
    if pending:
        current+=['The overall baseline remains pending. Live progress: `artifacts/extended_report/remaining_progress.json` and per-worker progress JSON.','']
    else:
        current += [f"Best new model: **{ranking['best_new']['model']}**. Overall baseline: **{ranking['selected']['model']}**.",
                    f"Overall mean validation macro-F1: {ranking['selected']['validation_macro_f1']:.6f}; prior MLP: {ranking['previous_MLP']['validation_macro_f1']:.6f}.",
                    'Selection uses validation only. Separate four-dataset test metrics and confusion/ROC figures are linked in the report.','']
    current+=['SVM uses the user-selected exact degree-2 polynomial kernel; KNN retains the full training reference population with exact search. The sections below preserve the initial five-model phase.','']
    run_status=PROJECT/'RUN_STATUS.md'
    if run_status.exists():
        text=run_status.read_text(encoding='utf-8')
        text=re.sub(r'## Expanded comparison(?: in progress)?\n.*?(?=Full audit trail)', '\n'.join(current)+'\n',text,flags=re.S)
        run_status.write_text(text,encoding='utf-8')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROJECT.glob('*.py')}
    (output/'source_code_sha256.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    print('Extended report refreshed:',len(candidates)-1,'new completed candidates;',len(rows),'model/test evaluations; pending',pending,flush=True)


if __name__=='__main__':main()
