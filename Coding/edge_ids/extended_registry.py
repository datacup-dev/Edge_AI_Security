"""Validation-only candidate registry; no test metrics participate in selection."""
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parent
NEW_NEURAL=['Autoencoder','GNN','Transformer','GAN']
NEW_CLASSICAL=['LR','KNN','DT','RF','SVM','NB','GB']


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):return None


def saved_bytes(model,checkpoint):
    if model=='KNN':
        parent=checkpoint.parent
        return sum(p.stat().st_size for p in parent.glob('*.npy'))+checkpoint.stat().st_size
    return checkpoint.stat().st_size


def candidates():
    completed=[];pending=[]
    for family,models in [('expanded_neural',NEW_NEURAL),('expanded_classical',NEW_CLASSICAL)]:
        for model in models:
            directory=PROJECT/'artifacts'/family/model
            selected=read_json(directory/'selection.json')
            if not (directory/'complete.json').exists() or selected is None:
                pending.append(model);continue
            checkpoint=PROJECT/selected['checkpoint']
            entry=dict(selected,family=family,model=model,artifact_bytes=saved_bytes(model,checkpoint),
                       status='training and four test evaluations complete',
                       test_directory=str((directory/'test').relative_to(PROJECT)).replace('\\','/'))
            completed.append(entry)
    old=read_json(PROJECT/'artifacts/adapted_experiment/baseline.json')['selected']
    checkpoint=PROJECT/'artifacts/adapted_experiment/MLP/best.pt'
    incumbent=dict(old,family='adapted_experiment',checkpoint=str(checkpoint.relative_to(PROJECT)).replace('\\','/'),
                   artifact_bytes=checkpoint.stat().st_size,test_directory='artifacts/adapted_experiment/MLP/test')
    return completed,pending,incumbent


def choose_overall():
    completed,pending,incumbent=candidates()
    if pending:return dict(status='incomplete',pending=pending,completed_candidates=completed,incumbent=incumbent)
    new_best=max(completed,key=lambda c:(c['validation_macro_f1'],c['validation_ap'],-c['artifact_bytes'],c['model']))
    new_pair=(new_best['validation_macro_f1'],new_best['validation_ap'])
    old_pair=(incumbent['validation_macro_f1'],incumbent['validation_ap'])
    winner=new_best if new_pair>old_pair else incumbent
    return dict(status='selected_pending_artifact_audit',selected=winner,best_new=new_best,previous_MLP=incumbent,
                new_candidates=completed,new_candidate_count=11,comparison_candidate_count=12,
                selection_uses_test=False,
                rule='Highest mean validation macro-F1, then mean AP; smaller saved artifact breaks new-candidate ties; retain incumbent MLP on exact validation pair tie',
                scope='Known-domain native-schema adapted pipelines; reused historical test sets, not unseen-domain transfer')
