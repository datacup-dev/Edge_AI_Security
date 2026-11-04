"""Audit all 44 new model/test evaluations and validation-only overall choice."""
import json
from pathlib import Path
import numpy as np
from extended_registry import PROJECT,candidates,choose_overall,NEW_NEURAL,NEW_CLASSICAL


def main():
    completed,pending,old=candidates()
    if pending:raise RuntimeError(f'Cannot audit incomplete candidates: {pending}')
    result=[];labels={}
    for name in ['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']:
        with np.load(PROJECT/'artifacts/native_splits'/f'{name}.npz') as data:
            rows=np.flatnonzero(data['split']==2);labels[name]=(rows,data['labels'][rows])
    for candidate in completed:
        model=candidate['model']
        for name,(expected_rows,expected_y) in labels.items():
            directory=PROJECT/candidate['test_directory']/name
            metrics=json.loads((directory/'metrics.json').read_text())
            source_rows=np.load(directory/'source_rows.npy');assert np.array_equal(source_rows,expected_rows)
            with np.load(directory/'predictions.npz') as data:
                y,p,predicted=(data[k] for k in ['labels','probability','predicted'])
                assert np.array_equal(y,expected_y)
                assert y.shape==p.shape==predicted.shape
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                assert np.array_equal(predicted,p>=.5)
                count=np.bincount(y.astype(np.uint8)*2+predicted,minlength=4)
                assert count.tolist()==[metrics[k] for k in ['tn','fp','fn','tp']]
            assert sum(count)==len(expected_rows)==metrics['n']
            assert np.isclose((count[0]+count[3])/len(expected_rows),metrics['accuracy'])
            assert (directory/'confusion_matrix.png').exists() and (directory/'roc_pr.png').exists()
            result.append(dict(model=model,dataset=name,rows=len(expected_rows),status='passed'))
        config=json.loads((PROJECT/'artifacts'/candidate['family']/model/'configuration.json').read_text()) if (PROJECT/'artifacts'/candidate['family']/model/'configuration.json').exists() else None
        if 'training_rows' in candidate:assert candidate['training_rows']==35030746
        history_path=PROJECT/'artifacts'/candidate['family']/model/'history.json'
        if history_path.exists() and model!='GB':
            history=json.loads(history_path.read_text())
            assert all(r['training_rows']==35030746 for r in history)
            pair=max((r['mean_validation_macro_f1'],r['mean_validation_ap']) for r in history)
            assert pair==(candidate['validation_macro_f1'],candidate['validation_ap'])
        if 'validation' in candidate:
            assert np.isclose(np.mean([v['macro_f1'] for v in candidate['validation'].values()]),candidate['validation_macro_f1'])
        if model=='KNN':
            metadata=json.loads((PROJECT/candidate['checkpoint']).read_text())
            assert metadata['rows']==35030746 and metadata['eps']==0 and metadata['k']==5
            verified_rows=0
            for code,record in enumerate(metadata['domains']):
                root=PROJECT/'artifacts/encoded_inputs'/record['name']
                original=np.load(root/'features.npy',mmap_mode='r');original_labels=np.load(root/'labels.npy')
                train_rows=np.flatnonzero(np.load(root/'split.npy')==0)
                reference=(PROJECT/candidate['checkpoint']).parent
                points=np.load(reference/f'{code}_points.npy',mmap_mode='r')
                point_labels=np.load(reference/f'{code}_labels.npy',mmap_mode='r')
                assert len(points)==record['rows']==len(train_rows)
                assert np.array_equal(point_labels,original_labels[train_rows])
                for start in range(0,len(train_rows),100_000):
                    selected=train_rows[start:start+100_000]
                    block=np.asarray(original[selected],dtype=np.float64)
                    expected_points=block[:,record['active']]
                    if record.get('rotation') is not None:
                        rotation=np.asarray(record['rotation']);center=np.asarray(record['center'])
                        assert np.allclose(rotation.T@rotation,np.eye(len(rotation)),atol=1e-12)
                        expected_points=(expected_points-center)@rotation
                        assert np.allclose(points[start:start+len(selected)],expected_points,rtol=1e-12,atol=1e-10)
                    else:assert np.array_equal(points[start:start+len(selected)],expected_points)
                    if record['fixed']:assert np.all(block[:,record['fixed']]==np.asarray(record['constants']))
                    verified_rows+=len(selected)
                other=np.asarray(record['other_indices'])
                vectors=np.asarray(original[train_rows[other]],dtype=np.float64)
                assert np.allclose(np.einsum('ij,ij->i',vectors,vectors),record['other_norms'])
                assert np.array_equal(original_labels[train_rows[other]],record['other_labels'])
                del original,original_labels,points,point_labels
            assert verified_rows==35030746
    assert len(result)==44
    chosen=choose_overall();assert chosen['selection_uses_test'] is False
    if (PROJECT/'artifacts/overall_baseline.json').exists():
        saved=json.loads((PROJECT/'artifacts/overall_baseline.json').read_text())
        assert chosen['selected']['model']==saved['selected']['model']
    report=dict(status='passed',new_models=11,new_model_test_evaluations=44,evaluations=result,
                selection_uses_test=False,overall_model=chosen['selected']['model'],
                scope='Membership, labels, threshold, confusion counts, epoch totals, score aggregation and selection audit; not re-inference of every test row')
    directory=PROJECT/'artifacts/extended_report';directory.mkdir(exist_ok=True)
    (directory/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    chosen['status']='verified_complete';chosen['verification']='artifacts/extended_report/verification.json'
    (PROJECT/'artifacts/overall_baseline.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    print('Verified all 44 new model/test evaluations; overall baseline:',chosen['selected']['model'],flush=True)


if __name__=='__main__':main()
