"""Full-data Gaussian NB, decision tree, random forest and exact KNN."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import joblib
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,average_precision_score
from threadpoolctl import threadpool_limits
from classical_data import layout,sparse_batch,load_csc
from native_gaussian_nb import NativeGaussianNB
from exact_classical_models import ExactBlockKNN
from train_adapted import load_domains,save_json
from extended_evaluation import margin_probability,export_evaluation

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts/expanded_classical'


def predict(model,kind,d,code,rows,offsets,dimension,directory,split_name):
    cache=directory/'prediction_cache';cache.mkdir(exist_ok=True)
    result_file=cache/f'{split_name}_{code}.npy';progress_file=cache/f'{split_name}_{code}.json'
    if result_file.exists() and progress_file.exists():
        result=np.load(result_file,mmap_mode='r+')
        position=json.loads(progress_file.read_text())['next_row']
        assert result.shape==(len(rows),)
    else:
        result=np.lib.format.open_memmap(result_file,mode='w+',dtype=np.float64,shape=(len(rows),));position=0
    started=time.perf_counter()
    for start in range(position,len(rows),8192):
        selected=rows[start:start+8192];x=np.asarray(d['x'][selected])
        if kind=='NB':value=model.decision_function(x,code)
        elif kind=='KNN':value=model.predict_probability(x,code)
        else:value=model.predict_proba(sparse_batch(x,code,offsets,dimension))[:,1]
        result[start:start+len(selected)]=value
        if kind=='KNN' or start//8192%100==0:
            result.flush();save_json(progress_file,dict(next_row=start+len(selected)))
            save_json(OUT/'cpu_progress.json',dict(stage=f'{split_name}_prediction',model=kind,dataset=d['name'],
                      rows_processed=start+len(selected),rows_total=len(rows),elapsed_seconds=time.perf_counter()-started))
    result.flush();save_json(progress_file,dict(next_row=len(rows)))
    return np.asarray(result)


def main():
    OUT.mkdir(exist_ok=True);domains=load_domains();widths,offsets,dimension=layout(domains)
    n=sum(len(d['indices'][0]) for d in domains)
    configs={
        'NB':dict(variant='Gaussian NB on complete native blocks',var_smoothing=1e-9,accumulator_dtype='float64'),
        'DT':dict(variant='sklearn DecisionTreeClassifier',max_depth=16,min_samples_leaf=50,max_features='sqrt',random_state=42),
        'RF':dict(variant='sklearn RandomForestClassifier',n_estimators=64,max_depth=12,min_samples_leaf=50,
                  max_features='sqrt',bootstrap=True,max_samples=None,n_jobs=2,random_state=42),
        'KNN':dict(variant='Exact pooled Euclidean KNN via exact block decomposition',n_neighbors=5,weights='uniform',
                   eps=0.,leafsize=128,workers=4,approximate=False,dimension=dimension,
                   search_layout='Native-coordinate exact KD-trees; only constant coordinates removed with exact distance offset')}
    for kind in ['NB','DT','RF','KNN']:
        directory=OUT/kind;directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        config=dict(model=kind,training_rows=n,feature_dimension=dimension,dimension_reduction=False,
                    seed=42,representation='Full native blocks plus four one-hot domain markers',
                    validation='Mean four-dataset macro-F1; mean AP tie-break',threshold=.5,**configs[kind])
        save_json(directory/'configuration.json',config)
        save_json(OUT/'cpu_progress.json',dict(stage='fitting',model=kind,training_rows=n))
        started=time.perf_counter();model_path=directory/('model.npz' if kind=='NB' else 'model.joblib')
        if kind=='NB':
            model=NativeGaussianNB.load(model_path) if model_path.exists() else NativeGaussianNB().fit(domains)
            if not model_path.exists():model.save(model_path)
        elif kind=='KNN':
            model_path=directory/'reference_data/model.json'
            # Restore the saved basis together with its reference vectors. Re-fitting
            # a rotation while resuming prediction caches could change the index.
            if model_path.exists():
                model=ExactBlockKNN.load(directory/'reference_data')
            else:
                model=ExactBlockKNN(directory/'reference_data').fit(domains)
            benchmark_path=directory/'worker_benchmark.json'
            if not benchmark_path.exists():
                # Compare CPU parallelism on identical, unlabelled query features.
                # This changes scheduling only, never k, eps, or the reference set.
                d=domains[2];cursor_path=directory/'prediction_cache/validation_2.json'
                cursor=json.loads(cursor_path.read_text())['next_row'] if cursor_path.exists() else 0
                rows=d['indices'][1][cursor:cursor+8192:64]
                if not len(rows):rows=d['indices'][1][:8192:64]
                queries=np.asarray(d['x'][rows]);timings=[];reference=None
                for workers in [4,8,16]:
                    model.workers=workers;begin=time.perf_counter()
                    answer=model.predict_probability(queries,2,return_neighbors=True)
                    seconds=time.perf_counter()-begin
                    if reference is None:reference=answer
                    else:
                        assert np.array_equal(answer[0],reference[0])
                        assert np.array_equal(answer[1],reference[1])
                        assert np.array_equal(answer[2],reference[2])
                    timings.append(dict(workers=workers,seconds=seconds))
                    print('Exact KNN parallelism benchmark',timings[-1],flush=True)
                model.workers=min(timings,key=lambda r:r['seconds'])['workers']
                save_json(benchmark_path,dict(query_rows=rows.tolist(),dataset=d['name'],
                    timings=timings,selected_workers=model.workers,identical_neighbors_and_votes=True,
                    labels_used=False,note='One feature-only latency trial per worker count; not an accuracy experiment'))
                metadata=json.loads(model_path.read_text());metadata['workers']=model.workers
                save_json(model_path,metadata)
            config['workers']=model.workers;save_json(directory/'configuration.json',config)
        elif model_path.exists():model=joblib.load(model_path)
        else:
            matrix,labels,info=load_csc();assert matrix.shape==(n,dimension)
            if kind=='DT':
                model=DecisionTreeClassifier(max_depth=16,min_samples_leaf=50,max_features='sqrt',random_state=42)
                model.fit(matrix,labels);joblib.dump(model,model_path)
            else:
                model=RandomForestClassifier(n_estimators=8,max_depth=12,min_samples_leaf=50,max_features='sqrt',
                    bootstrap=True,max_samples=None,n_jobs=2,random_state=42,warm_start=True)
                for trees in range(8,65,8):
                    model.set_params(n_estimators=trees);model.fit(matrix,labels)
                    save_json(OUT/'cpu_progress.json',dict(stage='fitting',model=kind,trees_completed=trees,trees_total=64,
                              elapsed_seconds=time.perf_counter()-started))
                    joblib.dump(model,directory/'forest_partial.joblib')
                joblib.dump(model,model_path)
            del matrix,labels;gc.collect()
        fit_seconds=time.perf_counter()-started
        validation={}
        for code,d in enumerate(domains):
            rows=d['indices'][1];score=predict(model,kind,d,code,rows,offsets,dimension,directory,'validation')
            p=margin_probability(score) if kind=='NB' else score
            validation[d['name']]=dict(macro_f1=float(f1_score(d['y'][rows],p>=.5,average='macro',zero_division=0)),
                                      average_precision=float(average_precision_score(d['y'][rows],score)))
        selection=dict(model=kind,validation_macro_f1=float(np.mean([v['macro_f1'] for v in validation.values()])),
                       validation_ap=float(np.mean([v['average_precision'] for v in validation.values()])),
                       validation=validation,training_rows=n,fit_seconds=fit_seconds,
                       artifact_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file() and 'prediction_cache' not in str(p)),
                       checkpoint=str(model_path.relative_to(PROJECT)))
        save_json(directory/'selection.json',selection)
        print('Frozen classical candidate',selection,flush=True)
        for code,d in enumerate(domains):
            target=directory/'test'/d['name']
            if (target/'source_rows.npy').exists() and (target/'roc_pr.png').exists():continue
            rows=d['indices'][2];score=predict(model,kind,d,code,rows,offsets,dimension,directory,'test')
            p=margin_probability(score) if kind=='NB' else score
            export_evaluation(d['y'][rows],p,rows,target,f'{kind} — {d["name"]}',raw_score=score if kind=='NB' else None,
                              probability_note='Gaussian model posterior' if kind=='NB' else None)
        save_json(directory/'complete.json',dict(model=kind,test_datasets=4))
        del model;gc.collect()
    save_json(OUT/'cpu_progress.json',dict(stage='complete',models=['NB','DT','RF','KNN'],test_evaluations=16))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
