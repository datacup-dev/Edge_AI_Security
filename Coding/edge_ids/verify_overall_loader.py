"""Check the audited overall deployment loader against saved test predictions."""
import json
import time
from pathlib import Path
import numpy as np
from overall_baseline import load_overall_baseline,NAMES

PROJECT=Path(__file__).resolve().parent


def main():
    started=time.perf_counter();baseline=load_overall_baseline()
    load_seconds=time.perf_counter()-started;results=[]
    generator=np.random.default_rng(42)
    for name in NAMES:
        folder=PROJECT/baseline.selected['test_directory']/name
        rows=np.load(folder/'source_rows.npy',mmap_mode='r')
        positions=np.unique(np.r_[np.arange(min(16,len(rows))),generator.choice(len(rows),32,replace=False)])
        features=np.load(PROJECT/'artifacts/encoded_inputs'/name/'features.npy',mmap_mode='r')
        x=np.asarray(features[rows[positions]])
        with np.load(folder/'predictions.npz') as data:expected=data['probability'][positions]
        started=time.perf_counter();actual=baseline.predict_probability(x,name)
        seconds=time.perf_counter()-started
        tolerance=.002 if baseline.kind in ['MLP','Autoencoder','GNN','Transformer','GAN'] else 1e-5
        assert np.allclose(actual,expected,rtol=0,atol=tolerance),(name,float(np.max(np.abs(actual-expected))))
        assert np.array_equal(actual>=.5,expected>=.5),(name,'Threshold decisions differ')
        results.append(dict(dataset=name,sample_rows=rows[positions].tolist(),rows=len(x),
            maximum_probability_difference=float(np.max(np.abs(actual-expected))),
            absolute_tolerance=tolerance,prediction_seconds=seconds,status='passed'))
    report=dict(status='passed',model=baseline.kind,load_seconds=load_seconds,domains=results,
        scope='Sampled loader consistency and cold-call workstation timing; excludes extraction, preprocessing and window delay; not an edge real-time claim')
    (PROJECT/'artifacts/extended_report/loader_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Overall baseline loader matched saved test samples in all four domains:',baseline.kind,flush=True)


if __name__=='__main__':main()
