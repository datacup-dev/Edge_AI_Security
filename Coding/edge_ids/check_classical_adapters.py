"""Independent dense-reference checks for native sparse adapters and Gaussian NB."""
import json
from pathlib import Path
import numpy as np
from sklearn.naive_bayes import GaussianNB
from native_gaussian_nb import NativeGaussianNB
from classical_data import layout,sparse_batch
from train_gradient_boosting import NativeIterator
import xgboost as xgb


def main():
    rng=np.random.default_rng(42);domains=[]
    for code,f in enumerate([3,2,4,3]):
        x=rng.normal(size=(32,f)).astype(np.float32)
        y=np.tile(np.array([0,1],np.uint8),16)
        domains.append(dict(name=f'synthetic_{code}',x=x,y=y,indices=[np.arange(32)]))
    widths,offsets,dimension=layout(domains)
    blocks=[]
    for code,d in enumerate(domains):
        dense=np.zeros((32,dimension));dense[:,offsets[code]:offsets[code+1]]=d['x'];dense[:,offsets[-1]+code]=1
        assert np.array_equal(sparse_batch(d['x'],code,offsets,dimension).toarray(),dense)
        blocks.append(dense)
    full=np.vstack(blocks);labels=np.concatenate([d['y'] for d in domains])
    reference=GaussianNB(var_smoothing=1e-9).fit(full,labels)
    native=NativeGaussianNB().fit(domains);error=0.
    for code,d in enumerate(domains):
        expected=reference.predict_log_proba(blocks[code]);margin=expected[:,1]-expected[:,0]
        result=native.decision_function(d['x'],code)
        error=max(error,float(np.max(np.abs(result-margin))))
        assert np.allclose(result,margin,rtol=1e-9,atol=1e-9)
    quantile=xgb.QuantileDMatrix(NativeIterator(domains,batch_size=17),max_bin=16,nthread=2)
    assert quantile.num_row()==128 and quantile.num_col()==dimension
    booster=xgb.train(dict(objective='binary:logistic',tree_method='hist',device='cpu',max_bin=16,nthread=2),quantile,num_boost_round=2)
    assert np.isfinite(booster.predict(quantile)).all()
    result=dict(status='passed',sparse_adapter='Matches independently materialized dense native namespace',
                GaussianNB='Matches sklearn GaussianNB on dense reference',max_log_odds_error=error,
                XGBoost='Streaming quantile iterator processed every synthetic source row; training/prediction API passed',
                scope='Synthetic checks only, not subset results or final dataset metrics')
    target=Path(__file__).resolve().parent/'artifacts/expanded_classical/adapter_preflight.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
