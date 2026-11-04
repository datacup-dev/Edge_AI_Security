"""Regression: memory/external quantile iterators and saved-tree continuation.

Uses small synthetic native-schema batches; does not touch real checkpoints.
"""
import gc
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import xgboost as xgb
import train_adversarial_gb as training

def main():
    rng=np.random.default_rng(42)
    codes=np.repeat(np.arange(4,dtype=np.uint8),100)
    values=np.full((400,64),np.nan,np.float32)
    for c,w in enumerate([64,25,39,25]):values[codes==c,:w]=rng.normal(size=(100,w))
    labels=(values[:,0]>0).astype(np.uint8)
    def batches(attack,batch_size):
        for start in [0,200]:
            sl=slice(start,start+200)
            yield values[sl],labels[sl],codes[sl],np.arange(start,start+200),np.zeros(200,bool)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);manifest=root/'artifacts/augmented_training/FGSM/manifest.json'
        manifest.parent.mkdir(parents=True);manifest.write_text('{}')
        with patch.object(training,'PROJECT',root),patch.object(training,'batches',batches),patch.dict('os.environ',{'EDGE_IDS_DEVICE':'cpu'}):
            for mode in ['memory','external']:
                cache=root/mode;cache.mkdir()
                it=training.Iterator('FGSM',cache,training.digest(manifest),matrix_mode=mode)
                matrix_class=xgb.QuantileDMatrix if mode=='memory' else xgb.ExtMemQuantileDMatrix
                matrix=matrix_class(it,max_bin=128,nthread=2)
                assert matrix.num_row()==400 and matrix.num_col()==157
                params=dict(objective='binary:logistic',tree_method='hist',device='cpu',max_bin=128,max_depth=2,nthread=2)
                model=xgb.train(params,matrix,num_boost_round=2)
                checkpoint=root/f'{mode}.ubj';model.save_model(checkpoint)
                resumed=xgb.train(params,matrix,num_boost_round=1,xgb_model=str(checkpoint))
                assert resumed.num_boosted_rounds()==3
                assert np.isfinite(resumed.predict(matrix)).all()
                del resumed,model,matrix,it;gc.collect()
                print(mode,'quantile input and checkpoint continuation passed',flush=True)

if __name__=='__main__':main()
