"""Small explicit CUDA preflight; does not train or overwrite a defense."""
import os
import tempfile
from pathlib import Path
import cupy as cp
import xgboost as xgb
import numpy as np
os.environ['EDGE_IDS_DEVICE']='cuda'
from train_adversarial_gb import Iterator,PROJECT,digest

pool=cp.cuda.MemoryAsyncPool();cp.cuda.set_allocator(pool.malloc)
class Limited(Iterator):
    def reset(self):super().reset();self.count=0
    def next(self,input_data):
        if self.count>=2:return False
        ok=super().next(input_data);self.count+=int(ok);return ok

with tempfile.TemporaryDirectory() as tmp:
    it=Limited('FGSM',Path(tmp),digest(PROJECT/'artifacts/augmented_training/FGSM/manifest.json'),matrix_mode='memory')
    matrix=xgb.QuantileDMatrix(it,max_bin=128,nthread=4)
    assert matrix.num_row()==32768 and matrix.num_col()==157
    params=dict(objective='binary:logistic',device='cuda',tree_method='hist',max_depth=8,max_bin=128,seed=42,nthread=4)
    first=xgb.train(params,matrix,num_boost_round=2)
    checkpoint=Path(tmp)/'smoke.ubj';first.save_model(checkpoint)
    resumed=xgb.train(params,matrix,num_boost_round=1,xgb_model=str(checkpoint))
    assert resumed.num_boosted_rounds()==3
    assert '"device":"cuda:' in resumed.save_config(),'Unexpected CPU fallback'
    p=resumed.predict(matrix);assert np.isfinite(p).all()
    print('CUDA iterator, quantile matrix, UBJ save/load and continuation passed on 32768 preflight rows.')
    print('This is not a full-data memory or performance benchmark.')
