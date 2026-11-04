"""Real both-class training cases and an independent known-boundary check."""
import random
import numpy as np
import torch
import xgboost as xgb
from torch import nn
from classical_data import sparse_batch
from adv_mixed_data import ROOT,NAMES,EPS,Data,Target,Rule,proxy,save
from adv_mixed_prepare import Generator
from adv_mixed_algorithms import METHODS,GRADIENT,SCORE,QUERY,gradient,scores,queries,goal_probability


def main():
    torch.set_num_threads(2);save(ROOT/'preflight.json',dict(status='running'));data=Data();target=Target(data)
    models={True:proxy(data,True),False:proxy(data,False)};device=models[True][1]
    generators=nn.ModuleList(Generator(f) for f in data.widths).to(device)
    generators.load_state_dict(torch.load(ROOT/'GAN_model/model.pt',map_location=device,weights_only=True)['state_dict']);generators.eval()
    teacher=np.load(ROOT/'clean_probability.npy',mmap_mode='r');checks=[]
    for code,name in enumerate(NAMES):
        positions=[]
        for label in [0,1]:
            candidates=np.flatnonzero((data.records['dataset_id']==code)&(data.records['label']==label)&((teacher>=.5)==label))
            assert len(candidates);positions.append(candidates[0])
        records=data.records[positions];x=data.local(records,code);y=np.asarray(records['label']);rule=data.rules[code]
        independent=target.model.predict(xgb.DMatrix(sparse_batch(x,code,data.offsets,157)))
        assert np.allclose(independent,target.local(x,code),rtol=0,atol=1e-6)
        assert np.allclose(teacher[positions],independent,rtol=0,atol=1e-6)
        for i,method in enumerate(METHODS):
            np.random.seed(42+i);random.seed(42+i);torch.manual_seed(42+i);rng=np.random.default_rng(42+i)
            if method in GRADIENT:
                model,device=models[method!='Transfer'];z=gradient(method,x,y,code,model,device,rule,rng)
            elif method in SCORE:z=scores(method,x,y,code,target,rule,rng)
            elif method in QUERY:
                z,counts,capped,failed=queries(method,x,y,code,target,rule,rng);assert np.all(counts<=4096)
            else:
                with torch.inference_mode():
                    tensor=torch.as_tensor(x,device=device);labels=torch.as_tensor((1-y).astype(np.int64),device=device)
                    z=rule.tensor(tensor+generators[code](tensor,labels),tensor).cpu().numpy()
            z=rule.project(z,x)
            assert z.shape==x.shape and np.isfinite(z).all() and np.max(np.abs(z.astype(np.float64)-x))<=EPS+1e-7
            assert np.array_equal(z[:,~rule.mask],x[:,~rule.mask])
            checks.append(dict(dataset=name,method=method,source_labels=y.tolist(),status='passed'));print('BOTH-CLASS PREFLIGHT',name,method,'passed',flush=True)
    class Linear:
        def __call__(self,x,code):return x[:,0]-2*x[:,1]-.4
    class Oracle:
        def __init__(self):self.queries=0
        def local(self,x,code):self.queries+=len(x);return (1/(1+np.exp(-(x[:,0]-2*x[:,1]-.4)))).astype(np.float32)
    x=np.array([[0.,0.,3.],[1.,0.,3.]],np.float32);y=np.array([0,1],np.uint8)
    rule=Rule(['Rate','Duration','Protocol Type'],[-10]*3,[10]*3);linear=Linear();oracle=Oracle();clean=goal_probability(oracle.local(x,0),y);math=[]
    for i,method in enumerate(GRADIENT+SCORE+QUERY):
        np.random.seed(42+i);torch.manual_seed(42+i);rng=np.random.default_rng(42+i)
        if method in GRADIENT:z=gradient(method,x,y,0,linear,torch.device('cpu'),rule,rng)
        elif method in SCORE:z=scores(method,x,y,0,oracle,rule,rng)
        else:z,_,_,_=queries(method,x,y,0,oracle,rule,rng)
        z=rule.project(z,x);assert np.all(goal_probability(oracle.local(z,0),y)>=clean-1e-6)
        assert np.array_equal(z[:,2],x[:,2])
        if method in ['FGSM','PGD','BIM','MIM','Transfer']:assert np.allclose(z[:,:2]-x[:,:2],[[.3,-.3],[-.3,.3]],atol=1e-6),method
        if method=='DeepFool':assert np.all((oracle.local(z,0)>=.5)!=y)
        math.append(dict(method=method,opposite_class_directions_verified=True,status='passed'))
    save(ROOT/'preflight.json',dict(status='passed',both_classes_checked=True,real_checks=checks,independent_math_checks=math,
        trained_sparse_adapter_equivalent=True,scope='Correctness examples from training only; no reduction of final population'))


if __name__=='__main__':main()
