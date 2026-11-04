"""Independent exact-kernel and exact-neighbor checks on synthetic data."""
import gc
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from exact_classical_models import ExactBlockKNN,ExactPolynomialSVM


def main():
    project=Path(__file__).resolve().parent
    rng=np.random.default_rng(42);widths=[3,2,4,3];domains=[]
    for code,f in enumerate(widths):
        x=rng.normal(size=(23,f)).astype(np.float32)
        if code==0:x+=10
        if code==1:x[:,0]=3
        domains.append(dict(name=f'synthetic_{code}',x=x,y=rng.integers(0,2,23,dtype=np.uint8),indices=[np.arange(23)]))
    offsets=np.cumsum([0]+widths);dimension=int(offsets[-1]+4)
    full=np.zeros((92,dimension),np.float64)
    for d,domain in enumerate(domains):
        full[d*23:(d+1)*23,offsets[d]:offsets[d+1]]=domain['x']
        full[d*23:(d+1)*23,offsets[-1]+d]=1
    labels=np.concatenate([d['y'] for d in domains]);maximum_distance_error=0.
    with tempfile.TemporaryDirectory(dir=project/'artifacts') as temporary:
        assert Path(temporary).resolve().is_relative_to((project/'artifacts').resolve())
        model=ExactBlockKNN(temporary,k=5,workers=1,rotate_domains=[d['name'] for d in domains]).fit(domains)
        for code,f in enumerate(widths):
            query=rng.normal(size=(7,f));query[0]=0
            p,idx,dist=model.predict_probability(query,code,True)
            embedded=np.zeros((len(query),dimension));embedded[:,offsets[code]:offsets[code+1]]=query
            embedded[:,offsets[-1]+code]=1
            brute=np.square(embedded[:,None,:]-full[None,:,:]).sum(-1)
            nearest=np.argsort(brute,axis=1)[:,:5]
            expected=np.sqrt(np.take_along_axis(brute,nearest,1))
            maximum_distance_error=max(maximum_distance_error,float(np.max(np.abs(dist-expected))))
            assert np.allclose(dist,expected,atol=1e-10,rtol=0)
            assert np.allclose(p,labels[nearest].mean(1))
        del model;gc.collect()
    kernel=ExactPolynomialSVM(widths).double()
    lengths=[f*(f+1)//2+f+1 for f in widths];map_offsets=np.cumsum([0]+lengths)
    mapped=np.zeros((len(full),int(map_offsets[-1]+1)))
    for code,d in enumerate(domains):
        x=torch.from_numpy(d['x'].astype(np.float64))
        q,l,marker=kernel.features(x,code);a=int(map_offsets[code]);nq=q.shape[1]
        mapped[code*23:(code+1)*23,a:a+nq]=q.numpy()
        mapped[code*23:(code+1)*23,a+nq:a+nq+widths[code]]=l.numpy()
        mapped[code*23:(code+1)*23,a+lengths[code]-1]=marker
    mapped[:,-1]=kernel.coef0
    expected=(kernel.gamma*(full@full.T)+kernel.coef0)**2
    error=float(np.max(np.abs(mapped@mapped.T-expected)))
    assert np.allclose(mapped@mapped.T,expected,rtol=1e-7,atol=1e-8)
    result=dict(status='passed',KNN='KD-tree decomposition matches independent full-block brute-force neighbors',
                max_neighbor_distance_error=maximum_distance_error,
                search_rotation='All covariance eigenvector directions retained; no whitening, no dimensionality reduction',
                SVM='Complete polynomial feature map matches direct nonlinear kernel Gram matrix',
                max_kernel_entry_error=error,kernel='degree 2 polynomial; no Nystroem/random-feature approximation',
                scope='Synthetic algorithm checks only; not training subsets or test results')
    out=project/'artifacts/expanded_classical';out.mkdir(exist_ok=True)
    (out/'exactness_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
