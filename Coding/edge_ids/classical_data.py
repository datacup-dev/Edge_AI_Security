"""Full native feature namespace; no PCA or training-row subsampling."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from train_adapted import load_domains,save_json

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts/classical_native'


def layout(domains):
    widths=[d['x'].shape[1] for d in domains]
    offsets=np.cumsum([0]+widths).tolist()
    return widths,offsets,offsets[-1]+len(domains)


def sparse_batch(x,code,offsets,dimensions):
    """CSR with every active native coordinate (including observed zeros)."""
    n,f=x.shape
    data=np.empty((n,f+1),np.float32);data[:,:f]=x;data[:,f]=1
    columns=np.r_[np.arange(offsets[code],offsets[code+1]),offsets[-1]+code].astype(np.int32)
    indices=np.broadcast_to(columns,(n,f+1)).copy().reshape(-1)
    indptr=np.arange(0,(n+1)*(f+1),f+1,dtype=np.int32)
    return sparse.csr_matrix((data.reshape(-1),indices,indptr),shape=(n,dimensions),copy=False)


def load_csc():
    info=json.loads((OUT/'manifest.json').read_text())
    matrix=sparse.csc_matrix((np.load(OUT/'data.npy',mmap_mode='r'),
                             np.load(OUT/'indices.npy',mmap_mode='r'),
                             np.load(OUT/'indptr.npy',mmap_mode='r')),
                            shape=tuple(info['shape']),copy=False)
    matrix.has_sorted_indices=True;matrix.has_canonical_format=True
    return matrix,np.load(OUT/'labels.npy',mmap_mode='r'),info


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        print('Existing full-native CSC representation reused',flush=True);return
    domains=load_domains();widths,offsets,dimensions=layout(domains)
    counts=np.zeros(dimensions,np.int64);mins=[];maxs=[]
    n=sum(len(d['indices'][0]) for d in domains)
    for code,d in enumerate(domains):
        low=np.full(widths[code],np.inf);high=np.full(widths[code],-np.inf)
        rows=d['indices'][0]
        for start in range(0,len(rows),100_000):
            block=np.asarray(d['x'][rows[start:start+100_000]])
            counts[offsets[code]:offsets[code+1]]+=np.count_nonzero(block,axis=0)
            low=np.minimum(low,block.min(0));high=np.maximum(high,block.max(0))
        counts[offsets[-1]+code]=len(rows);mins.append(low.tolist());maxs.append(high.tolist())
        print('Counted full training namespace',d['name'],len(rows),flush=True)
    nnz=int(counts.sum());assert nnz<np.iinfo(np.int32).max
    pointers=np.r_[0,np.cumsum(counts)].astype(np.int32)
    data=np.lib.format.open_memmap(OUT/'data.npy',mode='w+',dtype=np.float32,shape=(nnz,))
    indices=np.lib.format.open_memmap(OUT/'indices.npy',mode='w+',dtype=np.int32,shape=(nnz,))
    labels=np.lib.format.open_memmap(OUT/'labels.npy',mode='w+',dtype=np.uint8,shape=(n,))
    global_offset=0
    for code,d in enumerate(domains):
        rows=d['indices'][0];labels[global_offset:global_offset+len(rows)]=d['y'][rows]
        for j in range(widths[code]):
            values=np.asarray(d['x'][rows,j]);mask=values!=0;col=offsets[code]+j
            a,b=map(int,pointers[col:col+2]);assert b-a==int(mask.sum())
            data[a:b]=values[mask]
            indices[a:b]=np.flatnonzero(mask).astype(np.int32)+global_offset
        col=offsets[-1]+code;a,b=map(int,pointers[col:col+2]);data[a:b]=1
        indices[a:b]=np.arange(global_offset,global_offset+len(rows),dtype=np.int32)
        global_offset+=len(rows)
        data.flush();indices.flush();labels.flush()
        print('Wrote full native block',d['name'],flush=True)
    np.save(OUT/'indptr.npy',pointers)
    info=dict(shape=[n,dimensions],nnz=nnz,widths=widths,feature_offsets=offsets,
              domain_order=[d['name'] for d in domains],train_rows=[len(d['indices'][0]) for d in domains],
              training_min=mins,training_max=maxs,
              representation='Disjoint native standardized feature blocks plus one-hot domain markers',
              reduction='None; all selected native features and all training records preserved',
              sparse_zero='Zero coefficients omitted in CSC storage; these are abstract coordinates, not measured missing network fields')
    save_json(OUT/'manifest.json',info)
    print('Full native CSC prepared',json.dumps(dict(rows=n,dimensions=dimensions,nnz=nnz)),flush=True)


if __name__=='__main__':main()
