"""Exact full-block KNN and exact finite polynomial-kernel feature map."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from classical_data import layout
from train_adapted import save_json


class NativeLinear(nn.Module):
    def __init__(self,widths):
        super().__init__()
        self.weights=nn.ParameterList(nn.Parameter(torch.zeros(f)) for f in widths)
        self.domain_bias=nn.Parameter(torch.zeros(len(widths)))
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x,domain):
        return x@self.weights[domain]+self.domain_bias[domain]+self.bias
    def squared_norm(self):
        return sum(p.square().sum() for p in self.weights)+self.domain_bias.square().sum()


class ExactPolynomialSVM(nn.Module):
    """K(u,v)=(gamma*u.dot(v)+coef0)^2 on full native-block+marker u/v.

    Cross-domain raw-feature products are identically zero. Proportional linear
    and feature*domain-marker coordinates are merged with their exact norm.
    The map itself is exact; finite optimization is not an optimality certificate.
    """
    def __init__(self,widths,gamma=None,coef0=1.):
        super().__init__()
        self.gamma=gamma or 1/(sum(widths)+len(widths));self.coef0=coef0
        self.quadratic=nn.ParameterList(nn.Parameter(torch.zeros(f*(f+1)//2)) for f in widths)
        self.linear=nn.ParameterList(nn.Parameter(torch.zeros(f)) for f in widths)
        self.domain_weight=nn.Parameter(torch.zeros(len(widths)))
        self.constant=nn.Parameter(torch.zeros(()));self.bias=nn.Parameter(torch.zeros(()))
        for d,f in enumerate(widths):
            ij=torch.triu_indices(f,f)
            self.register_buffer(f'ij_{d}',ij)
            scale=torch.ones(len(ij[0]),dtype=torch.float64)
            scale[ij[0]!=ij[1]]=2.**.5
            self.register_buffer(f'scale_{d}',scale)
    def features(self,x,domain):
        ij=getattr(self,f'ij_{domain}');scale=getattr(self,f'scale_{domain}').to(dtype=x.dtype)
        q=self.gamma*x[:,ij[0]]*x[:,ij[1]]*scale
        linear=(2*self.gamma*(self.gamma+self.coef0))**.5*x
        marker=(self.gamma**2+2*self.gamma*self.coef0)**.5
        return q,linear,marker
    def forward(self,x,domain):
        q,linear,marker=self.features(x,domain)
        return q@self.quadratic[domain]+linear@self.linear[domain]+marker*self.domain_weight[domain]+self.coef0*self.constant+self.bias
    def squared_norm(self):
        return sum(p.square().sum() for p in self.quadratic)+sum(p.square().sum() for p in self.linear)+self.domain_weight.square().sum()+self.constant.square()


class ExactBlockKNN:
    """Exact pooled Euclidean KNN, implemented with native-domain KD-trees.

    For different blocks, squared distance=||q||²+||t||²+2. Thus k minimum-norm
    points from every other block suffice; own-block KD-tree queries use eps=0.
    Constant native coordinates remain included through a query distance offset.
    """
    def __init__(self,directory,k=5,workers=4,rotate_domains=None):
        self.directory=Path(directory);self.k=k;self.workers=workers;self.trees=[];self.records=[]
        self.rotate_domains=set(rotate_domains or [])

    def fit(self,domains):
        self.directory.mkdir(exist_ok=True,parents=True)
        offset=0
        for code,d in enumerate(domains):
            rows=d['indices'][0];f=d['x'].shape[1];n=len(rows)
            low=np.full(f,np.inf);high=np.full(f,-np.inf)
            for start in range(0,n,100_000):
                block=np.asarray(d['x'][rows[start:start+100_000]])
                low=np.minimum(low,block.min(0));high=np.maximum(high,block.max(0))
            active=np.flatnonzero(low!=high);fixed=np.flatnonzero(low==high)
            if len(active)==0:active=np.array([0]);fixed=fixed[fixed!=0]
            rotation=None;center=None
            if d['name'] in self.rotate_domains:
                total=np.zeros(len(active));cross=np.zeros((len(active),len(active)))
                for start in range(0,n,100_000):
                    block=np.asarray(d['x'][rows[start:start+100_000]],dtype=np.float64)[:,active]
                    total+=block.sum(0);cross+=block.T@block
                center=total/n;covariance=cross/n-np.outer(center,center)
                eigenvalues,axes=np.linalg.eigh(covariance)
                rotation=axes[:,np.argsort(eigenvalues)[::-1]]
                assert np.allclose(rotation.T@rotation,np.eye(len(active)),atol=1e-12)
                print('Exact full-rank search rotation',d['name'],len(active),'directions retained',flush=True)
            target=self.directory/f'{code}_points.npy'
            if not target.exists():
                points=np.lib.format.open_memmap(target,mode='w+',dtype=np.float64,shape=(n,len(active)))
                for start in range(0,n,100_000):
                    block=np.asarray(d['x'][rows[start:start+100_000]],dtype=np.float64)[:,active]
                    points[start:start+100_000]=(block-center)@rotation if rotation is not None else block
                points.flush();del points
            points=np.load(target,mmap_mode='r')
            assert points.shape==(n,len(active))
            labels=np.asarray(d['y'][rows]).copy();np.save(self.directory/f'{code}_labels.npy',labels)
            norms=np.empty(n,np.float64)
            for start in range(0,n,100_000):
                block=np.asarray(d['x'][rows[start:start+100_000]],dtype=np.float64)
                norms[start:start+len(block)]=np.einsum('ij,ij->i',block,block)
            nearest=np.argpartition(norms,self.k-1)[:self.k]
            tree=cKDTree(points,leafsize=128,compact_nodes=True,balanced_tree=True,copy_data=False)
            record=dict(name=d['name'],rows=n,offset=offset,active=active.tolist(),fixed=fixed.tolist(),
                        constants=low[fixed].tolist(),other_indices=nearest.tolist(),
                        other_norms=norms[nearest].tolist(),other_labels=labels[nearest].tolist(),
                        rotation=rotation.tolist() if rotation is not None else None,
                        center=center.tolist() if center is not None else None)
            self.records.append(record);self.trees.append(tree)
            record['labels']=labels
            offset+=n;del norms
            save_json(self.directory/'progress.json',dict(stage='indexing',dataset=d['name'],rows=n,active_dimensions=len(active)))
            print('Exact KNN indexed',d['name'],n,'records',len(active),'nonconstant coordinates',flush=True)
        saved=[{k:v for k,v in r.items() if k!='labels'} for r in self.records]
        save_json(self.directory/'model.json',dict(k=self.k,metric='exact Euclidean in 157-dimensional full native namespace',eps=0.,
            workers=self.workers,domains=saved,rows=offset,checkpoint='Complete training reference vectors, labels and metadata; KD-trees reconstruct from these files'))
        return self

    def predict_probability(self,x,domain,return_neighbors=False):
        x=np.asarray(x,dtype=np.float64);r=self.records[domain]
        search=x[:,r['active']]
        if r.get('rotation') is not None:search=(search-np.asarray(r['center']))@np.asarray(r['rotation'])
        distances,indices=self.trees[domain].query(search,k=self.k,eps=0,workers=self.workers)
        same_squared=distances**2
        if r['fixed']:
            same_squared+=np.square(x[:,r['fixed']]-np.asarray(r['constants'])).sum(1,keepdims=True)
        labels=[r['labels'][indices]];ids=[indices+r['offset']];squared=[same_squared]
        qnorm=np.einsum('ij,ij->i',x,x)
        for code,other in enumerate(self.records):
            if code==domain:continue
            squared.append(qnorm[:,None]+np.asarray(other['other_norms'])[None,:]+2.)
            labels.append(np.broadcast_to(np.asarray(other['other_labels']), (len(x),self.k)))
            ids.append(np.broadcast_to(np.asarray(other['other_indices'])+other['offset'],(len(x),self.k)))
        squared=np.concatenate(squared,1);labels=np.concatenate(labels,1);ids=np.concatenate(ids,1)
        selected=np.argsort(squared,axis=1,kind='stable')[:,:self.k]
        chosen_labels=np.take_along_axis(labels,selected,1)
        probability=chosen_labels.mean(1)
        if return_neighbors:
            return probability,np.take_along_axis(ids,selected,1),np.sqrt(np.take_along_axis(squared,selected,1))
        return probability

    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        metadata=json.loads((directory/'model.json').read_text())
        model=cls(directory,k=metadata['k'],workers=metadata['workers'])
        for code,record in enumerate(metadata['domains']):
            points=np.load(directory/f'{code}_points.npy',mmap_mode='r')
            assert len(points)==record['rows']
            model.trees.append(cKDTree(points,leafsize=128,compact_nodes=True,balanced_tree=True,copy_data=False))
            record['labels']=np.load(directory/f'{code}_labels.npy',mmap_mode='r')
            model.records.append(record)
        return model
