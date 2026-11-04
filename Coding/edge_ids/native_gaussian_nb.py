"""Gaussian NB sufficient statistics for the full native block namespace."""
import numpy as np
from classical_data import layout


class NativeGaussianNB:
    def fit(self,domains):
        widths,self.offsets,self.dimension=layout(domains)
        counts=np.zeros(2,np.float64);sums=np.zeros((2,self.dimension));squares=np.zeros_like(sums)
        for code,d in enumerate(domains):
            rows=d['indices'][0];a,b=self.offsets[code:code+2]
            for start in range(0,len(rows),100_000):
                take=rows[start:start+100_000];x=np.asarray(d['x'][take],dtype=np.float64);y=d['y'][take]
                for label in (0,1):
                    selected=x[y==label];n=len(selected);counts[label]+=n
                    sums[label,a:b]+=selected.sum(0);squares[label,a:b]+=np.square(selected).sum(0)
                    sums[label,self.offsets[-1]+code]+=n;squares[label,self.offsets[-1]+code]+=n
            print('Gaussian NB accumulated all training rows',d['name'],len(rows),flush=True)
        self.counts=counts;self.means=sums/counts[:,None]
        global_mean=sums.sum(0)/counts.sum()
        global_variance=squares.sum(0)/counts.sum()-global_mean**2
        self.epsilon=1e-9*float(global_variance.max())
        self.variances=np.maximum(squares/counts[:,None]-self.means**2,0)+self.epsilon
        self.priors=counts/counts.sum()
        self.base=np.log(self.priors)-.5*np.sum(np.log(2*np.pi*self.variances)+self.means**2/self.variances,axis=1)
        self.linear=self.means[1]/self.variances[1]-self.means[0]/self.variances[0]
        self.quadratic=-.5*(1/self.variances[1]-1/self.variances[0])
        return self

    def decision_function(self,x,domain):
        x=np.asarray(x,dtype=np.float64);a,b=self.offsets[domain:domain+2];marker=self.offsets[-1]+domain
        return self.base[1]-self.base[0]+self.linear[marker]+self.quadratic[marker]+x@self.linear[a:b]+np.square(x)@self.quadratic[a:b]

    def save(self,path):
        np.savez(path,counts=self.counts,means=self.means,variances=self.variances,priors=self.priors,
                 offsets=self.offsets,base=self.base,linear=self.linear,quadratic=self.quadratic,epsilon=self.epsilon)

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path) as archive:
            for name in archive.files:setattr(model,name,archive[name])
        model.offsets=model.offsets.tolist();model.dimension=len(model.linear)
        return model
