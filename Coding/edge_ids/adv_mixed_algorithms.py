"""Per-label opposite-class attacks. Tree gradients are never fabricated."""
import logging
import numpy as np
import torch
from torch.nn import functional as F
from adv_mixed_data import EPS

GRADIENT=['FGSM','CW','PGD','JSMA','DeepFool','BIM','MIM','Transfer']
SCORE=['NES','SPSA','Square'];QUERY=['Boundary','HopSkipJump','ZOO']
METHODS=GRADIENT+SCORE+QUERY+['GAN']
PARAMETERS={
 'FGSM':dict(steps=1,epsilon=EPS,access='Distilled-proxy targeted gradient transfer'),
 'CW':dict(steps=60,searches=3,optimizer='Adam',lr=.01,initial_constant=.1,confidence=0,variant='Targeted L2 plus margin hinge, additional local Linf box'),
 'PGD':dict(steps=20,step=.03,random_start=True),
 'JSMA':dict(theta=.05,gamma=.2,variant='Positive two-coordinate binary probability-Jacobian saliency, local projection'),
 'DeepFool':dict(steps=20,overshoot=.02,variant='Binary signed-logit linearization with local projection'),
 'BIM':dict(steps=20,step=.03,random_start=False),
 'MIM':dict(steps=20,step=.03,decay=1,normalization='L1'),
 'Transfer':dict(steps=20,step=.03,random_start=True,access='Independent original supervised MLP PGD to GB'),
 'NES':dict(steps=20,directions=16,sigma=.05,step=.03,distribution='Antithetic Gaussian',acceptance='GB target probability improvement'),
 'SPSA':dict(steps=20,directions=16,sigma=.05,step=.03,distribution='Antithetic Rademacher',acceptance='GB target probability improvement'),
 'Square':dict(steps=100,p_init=.8,variant='Tabular native-coordinate block adaptation, no image geometry claim'),
 'Boundary':dict(steps=15,trials=5,samples=8,initial_local_candidates=64,query_cap=4096,access='GB hard decisions only'),
 'HopSkipJump':dict(steps=10,max_eval=64,init_eval=16,initial_local_candidates=64,query_cap=4096,access='GB hard decisions only'),
 'ZOO':dict(steps=20,searches=2,lr=.02,h=.02,parallel_coordinates=8,query_cap=4096,access='GB probabilities; ART coordinate Adam on deltas'),
 'GAN':dict(epochs=5,batch=4096,access='Label-conditioned training-only perturbation generator, GB-evaluated')}


def goal_probability(p,y):return np.where(y==0,p,1-p)


def gradient(method,x,y,code,model,device,rule,rng):
    original=torch.as_tensor(x,device=device);truth=torch.as_tensor(y.astype(np.int64),device=device);sign=1-2*truth
    mask=torch.as_tensor(rule.mask,device=device);project=lambda z:rule.tensor(z,original)
    if method=='CW':
        low,high=rule.bounds(x);low=torch.as_tensor(low,device=device);high=torch.as_tensor(high,device=device)
        mid=(low+high)/2;half=((high-low)/2).clamp_min(1e-7);initial=torch.atanh(((original-mid)/half).clamp(-.999999,.999999))
        c=torch.full((len(x),),.1,device=device);lower=torch.zeros_like(c);upper=torch.full_like(c,float('inf'))
        best=original.clone();distance=torch.full_like(c,float('inf'));fallback=original.clone();strength=(sign*model(original,code)).detach()
        for search in range(3):
            w=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([w],lr=.01);found=torch.zeros(len(x),dtype=torch.bool,device=device)
            for step in range(60):
                z=project(mid+half*torch.tanh(w));logit=model(z,code);margin=sign*logit;norm=(z-original).square().sum(1)
                objective=norm+c*F.relu(-margin);optimizer.zero_grad();objective.sum().backward();optimizer.step()
                with torch.no_grad():
                    hit=(logit>=0)!=(truth.bool());found|=hit;improved=hit&(norm<distance)
                    best[improved]=z[improved];distance[improved]=norm[improved]
                    stronger=margin>strength;fallback[stronger]=z[stronger];strength[stronger]=margin[stronger]
            upper=torch.where(found,torch.minimum(upper,c),upper);lower=torch.where(found,lower,torch.maximum(lower,c))
            c=torch.where(torch.isfinite(upper),(lower+upper)/2,c*10)
        return torch.where(torch.isfinite(distance)[:,None],best,fallback).detach().cpu().numpy()
    z=original.clone()
    if method=='JSMA':
        available=mask.expand_as(z).clone()
        # gamma is the fraction of all native input coordinates (as in JSMA),
        # while immutable coordinates remain excluded from saliency search.
        for step in range(int(x.shape[1]*.2)//2):
            variable=z.detach().requires_grad_(True);logit=model(variable,code);goal=torch.sigmoid(sign*logit)
            g=torch.autograd.grad(goal.sum(),variable)[0]
            active=(logit.detach()>=0)==truth.bool()
            possible=available&(project(z+.05)>z+1e-7)&active[:,None]
            g=torch.where(possible,g,torch.full_like(g,-float('inf')));pairs=g[:,:,None]+g[:,None,:]
            triangle=torch.triu(torch.ones(z.shape[1],z.shape[1],dtype=torch.bool,device=device),1)
            saliency=torch.where(triangle&(pairs>0),pairs.square(),torch.full_like(pairs,-float('inf'))).flatten(1)
            selected=saliency.argmax(1);active=torch.isfinite(saliency.max(1).values);rows=torch.arange(len(x),device=device)[active]
            a=(selected//z.shape[1])[active];b=(selected%z.shape[1])[active]
            z[rows,a]+=.05;z[rows,b]+=.05;z=project(z).detach();available&=project(z+.05)>z+1e-7
        return z.cpu().numpy()
    if method=='DeepFool':
        for step in range(20):
            variable=z.detach().requires_grad_(True);margin=sign*model(variable,code)
            g=torch.autograd.grad(margin.sum(),variable)[0]*mask
            z=project(z+1.02*(-margin.detach()+1e-5).clamp_min(0)[:,None]*g/(g.square().sum(1,keepdim=True)+1e-12)).detach()
        return z.cpu().numpy()
    if method in ['PGD','Transfer']:z=project(z+torch.as_tensor(rng.uniform(-EPS,EPS,x.shape).astype(np.float32),device=device)*mask)
    momentum=torch.zeros_like(z)
    for step in range(1 if method=='FGSM' else 20):
        variable=z.detach().requires_grad_(True);g=torch.autograd.grad(F.logsigmoid(sign*model(variable,code)).sum(),variable)[0]*mask
        if method=='MIM':momentum+=g/(g.abs().sum(1,keepdim=True)+1e-12);g=momentum
        z=project(z+(EPS if method=='FGSM' else .03)*g.sign()).detach()
    return z.cpu().numpy()


def scores(method,x,y,code,target,rule,rng):
    z=x.copy();best=goal_probability(target.local(z,code),y);coordinates=np.flatnonzero(rule.mask)
    if not len(coordinates):return z
    for step in range(100 if method=='Square' else 20):
        if method=='Square':
            fraction=.8/(2**sum(step/100>=s for s in [.01,.05,.2,.5,.8]));width=max(1,int(np.ceil(fraction*len(coordinates))));candidate=z.copy()
            for row in range(len(x)):
                start=rng.integers(0,len(coordinates)-width+1);indices=coordinates[start:start+width]
                candidate[row,indices]=x[row,indices]+rng.choice([-EPS,EPS],width)
        else:
            directions=(rng.normal(size=(len(x),16,x.shape[1])) if method=='NES' else rng.choice([-1.,1.],size=(len(x),16,x.shape[1]))).astype(np.float32)*rule.mask
            plus=rule.project(z[:,None,:]+.05*directions,x[:,None,:]);minus=rule.project(z[:,None,:]-.05*directions,x[:,None,:])
            hi=goal_probability(target.local(plus.reshape(-1,x.shape[1]),code).reshape(len(x),16),y[:,None])
            lo=goal_probability(target.local(minus.reshape(-1,x.shape[1]),code).reshape(len(x),16),y[:,None])
            g=(((hi-lo)/.1)[:,:,None]*directions).mean(1);candidate=z+.03*np.sign(g)
        candidate=rule.project(candidate,x);p=goal_probability(target.local(candidate,code),y)
        better=p>best;z[better]=candidate[better];best[better]=p[better]
    return z


class QueryBudget(RuntimeError):pass


def queries(method,x,y,code,target,rule,rng):
    from art.estimators.classification import BlackBoxClassifier
    from art.attacks.evasion import BoundaryAttack,HopSkipJump,ZooAttack
    logging.getLogger('art').setLevel(logging.ERROR)
    output=x.copy();counts=np.zeros(len(x),np.int64);capped=np.zeros(len(x),bool);init_failed=np.zeros(len(x),bool)
    initial=[None]*len(x)
    if method!='ZOO':
        # Batch the seed search, using hard labels only. Failure to find a
        # feasible target seed is retained, not hidden by dropping the record.
        seeds=rule.project(x[:,None,:]+rng.uniform(-EPS,EPS,(len(x),64,x.shape[1])).astype(np.float32),x[:,None,:])
        labels=(target.local(seeds.reshape(-1,x.shape[1]),code)>=.5).reshape(len(x),64)
        counts[:]=64
        for row in range(len(x)):
            positive=np.flatnonzero(labels[row]==1-y[row])
            if len(positive):initial[row]=seeds[row,positive[0]:positive[0]+1]
            else:init_failed[row]=True
    for row in range(len(x)):
        if init_failed[row]:continue
        anchor=x[row:row+1];target_label=int(1-y[row]);closest=initial[row][0].copy() if initial[row] is not None else None
        distance=float(np.linalg.norm(closest.astype(np.float64)-anchor[0])) if closest is not None else float('inf')
        fallback=anchor[0].copy();strength=-float('inf')
        def predict(values):
            nonlocal closest,distance,fallback,strength
            if counts[row]+len(values)>4096:raise QueryBudget()
            actual=rule.project(values,anchor);p=target.local(actual,code);counts[row]+=len(values)
            hit=np.flatnonzero((p>=.5)==target_label)
            if len(hit):
                norm=np.linalg.norm(actual[hit].astype(np.float64)-anchor.astype(np.float64),axis=1);i=np.argmin(norm)
                if norm[i]<distance:closest=actual[hit[i]].copy();distance=norm[i]
            if method!='ZOO':
                label=p>=.5;return np.column_stack([~label,label]).astype(np.float32)
            goal=p if target_label else 1-p;i=np.argmax(goal)
            if goal[i]>strength:fallback=actual[i].copy();strength=goal[i]
            p=np.where(p==.5,np.nextafter(np.float32(.5),np.float32(1)),p)
            return np.column_stack([1-p,p])
        try:
            target_onehot=np.eye(2,dtype=np.float32)[[target_label]]
            if method=='ZOO':
                oracle=BlackBoxClassifier(lambda d:predict(anchor+d),input_shape=(x.shape[1],),nb_classes=2,clip_values=(-EPS,EPS))
                attack=ZooAttack(oracle,targeted=True,learning_rate=.02,max_iter=20,binary_search_steps=2,use_resize=False,use_importance=False,nb_parallel=min(8,x.shape[1]),variable_h=.02,batch_size=1,verbose=False)
                result=anchor+attack.generate(np.zeros_like(anchor),y=target_onehot)
            else:
                lo,hi=rule.bounds(anchor);oracle=BlackBoxClassifier(predict,input_shape=(x.shape[1],),nb_classes=2,clip_values=(lo[0],np.maximum(hi[0],lo[0]+1e-5)))
                if method=='Boundary':
                    attack=BoundaryAttack(oracle,targeted=True,max_iter=15,num_trial=5,sample_size=8,init_size=1,verbose=False)
                    result=attack.generate(anchor.copy(),y=target_onehot,x_adv_init=initial[row])
                else:
                    attack=HopSkipJump(oracle,targeted=True,max_iter=10,max_eval=64,init_eval=16,init_size=1,verbose=False)
                    result=attack.generate(anchor.copy(),y=target_onehot,x_adv_init=initial[row],mask=rule.mask.astype(np.float32))
            output[row]=rule.project(result,anchor)[0]
        except QueryBudget:
            capped[row]=True;output[row]=closest if closest is not None else fallback
        assert counts[row]<=4096
    return output,counts,capped,init_failed
