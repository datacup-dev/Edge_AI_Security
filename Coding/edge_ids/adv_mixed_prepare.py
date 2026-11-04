"""Verify all mixed training records, distill GB, fit a label-conditioned GAN."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from adv_mixed_data import PROJECT,ROOT,NAMES,COUNT,EPS,Data,Target,read,save,digest,proxy
from adapted_models import AdaptedModel


class Generator(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width+18,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,width),nn.Tanh())
    def forward(self,x,target):
        condition=F.one_hot(target.long(),2).float();noise=torch.randn(len(x),16,device=x.device)
        return EPS*self.net(torch.cat([torch.tanh(x/5),condition,noise],1))


def main():
    ROOT.mkdir(parents=True,exist_ok=True);data=Data();target=Target(data,4)
    manifest=read(PROJECT/'artifacts/mixed_training/manifest.json');assert digest(data.mixed_path)==manifest['sha256']
    if (ROOT/'configuration.json').exists():
        previous=read(ROOT/'configuration.json');assert previous['baseline_sha256']==target.hash and previous['mixed_sha256']==manifest['sha256']
        assert previous['epsilon_standardized_linf']==EPS
    sources=[]
    for code,name in enumerate(NAMES):
        part=data.records[data.records['dataset_id']==code];directory=PROJECT/'artifacts/encoded_inputs'/name
        split=np.load(directory/'split.npy',mmap_mode='r');labels=np.load(directory/'labels.npy',mmap_mode='r')
        assert np.array_equal(np.sort(part['source_row']),np.flatnonzero(split==0))
        assert np.array_equal(part['label'],labels[part['source_row']])
        sources.append(dict(dataset=name,rows=len(part),benign=int((part['label']==0).sum()),attack=int((part['label']==1).sum())))
    assert sum(s['benign'] for s in sources)==2290435 and sum(s['attack'] for s in sources)==32740311
    save(ROOT/'configuration.json',dict(scope='All original mixed-training records, both classes, no final subsampling',training_rows=COUNT,
        benign=2290435,attack=32740311,sources=sources,mixed_path=str(data.mixed_path.relative_to(PROJECT)),mixed_sha256=manifest['sha256'],
        baseline_sha256=target.hash,epsilon_standardized_linf=EPS,target='1 minus original true label',validation_test_used=False,
        storage='Compact native values width64, trailing NaN; reconstruct frozen 157-coordinate GB namespace using immutable source dataset ID',seed=42,
        domains=[dict(dataset=NAMES[c],features=f,mutable=data.rules[c].mask.tolist(),training_min=data.rules[c].low.tolist(),training_max=data.rules[c].high.tolist()) for c,f in enumerate(data.features)]))
    print('VERIFIED ALL MIXED TRAINING RECORDS',sources,flush=True)
    teacher_path=ROOT/'clean_probability.npy'
    if not (ROOT/'teacher_complete.json').exists():
        p=np.lib.format.open_memmap(teacher_path,mode='w+',dtype=np.float32,shape=(COUNT,))
        for start in range(0,COUNT,32768):
            batch=data.records[start:start+32768];p[start:start+len(batch)]=target.compact(data.compact(batch),batch)
            if start//32768%100==0:save(ROOT/'preparation_progress.json',dict(stage='Frozen GB training-only teacher responses',rows=start+len(batch),total=COUNT))
        p.flush();save(ROOT/'teacher_complete.json',dict(rows=COUNT,baseline_sha256=target.hash))
    teacher=np.load(teacher_path,mmap_mode='r');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    directory=ROOT/'surrogate';directory.mkdir(exist_ok=True)
    if not (directory/'complete.json').exists():
        torch.manual_seed(42);model=AdaptedModel('MLP',data.features).to(device)
        model.load_state_dict(torch.load(PROJECT/'artifacts/adapted_experiment/MLP/best.pt',map_location=device,weights_only=True)['state_dict'])
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);history=[]
        if (directory/'last.pt').exists():
            state=torch.load(directory/'last.pt',map_location=device,weights_only=True);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        for epoch in range(len(history),3):
            order=np.random.default_rng(42+epoch).permutation(COUNT);model.train();seen=0;loss_sum=0.
            for start in range(0,COUNT,8192):
                positions=order[start:start+8192];batch=data.records[positions];optimizer.zero_grad(set_to_none=True);loss=0
                for code in range(4):
                    selected=np.flatnonzero(batch['dataset_id']==code)
                    if not len(selected):continue
                    x=torch.as_tensor(data.local(batch[selected],code),device=device);y=torch.as_tensor(np.asarray(teacher[positions[selected]]),device=device)
                    loss=loss+F.binary_cross_entropy_with_logits(model(x,code),y,reduction='sum')/len(batch)
                assert torch.isfinite(loss);loss.backward();optimizer.step();seen+=len(batch);loss_sum+=float(loss.detach())*len(batch)
                if start//8192%200==0:save(ROOT/'preparation_progress.json',dict(stage='Both-class mixed-training GB distillation',epoch=epoch+1,rows=seen,total=COUNT))
            assert seen==COUNT;history.append(dict(epoch=epoch+1,training_rows=seen,soft_bce=loss_sum/seen))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),directory/'last.pt')
            print('SURROGATE',history[-1],flush=True)
        torch.save(dict(state_dict=model.state_dict()),directory/'model.pt')
        save(directory/'complete.json',dict(optimizer='Adam',lr=.0001,betas=[.9,.999],dropout=.3,batch=8192,epochs=3,selection='Fixed final epoch; no validation/test fitting or selection',history=history))
        del model,optimizer
    directory=ROOT/'GAN_model';directory.mkdir(exist_ok=True)
    if not (directory/'complete.json').exists():
        torch.manual_seed(42);frozen,_=proxy(data);generators=nn.ModuleList(Generator(f) for f in data.widths).to(device)
        critics=nn.ModuleList(nn.Sequential(nn.Linear(f+2,64),nn.LeakyReLU(.2),nn.Linear(64,32),nn.LeakyReLU(.2),nn.Linear(32,1)) for f in data.widths).to(device)
        go=torch.optim.Adam(generators.parameters(),lr=.0001,betas=(.5,.9));co=torch.optim.Adam(critics.parameters(),lr=.0002,betas=(.5,.999));history=[]
        if (directory/'last.pt').exists():
            state=torch.load(directory/'last.pt',map_location=device,weights_only=True);generators.load_state_dict(state['generators']);critics.load_state_dict(state['critics']);go.load_state_dict(state['go']);co.load_state_dict(state['co']);history=state['history']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        for epoch in range(len(history),5):
            order=np.random.default_rng(42+epoch).permutation(COUNT);seen=0;gs=ds=0.
            for start in range(0,COUNT,4096):
                batch=data.records[order[start:start+4096]];parts=[];co.zero_grad(set_to_none=True);dloss=0
                for code in range(4):
                    selected=np.flatnonzero(batch['dataset_id']==code)
                    if not len(selected):continue
                    x=torch.as_tensor(data.local(batch[selected],code),device=device);y=torch.as_tensor(batch['label'][selected].astype(np.int64),device=device)
                    condition=F.one_hot(y,2).float();parts.append((code,x,y,condition));fake=data.rules[code].tensor(x+generators[code](x,1-y),x).detach()
                    dloss=dloss+(F.softplus(-critics[code](torch.cat([x,condition],1))).sum()+F.softplus(critics[code](torch.cat([fake,condition],1))).sum())/len(batch)
                assert torch.isfinite(dloss);dloss.backward();co.step()
                for p in critics.parameters():p.requires_grad_(False)
                go.zero_grad(set_to_none=True);gloss=0
                for code,x,y,condition in parts:
                    fake=data.rules[code].tensor(x+generators[code](x,1-y),x);margin=(1-2*y)*frozen(fake,code)
                    gloss=gloss+(F.softplus(-margin).sum()+.05*F.softplus(-critics[code](torch.cat([fake,condition],1))).sum()+.01*(fake-x).square().mean(1).sum())/len(batch)
                assert torch.isfinite(gloss);gloss.backward();go.step()
                for p in critics.parameters():p.requires_grad_(True)
                seen+=len(batch);gs+=float(gloss.detach())*len(batch);ds+=float(dloss.detach())*len(batch)
                if start//4096%200==0:save(ROOT/'preparation_progress.json',dict(stage='Both-class conditional GAN fitting',epoch=epoch+1,rows=seen,total=COUNT))
            assert seen==COUNT;history.append(dict(epoch=epoch+1,training_rows=seen,generator_loss=gs/seen,discriminator_loss=ds/seen))
            torch.save(dict(generators=generators.state_dict(),critics=critics.state_dict(),go=go.state_dict(),co=co.state_dict(),history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),directory/'last.pt');print('GAN',history[-1],flush=True)
        torch.save(dict(state_dict=generators.state_dict()),directory/'model.pt')
        save(directory/'complete.json',dict(generator_optimizer='Adam .0001 betas .5/.9',discriminator_optimizer='Adam .0002 betas .5/.999',dropout=0,batch=4096,epochs=5,selection='Fixed final epoch, training only',history=history))
    save(ROOT/'preparation_progress.json',dict(stage='complete'))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        save(ROOT/'preparation_failure.json',dict(status='failed',traceback=traceback.format_exc()));raise
