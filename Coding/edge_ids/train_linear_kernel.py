"""Full-native logistic regression and exact polynomial-kernel primal SVM."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_adapted import load_domains,save_json
from exact_classical_models import NativeLinear,ExactPolynomialSVM
from extended_evaluation import export_evaluation,margin_probability

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts/expanded_classical'


@torch.inference_mode()
def predict(model,d,code,rows,device):
    model.eval();scores=np.empty(len(rows),np.float32)
    for start in range(0,len(rows),8192):
        take=rows[start:start+8192]
        x=torch.from_numpy(np.asarray(d['x'][take])).to(device)
        scores[start:start+len(take)]=model(x,code).cpu().numpy()
    return scores


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domains=load_domains();widths=[d['x'].shape[1] for d in domains]
    n=sum(len(d['indices'][0]) for d in domains)
    for kind,constructor in [('LR',NativeLinear),('SVM',ExactPolynomialSVM)]:
        directory=OUT/kind;directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=constructor(widths).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        config=dict(model=kind,optimizer='Adam',learning_rate=.001,epoch_limit=20,patience=3,
                    seed=42,batch_size=8192,dropout=0,training_rows=n,feature_dimension=sum(widths)+4,
                    representation='Full native blocks plus one-hot domain markers; no PCA',C=1.,
                    regularization='0.5*||w||^2/(C*N), excluding free intercept',
                    loss='mean binary cross-entropy' if kind=='LR' else 'mean hinge loss',
                    precision='float32; no autocast',validation='Mean four-domain macro-F1; raw-score AP tie-break',
                    kernel=None if kind=='LR' else dict(type='exact degree-2 polynomial',degree=2,gamma=model.gamma,coef0=model.coef0,
                        map='Complete finite map; no Nystroem, random features or support-point subsampling',
                        optimization='Finite stochastic primal optimization; no global-optimality certificate'),
                    probability='Sigmoid of logistic logit' if kind=='LR' else 'Uncalibrated sigmoid of signed margin; ROC/AP use raw margin')
        save_json(directory/'configuration.json',config)
        history=[];best=(-1.,-1.);stale=0
        if (directory/'last.pt').exists():
            state=torch.load(directory/'last.pt',map_location=device,weights_only=False)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            history=state['history'];best=tuple(state['best']);stale=state['stale']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        for epoch in range(len(history),20 if not (directory/'selection.json').exists() else len(history)):
            started=time.perf_counter();rng=np.random.default_rng(42+epoch)
            sets=[rng.permutation(d['indices'][0]) for d in domains]
            schedule=[(code,start) for code,rows in enumerate(sets) for start in range(0,len(rows),8192)]
            rng.shuffle(schedule);seen=0;total=0.;model.train()
            for batch,(code,start) in enumerate(schedule):
                rows=sets[code][start:start+8192];d=domains[code]
                x=torch.from_numpy(np.asarray(d['x'][rows])).to(device)
                y=torch.from_numpy(d['y'][rows].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True);score=model(x,code)
                empirical=F.binary_cross_entropy_with_logits(score,y) if kind=='LR' else F.relu(1-(2*y-1)*score).mean()
                loss=empirical+.5*model.squared_norm()/n
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {kind} objective')
                loss.backward();optimizer.step();total+=float(loss.detach())*len(rows);seen+=len(rows)
                if batch%100==0:
                    save_json(OUT/'linear_progress.json',dict(stage='training',model=kind,epoch=epoch+1,batch=batch+1,
                              batches=len(schedule),rows_seen=seen,elapsed_seconds=time.perf_counter()-started))
            assert seen==n
            validation={}
            for code,d in enumerate(domains):
                rows=d['indices'][1];score=predict(model,d,code,rows,device);y=d['y'][rows]
                validation[d['name']]=dict(macro_f1=float(f1_score(y,score>=0,average='macro',zero_division=0)),
                                          average_precision=float(average_precision_score(y,score)))
            pair=(float(np.mean([r['macro_f1'] for r in validation.values()])),float(np.mean([r['average_precision'] for r in validation.values()])))
            record=dict(epoch=epoch+1,training_rows=seen,train_loss=total/seen,validation=validation,
                        mean_validation_macro_f1=pair[0],mean_validation_ap=pair[1],elapsed_seconds=time.perf_counter()-started)
            history.append(record);save_json(directory/'history.json',history)
            if pair>best:
                best=pair;stale=0
                torch.save(dict(state_dict=model.state_dict(),kind=kind,widths=widths,epoch=epoch+1),directory/'best.pt')
            else:stale+=1
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,stale=stale,
                            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),directory/'last.pt')
            print(kind,record,flush=True)
            if stale>=3:break
        save_json(directory/'selection.json',dict(model=kind,validation_macro_f1=best[0],validation_ap=best[1],
                  parameters=sum(p.numel() for p in model.parameters()),epochs_run=len(history),
                  artifact_bytes=(directory/'best.pt').stat().st_size,checkpoint=str((directory/'best.pt').relative_to(PROJECT))))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]);axes[0].set(xlabel='Epoch',ylabel='Training objective')
        axes[1].plot([r['epoch'] for r in history],[r['mean_validation_macro_f1'] for r in history]);axes[1].set(xlabel='Epoch',ylabel='Mean validation macro-F1')
        fig.suptitle(kind);fig.tight_layout();fig.savefig(directory/'learning_curves.png',dpi=160);plt.close(fig)
        checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
        for code,d in enumerate(domains):
            target=directory/'test'/d['name']
            if (target/'source_rows.npy').exists() and (target/'decision_score.npy').exists():continue
            save_json(OUT/'linear_progress.json',dict(stage='test_evaluation',model=kind,dataset=d['name']))
            rows=d['indices'][2];score=predict(model,d,code,rows,device)
            export_evaluation(d['y'][rows],margin_probability(score),rows,target,f'{kind} — {d["name"]}',
                              raw_score=score,probability_note=config['probability'])
        save_json(directory/'complete.json',dict(model=kind,test_datasets=4,checkpoint_epoch=checkpoint['epoch']))
        del model,optimizer
        if device.type=='cuda':torch.cuda.empty_cache()
    save_json(OUT/'linear_progress.json',dict(stage='complete',models=['LR','SVM'],test_evaluations=8))


if __name__=='__main__':main()
