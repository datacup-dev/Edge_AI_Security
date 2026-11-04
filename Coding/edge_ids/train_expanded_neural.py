"""Train four additional full-data pipelines; freeze ranking before new test use."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.metrics import f1_score,average_precision_score
from train_adapted import load_domains,save_json
from expanded_models import ExpandedModel,GRAPH_EDGES
from evaluate import evaluate_predictions
from report import confusion_figure

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts/expanded_neural'
KINDS=['Autoencoder','GNN','Transformer','GAN']


@torch.inference_mode()
def predict(model,domain,code,rows,device):
    model.eval()
    p=np.empty(len(rows),np.float32)
    for start in range(0,len(rows),8192):
        selected=rows[start:start+8192]
        x=torch.from_numpy(np.asarray(domain['x'][selected])).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(x,code)
        p[start:start+len(selected)]=torch.sigmoid(logits.float()).cpu().numpy()
    return p


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domains=load_domains()
    features=[d['features'] for d in domains]
    config=dict(models=KINDS,seed=42,optimizer='Adam',learning_rate=.001,dropout=.3,
                effective_batch_size=8192,physical_microbatch=8192,epoch_limit=20,patience=3,
                selection='Mean of four validation macro-F1 scores; mean AP breaks checkpoint ties',
                threshold=.5,training_rows=sum(len(d['indices'][0]) for d in domains),
                representation='Same trained domain feature-group encoders as the MLP architecture',
                autoencoder='Supervised bottleneck classifier plus 0.1 native-feature reconstruction MSE',
                graph='Eight feature groups, undirected prior relation edges; not a device/flow topology graph',graph_edges=GRAPH_EDGES,
                transformer=dict(d_model=64,heads=4,layers=2,ffn=128,dropout=.3),
                GAN='Conditional native-feature generator plus benign/attack/fake discriminator; not an adversarial network attack',
                GAN_losses='D=real CE+0.5 fake CE; G=conditional benign/attack CE; inference attack logit=D1-D0',
                test_policy='Freeze each candidate checkpoint on validation before its tests; overall ranking reads validation scores only',
                precision='CUDA float16 autocast with GradScaler and float32 parameters')
    save_json(OUT/'configuration.json',config)
    for kind in KINDS:
        directory=OUT/kind
        directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():
            print('Completed candidate reused',kind,flush=True)
            continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if device.type=='cuda':torch.cuda.manual_seed_all(42)
        model=ExpandedModel(kind,features).to(device)
        if kind=='GAN':
            generator_params=list(model.generators.parameters())
            generator_ids={id(p) for p in generator_params}
            discriminator_params=[p for p in model.parameters() if id(p) not in generator_ids]
            optimizer=torch.optim.Adam(discriminator_params,lr=.001)
            generator_optimizer=torch.optim.Adam(generator_params,lr=.001)
        else:
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        history=[];best=(-1.,-1.);stale=0
        if (directory/'last.pt').exists():
            state=torch.load(directory/'last.pt',map_location=device,weights_only=False)
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler']);history=state['history'];best=tuple(state['best']);stale=state['stale']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
            if kind=='GAN':generator_optimizer.load_state_dict(state['generator_optimizer'])
        for epoch in range(len(history),20 if not (directory/'selection.json').exists() else len(history)):
            started=time.perf_counter();rng=np.random.default_rng(42+epoch)
            indices=[rng.permutation(d['indices'][0]) for d in domains]
            batches=[(code,start) for code,rows in enumerate(indices) for start in range(0,len(rows),8192)]
            rng.shuffle(batches);model.train();seen=0;loss_sum=aux_sum=g_sum=0.
            for batch,(code,start) in enumerate(batches):
                rows=indices[code][start:start+8192];d=domains[code]
                x=torch.from_numpy(np.asarray(d['x'][rows])).to(device)
                y=torch.from_numpy(d['y'][rows].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    if kind=='GAN':
                        fake=model.generate(y.long(),code)
                        real_loss=F.cross_entropy(model.discriminate(x,code),y.long())
                        fake_loss=F.cross_entropy(model.discriminate(fake.detach(),code),torch.full_like(y.long(),2))
                        loss=real_loss+.5*fake_loss;aux=fake_loss.detach()
                    else:
                        loss,classification,aux=model.supervised_loss(x,y,code)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {kind}/{epoch}/{batch}')
                scaler.scale(loss).backward();scaler.step(optimizer)
                if kind=='GAN':
                    generator_optimizer.zero_grad(set_to_none=True)
                    for p in discriminator_params:p.requires_grad_(False)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        generated=model.generate(y.long(),code)
                        generator_loss=F.cross_entropy(model.discriminate(generated,code),y.long())
                    scaler.scale(generator_loss).backward();scaler.step(generator_optimizer)
                    for p in discriminator_params:p.requires_grad_(True)
                    g_sum+=float(generator_loss.detach())*len(rows)
                scaler.update();loss_sum+=float(loss.detach())*len(rows);aux_sum+=float(aux)*len(rows);seen+=len(rows)
                if batch%100==0:
                    save_json(OUT/'progress.json',dict(stage='training',model=kind,epoch=epoch+1,batch=batch+1,
                              batches=len(batches),rows_seen=seen,elapsed_seconds=time.perf_counter()-started))
            assert seen==config['training_rows']
            validation={}
            for code,d in enumerate(domains):
                rows=d['indices'][1];p=predict(model,d,code,rows,device);y=d['y'][rows]
                validation[d['name']]=dict(macro_f1=float(f1_score(y,p>=.5,average='macro',zero_division=0)),
                                          average_precision=float(average_precision_score(y,p)))
            score=(float(np.mean([v['macro_f1'] for v in validation.values()])),
                   float(np.mean([v['average_precision'] for v in validation.values()])))
            record=dict(epoch=epoch+1,training_rows=seen,train_loss=loss_sum/seen,auxiliary_loss=aux_sum/seen,
                        generator_loss=g_sum/seen,validation=validation,mean_validation_macro_f1=score[0],
                        mean_validation_ap=score[1],elapsed_seconds=time.perf_counter()-started)
            history.append(record);save_json(directory/'history.json',history)
            if score>best:
                best=score;stale=0
                torch.save(dict(state_dict=model.state_dict(),kind=kind,feature_lists=features,epoch=epoch+1),directory/'best.pt')
            else:stale+=1
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,
                       best=best,stale=stale,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            if kind=='GAN':state['generator_optimizer']=generator_optimizer.state_dict()
            torch.save(state,directory/'last.pt')
            print(kind,json.dumps(record),flush=True)
            if stale>=3:break
        save_json(directory/'selection.json',dict(model=kind,validation_macro_f1=best[0],validation_ap=best[1],
                  parameters=sum(p.numel() for p in model.parameters()),epochs_run=len(history),
                  checkpoint=str((directory/'best.pt').relative_to(PROJECT)),status='training complete; tests deferred'))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training objective')
        if kind=='Autoencoder':axes[0].plot([r['epoch'] for r in history],[r['auxiliary_loss'] for r in history],label='Reconstruction MSE')
        if kind=='GAN':axes[0].plot([r['epoch'] for r in history],[r['generator_loss'] for r in history],label='Generator CE')
        axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Loss')
        axes[1].plot([r['epoch'] for r in history],[r['mean_validation_macro_f1'] for r in history])
        axes[1].set(xlabel='Epoch',ylabel='Mean validation macro-F1');fig.suptitle(kind);fig.tight_layout()
        fig.savefig(directory/'learning_curves.png',dpi=160);plt.close(fig)
        selected=torch.load(directory/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(selected['state_dict'])
        for code,d in enumerate(domains):
            target=directory/'test'/d['name']
            if (target/'source_rows.npy').exists() and (target/'roc_pr.png').exists():continue
            save_json(OUT/'progress.json',dict(stage='test_evaluation',model=kind,dataset=d['name']))
            rows=d['indices'][2];p=predict(model,d,code,rows,device)
            metrics=evaluate_predictions(d['y'][rows],p,target)
            np.save(target/'source_rows.npy',rows)
            confusion_figure(metrics,target/'confusion_matrix.png',f'{kind} — {d["name"]}')
        save_json(directory/'complete.json',dict(model=kind,test_datasets=4,checkpoint_epoch=selected['epoch']))
        selection=json.loads((directory/'selection.json').read_text())
        selection['status']='training and four test evaluations complete'
        save_json(directory/'selection.json',selection)
        del model,optimizer,scaler
        if device.type=='cuda':torch.cuda.empty_cache()
    save_json(OUT/'progress.json',dict(stage='complete',models=KINDS,test_evaluations=16))
    print('Expanded neural training and 16 test evaluations complete',flush=True)


if __name__=='__main__':main()
