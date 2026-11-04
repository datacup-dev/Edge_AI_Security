"""Full mixed-population histogram gradient boosting with training-only cuts."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_adapted import load_domains,save_json
from classical_data import layout,sparse_batch
from extended_evaluation import export_evaluation,margin_probability

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts/expanded_classical/GB'


class NativeIterator(xgb.DataIter):
    def __init__(self,domains,batch_size=50000):
        super().__init__(release_data=True)
        self.domains=domains;self.batch_size=batch_size
        _,self.offsets,self.dimension=layout(domains);self.reset()
    def reset(self):
        self.code=0;self.position=0;self.keep=None
    def next(self,input_data):
        while self.code<len(self.domains):
            d=self.domains[self.code];rows=d['indices'][0]
            if self.position>=len(rows):self.code+=1;self.position=0;continue
            selected=rows[self.position:self.position+self.batch_size]
            block=np.asarray(d['x'][selected]);self.keep=sparse_batch(block,self.code,self.offsets,self.dimension)
            input_data(data=self.keep,label=d['y'][selected])
            self.position+=len(selected)
            return True
        return False


class Progress(xgb.callback.TrainingCallback):
    def __init__(self,started,base,history):
        self.started=started;self.base=base;self.history=history
    def after_iteration(self,model,epoch,evals_log):
        round_number=self.base+epoch+1
        record=dict(round=round_number,train_log_loss=float(evals_log['train']['logloss'][-1]),
                    elapsed_seconds=time.perf_counter()-self.started)
        self.history.append(record)
        if round_number%10==0:
            save_json(OUT/'history.json',self.history)
            save_json(OUT.parent/'gb_progress.json',dict(stage='boosting',total_rounds=300,**record))
        if round_number%25==0:model.save_model(OUT/'partial.ubj')
        return False


def predict(model,domains,code,rows):
    _,offsets,dimension=layout(domains);d=domains[code]
    score=np.empty(len(rows),np.float32)
    for start in range(0,len(rows),50000):
        selected=rows[start:start+50000]
        block=sparse_batch(np.asarray(d['x'][selected]),code,offsets,dimension)
        matrix=xgb.DMatrix(block,nthread=4)
        score[start:start+len(selected)]=model.predict(matrix,output_margin=True)
    return score


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():print('GB already complete',flush=True);return
    domains=load_domains();n=sum(len(d['indices'][0]) for d in domains)
    parameters=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',device='cuda',
                    eta=.1,max_depth=8,max_bin=128,subsample=1.,colsample_bytree=.8,seed=42,nthread=4)
    config=dict(model='GB',variant='XGBoost histogram gradient-boosted trees',xgboost=xgb.__version__,
                boosting_rounds=300,parameters=parameters,training_rows=n,features=157,
                input_adapter='Full native namespace; observed active zeros are explicit CSR entries, unavailable domain blocks are missing',
                quantiles='Built from training iterator only; no validation/test fitting',
                validation='Four separate validation sets; mean macro-F1/AP; fixed 300 rounds, no test tuning',threshold=.5)
    save_json(OUT/'configuration.json',config)
    started=time.perf_counter()
    if (OUT/'model.ubj').exists():
        model=xgb.Booster();model.load_model(OUT/'model.ubj');model.set_param({'device':'cuda','nthread':4})
        history=json.loads((OUT/'history.json').read_text())
    else:
        save_json(OUT.parent/'gb_progress.json',dict(stage='training_quantile_construction',training_rows=n))
        training=xgb.QuantileDMatrix(NativeIterator(domains),max_bin=128,nthread=4)
        assert training.num_row()==n and training.num_col()==157
        base=0;resume=None;history=[]
        if (OUT/'partial.ubj').exists():
            resume=xgb.Booster();resume.load_model(OUT/'partial.ubj');base=resume.num_boosted_rounds()
            if (OUT/'history.json').exists():history=[r for r in json.loads((OUT/'history.json').read_text()) if r['round']<=base]
        callback=Progress(started,base,history)
        try:
            model=xgb.train(parameters,training,num_boost_round=300-base,evals=[(training,'train')],
                            xgb_model=resume,callbacks=[callback],verbose_eval=25)
        except xgb.core.XGBoostError as error:
            if not any(s in str(error).lower() for s in ['memory','bad_alloc']):raise
            gc.collect();parameters['device']='cpu';config['fallback_reason']=str(error)
            config['parameters']=parameters;save_json(OUT/'configuration.json',config)
            callback=Progress(started,base,history[:base])
            model=xgb.train(parameters,training,num_boost_round=300-base,evals=[(training,'train')],
                            xgb_model=resume,callbacks=[callback],verbose_eval=25)
            history=callback.history
        assert model.num_boosted_rounds()==300
        model.save_model(OUT/'model.ubj');save_json(OUT/'history.json',callback.history)
        history=callback.history
        del training;gc.collect()
    (OUT/'booster_full_configuration.json').write_text(model.save_config(),encoding='utf-8')
    validation={}
    for code,d in enumerate(domains):
        rows=d['indices'][1];score=predict(model,domains,code,rows)
        validation[d['name']]=dict(macro_f1=float(f1_score(d['y'][rows],score>=0,average='macro',zero_division=0)),
                                  average_precision=float(average_precision_score(d['y'][rows],score)))
    selection=dict(model='GB',validation_macro_f1=float(np.mean([v['macro_f1'] for v in validation.values()])),
                   validation_ap=float(np.mean([v['average_precision'] for v in validation.values()])),validation=validation,
                   training_rows=n,rounds=300,artifact_bytes=(OUT/'model.ubj').stat().st_size,
                   checkpoint=str((OUT/'model.ubj').relative_to(PROJECT)),actual_device=parameters['device'])
    save_json(OUT/'selection.json',selection);print('Frozen GB candidate',selection,flush=True)
    fig,axis=plt.subplots(figsize=(7,4));axis.plot([r['round'] for r in history],[r['train_log_loss'] for r in history])
    axis.set(xlabel='Boosting round',ylabel='Training log loss',title='GB full-data training');fig.tight_layout()
    fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    for code,d in enumerate(domains):
        target=OUT/'test'/d['name']
        if (target/'source_rows.npy').exists() and (target/'decision_score.npy').exists():continue
        save_json(OUT.parent/'gb_progress.json',dict(stage='test_evaluation',dataset=d['name']))
        rows=d['indices'][2];score=predict(model,domains,code,rows)
        export_evaluation(d['y'][rows],margin_probability(score),rows,target,f'GB — {d["name"]}',raw_score=score,
                          probability_note='Sigmoid of binary-logistic boosted-tree margin')
    save_json(OUT/'complete.json',dict(model='GB',test_datasets=4))
    save_json(OUT.parent/'gb_progress.json',dict(stage='complete',model='GB',test_evaluations=4))


if __name__=='__main__':main()
