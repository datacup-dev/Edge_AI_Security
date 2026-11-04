"""Shared held-out reporting; preserve raw-margin ranking when appropriate."""
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import evaluate_predictions
from report import confusion_figure


def margin_probability(score):
    score=np.asarray(score)
    p=expit(score.astype(np.float64))
    p[(score<0)&(p>=.5)]=np.nextafter(.5,0.)
    return p


def export_evaluation(y,probability,source_rows,target,title,raw_score=None,probability_note=None):
    target.mkdir(parents=True,exist_ok=True)
    metrics=evaluate_predictions(y,probability,target)
    np.save(target/'source_rows.npy',source_rows)
    if raw_score is not None:
        raw_score=np.asarray(raw_score);assert np.isfinite(raw_score).all()
        np.save(target/'decision_score.npy',raw_score)
        metrics['roc_auc']=float(roc_auc_score(y,raw_score))
        metrics['average_precision']=float(average_precision_score(y,raw_score))
        metrics['ranking_score']='raw decision score, avoiding sigmoid saturation ties'
        fpr,tpr,_=roc_curve(y,raw_score);precision,recall,_=precision_recall_curve(y,raw_score)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(fpr,tpr);axes[0].plot([0,1],[0,1],'--',color='gray')
        axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC')
        axes[1].plot(recall,precision);axes[1].axhline(float(np.mean(y)),ls='--',color='gray')
        axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall')
        fig.suptitle(title);fig.tight_layout();fig.savefig(target/'roc_pr.png',dpi=160);plt.close(fig)
    if probability_note:metrics['probability_note']=probability_note
    (target/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    confusion_figure(metrics,target/'confusion_matrix.png',title)
    return metrics
