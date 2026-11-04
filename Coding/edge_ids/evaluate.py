"""Metrics and plots for one frozen model evaluated on one held-out dataset."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def evaluate_predictions(y, probability, output, threshold=0.5):
    y, probability = np.asarray(y), np.asarray(probability)
    if y.ndim != 1 or probability.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty aligned label/probability vectors')
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError('Labels must be binary')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid attack probabilities')
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    predicted = (probability >= threshold).astype(np.uint8)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    both_classes = len(np.unique(y)) == 2
    scores = dict(
        n=len(y), threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
        support_benign=tn+fp, support_attack=tp+fn,
        precision_benign=tn/(tn+fn) if tn+fn else 0.0,
        recall_benign=tn/(tn+fp) if tn+fp else 0.0,
        f1_benign=2*tn/(2*tn+fn+fp) if 2*tn+fn+fp else 0.0,
        accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        precision_attack=float(precision_score(y, predicted, zero_division=0)),
        recall_attack=float(recall_score(y, predicted, zero_division=0)),
        f1_attack=float(f1_score(y, predicted, zero_division=0)),
        f1_macro=float(f1_score(y, predicted, labels=[0, 1], average='macro', zero_division=0)),
        f1_weighted=float(f1_score(y, predicted, labels=[0, 1], average='weighted', zero_division=0)),
        mcc=float(matthews_corrcoef(y, predicted)),
        specificity=tn/(tn+fp) if tn+fp else None,
        false_positive_rate=fp/(tn+fp) if tn+fp else None,
        false_negative_rate=fn/(fn+tp) if fn+tp else None,
        roc_auc=float(roc_auc_score(y, probability)) if both_classes else None,
        average_precision=float(average_precision_score(y, probability)) if both_classes else None,
        log_loss=float(log_loss(y, probability, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, probability)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', labels=y, probability=probability,
                        predicted=predicted)
    (output / 'metrics.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha='center', va='center')
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Benign', 'Attack'],
             yticklabels=['Benign', 'Attack'], xlabel='Predicted', ylabel='Actual',
             title='Confusion matrix')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    if both_classes:
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC')
        axes[1].plot(recall, precision)
        axes[1].axhline(float(y.mean()), linestyle='--', color='gray')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
        fig.tight_layout()
        fig.savefig(output / 'roc_pr.png', dpi=160)
        plt.close(fig)
    return scores
