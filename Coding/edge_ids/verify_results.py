"""Verify completed experiment artifacts against original split assignments."""
import json
from pathlib import Path

import numpy as np


def main():
    project = Path(__file__).resolve().parent
    experiment = project/'artifacts/adapted_experiment'
    progress = json.loads((experiment/'progress.json').read_text())
    assert progress['stage'] == 'complete'
    baseline = json.loads((experiment/'baseline.json').read_text())
    assert baseline['selection_uses_test'] is False
    best = max(baseline['candidates'], key=lambda c:(c['validation_macro_f1'],c['validation_ap'],-c['parameters']))
    assert best == baseline['selected']
    results = []
    for candidate in baseline['candidates']:
        kind = candidate['model']
        history = json.loads((experiment/kind/'history.json').read_text())
        assert all(r['training_rows'] == 35030746 for r in history)
        assert len(history) == candidate['epochs_run']
        assert max(r['mean_validation_macro_f1'] for r in history) == candidate['validation_macro_f1']
        for name in ['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']:
            target = experiment/kind/'test'/name
            metrics = json.loads((target/'metrics.json').read_text())
            rows = np.load(target/'source_rows.npy')
            with np.load(project/'artifacts/native_splits'/f'{name}.npz') as original:
                assert np.array_equal(rows, np.flatnonzero(original['split'] == 2))
                expected_labels = original['labels'][rows]
            with np.load(target/'predictions.npz') as prediction:
                y, probability, predicted = (prediction[k] for k in ['labels','probability','predicted'])
                assert np.array_equal(y, expected_labels)
                assert probability.shape == y.shape == predicted.shape
                assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
                assert np.array_equal(predicted, probability >= metrics['threshold'])
                matrix = np.bincount(y.astype(np.uint8)*2 + predicted, minlength=4)
            assert matrix.tolist() == [metrics[k] for k in ['tn','fp','fn','tp']]
            assert sum(matrix) == metrics['n'] == len(rows)
            assert np.isclose((matrix[0]+matrix[3])/len(rows), metrics['accuracy'])
            assert (target/'confusion_matrix.png').is_file() and (target/'roc_pr.png').is_file()
            results.append(dict(model=kind, dataset=name, rows=len(rows), status='verified'))
    assert len(results) == 20
    report = dict(status='passed', baseline=best['model'], evaluations=results,
                  checks=['Exact original test membership and labels',
                          'Saved probabilities, threshold and confusion counts agree',
                          'All completed epochs use all mixed training rows',
                          'Baseline ranking uses validation scores only',
                          'All expected confusion and ROC/PR figures exist'])
    (experiment/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Verified all 20 test evaluations and validation-only baseline selection',flush=True)


if __name__ == '__main__':
    main()
