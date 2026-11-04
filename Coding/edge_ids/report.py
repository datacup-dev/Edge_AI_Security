"""Create a readable preview and aggregate completed model results."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT / 'artifacts'


def confusion_figure(metrics, path, title):
    matrix = np.array([[metrics['tn'],metrics['fp']], [metrics['fn'],metrics['tp']]])
    normalized = matrix / np.maximum(matrix.sum(axis=1,keepdims=True),1)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for axis, values, label in zip(axes,[matrix,normalized],['Counts','Normalized by actual class']):
        axis.imshow(values,cmap='Blues')
        for i in range(2):
            for j in range(2):
                text = f'{values[i,j]:,}' if label=='Counts' else f'{values[i,j]:.2%}'
                axis.text(j,i,text,ha='center',va='center',
                          color='white' if values[i,j]>values.max()/2 else 'black')
        axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Benign','Attack'],
                 yticklabels=['Benign','Attack'],xlabel='Predicted',ylabel='Actual',title=label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    output = ARTIFACTS / 'report'
    output.mkdir(parents=True, exist_ok=True)
    split_meta = [json.loads(p.read_text()) for p in sorted((ARTIFACTS/'native_splits').glob('*.json'))]
    names = [m['dataset'] for m in split_meta]
    benign = [m['counts']['train']['benign'] for m in split_meta]
    attack = [m['counts']['train']['attack'] for m in split_meta]
    fig, axis = plt.subplots(figsize=(9,4))
    position = np.arange(len(names))
    axis.bar(position-.18, benign, .36, label='Benign')
    axis.bar(position+.18, attack, .36, label='Attack')
    axis.set(xticks=position, xticklabels=names, yscale='log', ylabel='Training records (log scale)',
             title='Natural training proportions; no class/domain balancing')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output/'training_composition.png', dpi=160)
    plt.close(fig)
    rows = []
    for path in sorted((ARTIFACTS/'adapted_experiment').glob('*/test/*/metrics.json')):
        metrics = json.loads(path.read_text())
        tn, fp, fn, tp = (metrics[k] for k in ['tn','fp','fn','tp'])
        metrics.update(support_benign=tn+fp, support_attack=tp+fn,
                       precision_benign=tn/(tn+fn) if tn+fn else 0.0,
                       recall_benign=tn/(tn+fp) if tn+fp else 0.0,
                       f1_benign=2*tn/(2*tn+fn+fp) if 2*tn+fn+fp else 0.0)
        path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        model, dataset = path.parents[2].name, path.parent.name
        rows.append(dict(model=model, dataset=dataset, **metrics))
        confusion_figure(metrics,path.parent/'confusion_matrix.png',f'{model} — {dataset}')
    references = [dict(dataset=p.parent.name, **json.loads(p.read_text()))
                  for p in sorted((ARTIFACTS/'adapted_experiment'/'domain_prior_reference').glob('*/metrics.json'))]
    if references:
        with (output/'domain_prior_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(references[0]))
            writer.writeheader()
            writer.writerows(references)
    if rows:
        with (output/'test_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fig, axis = plt.subplots(figsize=(12,4))
        models = sorted({r['model'] for r in rows})
        for i, model in enumerate(models):
            values = [next((r['f1_macro'] for r in rows if r['model']==model and r['dataset']==n), np.nan) for n in names]
            axis.bar(position+(i-(len(models)-1)/2)*.15, values, .15, label=model)
        if len(references) == len(names):
            axis.plot(position, [next(r['f1_macro'] for r in references if r['dataset']==n) for n in names],
                      'k--o', label='Domain-prior reference')
        axis.set(xticks=position, xticklabels=names, ylim=(0,1), ylabel='Test macro-F1',
                 title='Separate test evaluation; baseline selected on validation only')
        axis.legend(loc='upper left',bbox_to_anchor=(1.01,1),ncol=1)
        fig.tight_layout()
        fig.savefig(output/'test_comparison.png', dpi=160)
        plt.close(fig)
    progress_file = ARTIFACTS/'adapted_experiment'/'progress.json'
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {'stage':'not started'}
    baseline_file = ARTIFACTS/'adapted_experiment'/'baseline.json'
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else None
    lines = ['# Adapted mixed-domain experiment report', '',
             'Full workflow and audit record: [Complete process documentation](FULL_PROCESS_DOCUMENTATION.md).', '',
             'User-authorized dataset-specific encoders; not the paper’s shared NetFlow representation.', '',
             f"Current stage: **{progress['stage']}**. Completed model/test evaluations: {len(rows)} of 20.", '',
             '![Training composition](artifacts/report/training_composition.png)', '',
             'All training rows are used at natural proportions. Select checkpoints and the baseline using validation only. '
             'Recurrent variants process feature tokens, not time. See [methods](EXPERIMENT_METHOD.md).', '']
    if baseline:
        lines += [f"Selected provisional baseline: **{baseline['selected']['model']}**.", '',
                  '| Model | Mean validation macro-F1 | Mean validation AP | Parameters | Epochs |',
                  '|---|---:|---:|---:|---:|']
        for candidate in baseline['candidates']:
            lines.append(f"| {candidate['model']} | {candidate['validation_macro_f1']:.6f} | {candidate['validation_ap']:.6f} | {candidate['parameters']:,} | {candidate['epochs_run']} |")
        lines += ['', 'The baseline was selected before test evaluation. Test scores do not determine this choice.', '']
    if rows:
        lines += ['| Model | Test dataset | Accuracy | Macro-F1 | Attack recall | False-positive rate | ROC-AUC | AP |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['model']} | {r['dataset']} | {r['accuracy']:.6f} | {r['f1_macro']:.6f} | {r['recall_attack']:.6f} | {r['false_positive_rate']:.6f} | {r['roc_auc']:.6f} | {r['average_precision']:.6f} |")
        lines += ['', '![Test macro-F1 comparison](artifacts/report/test_comparison.png)', '',
                  '[All recorded test metrics](artifacts/report/test_metrics.csv). '
                  'Per-model predictions, source rows and confusion/ROC/PR figures are in '
                  '`artifacts/adapted_experiment/<model>/test/<dataset>/`.', '']
        if baseline:
            selected_rows = {r['dataset']: r for r in rows if r['model']==baseline['selected']['model']}
            if 'DataSense' in selected_rows and 'CICIoT2023' in selected_rows:
                lines += [f"For the selected baseline, DataSense attack miss rate is {100*selected_rows['DataSense']['false_negative_rate']:.2f}%; "
                          f"CICIoT2023 benign false-positive rate is {100*selected_rows['CICIoT2023']['false_positive_rate']:.2f}%.", '']
    if references:
        lines += ['[Domain-prior reference metrics](artifacts/report/domain_prior_metrics.csv). '
                  'This reference uses training label prevalence and no input traffic features.', '']
    if (output/'inference_latency.csv').exists():
        lines += ['[Workstation inference timings](artifacts/report/inference_latency.csv). '
                  'Warm, preprocessed tensors only; excludes extraction, scaling, transfer, queueing '
                  'and window collection. This is not an end-to-end edge latency measurement.', '']
    if (ARTIFACTS/'adapted_experiment'/'verification.json').exists():
        lines += ['[Completed artifact audit](artifacts/adapted_experiment/verification.json): '
                  'all 20 model/test evaluations matched the original held-out memberships and labels.', '']
    lines += ['## Interpretation limits', '',
              'Native observation units remain different. Scenario labels and random row splits do not '
              'establish independence of captures, sessions, devices or duplicates. Dataset ID routes '
              'encoders; compare against the saved domain-prior reference. These results do not establish '
              'unseen-domain transfer, temporal modelling, adversarial robustness or real-time deployment.', '']
    (PROJECT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    markdown = (
        '# Adapted mixed-domain experiment preview\n\n'
        'Dataset-specific encoders are authorized. These native-record inputs are '
        'not harmonized flows. Recurrent variants process feature sequences, not time. '
        'CNN2D/3D are excluded because spatial axes are unavailable.\n\n'
        'All training records are used at natural proportions. Model selection uses '
        'the mean macro-F1 across four validation sets. Test sets remain separate. '
        'See EXPERIMENT_METHOD.md for assumptions and limitations.'
    )
    def cell(kind, source):
        value = dict(cell_type=kind, id=hashlib.sha256(source.encode()).hexdigest()[:12],
                     metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value
    cells = [cell('markdown', markdown), cell('code',
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\n"
        "project = Path.cwd()\nif not (project / 'artifacts').exists():\n    project = project / 'Coding' / 'edge_ids'\n"
        "artifacts = project / 'artifacts'\n"
        "configuration = artifacts / 'adapted_experiment' / 'configuration.json'\n"
        "if configuration.exists():\n    config = json.loads(configuration.read_text())\n"
        "    display({k: v for k, v in config.items() if k != 'feature_group_assignment'})\n"
        "progress = artifacts / 'adapted_experiment' / 'progress.json'\n"
        "display(json.loads(progress.read_text()) if progress.exists() else {'stage': 'not started'})\n"),
        cell('code', "display(Image(filename=str(artifacts / 'report' / 'training_composition.png')))\n"),
        cell('code',
        "files = sorted((artifacts / 'adapted_experiment').glob('*/history.json'))\n"
        "history = [{'model': f.parent.name, **r} for f in files for r in json.loads(f.read_text())]\n"
        "display(pd.DataFrame(history) if history else 'No completed training epochs yet.')\n"),
        cell('code',
        "for plot in sorted((artifacts / 'adapted_experiment').glob('*/learning_curves.png')):\n"
        "    print(plot.parent.name)\n    display(Image(filename=str(plot)))\n"),
        cell('code',
        "baseline = artifacts / 'adapted_experiment' / 'baseline.json'\n"
        "display(json.loads(baseline.read_text()) if baseline.exists() else 'Baseline not selected yet.')\n"
        "metrics = artifacts / 'report' / 'test_metrics.csv'\n"
        "display(pd.read_csv(metrics) if metrics.exists() else 'Test evaluation not complete.')\n"),
        cell('code',
        "comparison = artifacts / 'report' / 'test_comparison.png'\n"
        "if comparison.exists():\n    display(Image(filename=str(comparison)))\n"),
        cell('code',
        "latency = artifacts / 'report' / 'inference_latency.csv'\n"
        "if latency.exists():\n    display(pd.read_csv(latency))\n"
        "    print('Warm workstation inference only; extraction, scaling, transfer, queueing and window delay excluded.')\n"),
        cell('code',
        "if baseline.exists():\n"
        "    selected = json.loads(baseline.read_text())['selected']['model']\n"
        "    for folder in sorted((artifacts / 'adapted_experiment' / selected / 'test').glob('*')):\n"
        "        print(selected, folder.name)\n"
        "        for name in ['confusion_matrix.png', 'roc_pr.png']:\n"
        "            if (folder / name).exists():\n                display(Image(filename=str(folder / name)))\n"
        "else:\n    print('Baseline plots pending; all model plots will be saved in artifacts.')\n")]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python (edge_ids .venv)', language='python',name='python3')),
                    nbformat=4, nbformat_minor=5)
    (PROJECT/'preview.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(f'Report refreshed: {len(rows)} completed test evaluations', flush=True)


if __name__ == '__main__':
    main()
