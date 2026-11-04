"""Refresh the read-only experiment notebook after the expanded comparison."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT=Path(__file__).resolve().parent


def main(execute=True):
    path=PROJECT/'preview.ipynb';book=nbformat.read(path,as_version=4)
    book.cells=[c for c in book.cells if not c.get('id','').startswith('expanded-')]
    if not any(c.get('id')=='initial-phase-label' for c in book.cells):
        book.cells.insert(0,nbformat.v4.new_markdown_cell(
            '# Initial five-model phase (historical)\n\nThe original results below are preserved. '
            'The expanded comparison and current overall baseline appear in the final section.',id='initial-phase-label'))
    cells=[
        ('markdown','expanded-heading',
         '# Expanded comparison and overall baseline\n\n'
         'Eleven new candidates are compared with the previous MLP using validation scores only. '
         'Native observation units differ; these are authorized dataset-specific pipelines, not harmonized flows. '
         'See [Expanded methods](EXTENDED_EXPERIMENT_METHOD.md) and [Full results](EXTENDED_EXPERIMENT_REPORT.md).'),
        ('code','expanded-status',
         "extended = artifacts / 'extended_report'\n"
         "status = json.loads((extended / 'selection_status.json').read_text())\n"
         "display({'status': status['status'], 'pending': status.get('pending', []), 'selection_uses_test': status.get('selection_uses_test', False)})\n"
         "for progress_path in [extended / 'remaining_progress.json', artifacts / 'expanded_classical' / 'cpu_progress.json', artifacts / 'expanded_classical' / 'gb_progress.json']:\n"
         "    if progress_path.exists(): display(json.loads(progress_path.read_text()))\n"
         "display(pd.read_csv(extended / 'validation_comparison.csv'))\n"
         "overall_path = artifacts / 'overall_baseline.json'\n"
         "if overall_path.exists():\n"
         "    overall = json.loads(overall_path.read_text())\n"
         "    display({k: overall[k] for k in ['status', 'rule', 'scope']})\n"
         "    display(pd.DataFrame([{k: c.get(k) for k in ['model', 'validation_macro_f1', 'validation_ap', 'artifact_bytes']} for c in [overall['best_new'], overall['previous_MLP']]]))"),
        ('code','expanded-metrics',
         "display(pd.read_csv(extended / 'all_test_metrics.csv'))\n"
         "display(Image(filename=str(extended / 'test_macro_f1.png')))"),
        ('code','expanded-figures',
         "if overall_path.exists():\n"
         "    selected = json.loads(overall_path.read_text())['selected']\n"
         "    print('Overall baseline:', selected['model'])\n"
         "    for domain in ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']:\n"
         "        print(domain)\n"
         "        for figure in ['confusion_matrix.png', 'roc_pr.png']:\n"
         "            display(Image(filename=str(project / selected['test_directory'] / domain / figure)))\n"
         "for verification in ['verification.json', 'loader_verification.json']:\n"
         "    check = extended / verification\n"
         "    if check.exists(): display(json.loads(check.read_text()))"),
        ('markdown','expanded-limits',
         'The test sets were inspected in the initial stage and are reused here. Random record splits do not '
         'isolate captures/devices/duplicates/time. GNN nodes represent feature groups, transformer tokens '
         'are not temporal observations, and the GAN is a classifier rather than an adversarial defense. '
         'No adversarial robustness, explanation quality or end-to-end real-time edge performance is established.'),
    ]
    for kind,identifier,source in cells:
        make=nbformat.v4.new_code_cell if kind=='code' else nbformat.v4.new_markdown_cell
        book.cells.append(make(source,id=identifier))
    nbformat.validate(book);nbformat.write(book,path)
    if execute:
        client=NotebookClient(book,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(PROJECT)}})
        client.execute();nbformat.write(book,path)
        assert not any(o.get('output_type')=='error' for c in book.cells if c.cell_type=='code' for o in c.outputs)
    packages=sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()},key=str.lower)
    (PROJECT/'requirements-expanded-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu128\n'+'\n'.join(packages)+'\n',encoding='utf-8')
    import torch
    import xgboost
    environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        xgboost=xgboost.__version__,xgboost_build=xgboost.build_info())
    (PROJECT/'artifacts/extended_report/environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print('Expanded notebook refreshed; executed:',execute,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-execute',action='store_true')
    main(execute=not parser.parse_args().no_execute)
