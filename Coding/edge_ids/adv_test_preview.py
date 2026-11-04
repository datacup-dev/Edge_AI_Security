"""Read-only preview of independent held-out attack files."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT=Path(__file__).resolve().parent


def main():
    path=PROJECT/'preview.ipynb';book=nbformat.read(path,as_version=4)
    book.cells=[c for c in book.cells if not c.get('id','').startswith('separate-test-attacks-')]
    cells=[nbformat.v4.new_markdown_cell('# Separate held-out adversarial test datasets\n\n'
        'Fifteen methods × four original test sets, both source classes. Each test output is separate from '
        'training outputs and from other datasets. Proxy, GAN, preprocessing and attack settings are frozen; '
        'no test fitting occurs. [Test process](TEST_ATTACKS.md). [Live results](TEST_ATTACK_REPORT.md). '
        'Re-execute these cells to refresh progress.',id='separate-test-attacks-heading'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display\n"
            "test_project=Path.cwd()\nif not (test_project/'artifacts').exists(): test_project=test_project/'Coding'/'edge_ids'\n"
            "test_root=test_project/'artifacts'/'separate_test_attacks'\n"
            "display(json.loads((test_root/'status.json').read_text()))\n"
            "display(pd.read_csv(test_root/'progress.csv'))\n"
            "display(json.loads((test_root/'configuration.json').read_text())['clean_prediction_consistency'])\n"
            "for audit in ['partial_audit.json','full_audit.json']:\n"
            "    if (test_root/audit).exists(): display(json.loads((test_root/audit).read_text()))",id='separate-test-attacks-progress'),
        nbformat.v4.new_code_cell("test_method='FGSM'\ntest_dataset='Edge-IIoTset'\n"
            "test_files=sorted((test_root/test_method/test_dataset/'shards').glob('*.npz'))\n"
            "if test_files:\n"
            "    with np.load(test_files[0]) as z:\n"
            "        print('Test-only file:',test_files[0])\n"
            "        print('Source split:',str(z['source_split']),'(mixed training positions must be -1)')\n"
            "        assert np.all(z['mixed_position']==-1)\n"
            "        choices=np.flatnonzero(z['success'])\n"
            "        i=int(choices[0]) if len(choices) else 0\n"
            "        display({k:z[k][i].item() for k in ['source_dataset_id','source_row','cohort_position','labels','target_labels','eligible','changed','success','clean_probability','adversarial_probability','linf']})\n"
            "else:\n    print('First shard pending.')",id='separate-test-attacks-example')]
    fragment=nbformat.v4.new_notebook(cells=cells,metadata=book.metadata.copy())
    NotebookClient(fragment,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(PROJECT)}}).execute()
    book.cells.extend(fragment.cells);nbformat.validate(book);nbformat.write(book,path);print('Separate test attack notebook section executed',flush=True)


if __name__=='__main__':main()
