"""Add an executed validation-only preview without changing existing sections."""
import nbformat
from nbclient import NotebookClient
from adv_mixed_data import PROJECT

path=PROJECT/'preview.ipynb'
book=nbformat.read(path,as_version=4)
book.cells=[c for c in book.cells if not c.get('id','').startswith('separate-validation-attacks-')]
cells=[nbformat.v4.new_markdown_cell(
    '# Separate validation adversarial samples\n\nAll four validation datasets and both classes, with 15 separate attack outputs. '
    'Frozen training-fitted components; no validation fitting. Generation is resumable and incomplete until the full audit passes. '
    '[Protocol](VALIDATION_ATTACKS.md) · [Live report](VALIDATION_ATTACK_REPORT.md)',
    id='separate-validation-attacks-heading'),
    nbformat.v4.new_code_cell(
    "from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\n"
    "validation_project=Path.cwd()\n"
    "if not (validation_project/'artifacts').exists(): validation_project=validation_project/'Coding'/'edge_ids'\n"
    "validation_root=validation_project/'artifacts'/'separate_validation_attacks'\n"
    "display(json.loads((validation_root/'status.json').read_text()))\n"
    "display(pd.read_csv(validation_root/'progress.csv'))\n"
    "display(json.loads((validation_root/'configuration.json').read_text())['cohorts'])\n"
    "files=sorted((validation_root/'FGSM'/'Edge-IIoTset'/'shards').glob('*.npz'))\n"
    "if files:\n"
    "    with np.load(files[0]) as z:\n"
    "        assert str(z['source_split'])=='validation'\n"
    "        assert np.all(z['mixed_position']==-1)\n"
    "        print('Validation-only example:',files[0], 'rows:',len(z['labels']))\n",
    id='separate-validation-attacks-progress')]
fragment=nbformat.v4.new_notebook(cells=cells,metadata=book.metadata.copy())
NotebookClient(fragment,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(PROJECT)}}).execute()
book.cells.extend(fragment.cells)
nbformat.validate(book)
nbformat.write(book,path)
print('Validation preview executed')
