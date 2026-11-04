"""Export verified portable model checkpoints and exact training-file inventory."""
import json
import hashlib
import shutil
import time
import zipfile
from pathlib import Path
import xgboost as xgb
from augmented_training import PROJECT,ATTACKS,write

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    destination=PROJECT/'migration'/time.strftime('%Y%m%d_%H%M%S')
    bundle=destination/'edge_ids';bundle.mkdir(parents=True)
    required=set();models=[]
    def copy(relative):
        source=PROJECT/relative;target=bundle/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for p in PROJECT.glob('*.py'):copy(p.relative_to(PROJECT))
    for name in ['SERVER_MIGRATION.md','LAB_SERVER_SETUP.md','run_lab_server.sh','run_lab_gpu.sh','ADVERSARIAL_DEFENSE_TRAINING.md','requirements-server.txt','requirements-server-gpu.txt']:
        copy(Path(name))
    copy(Path('artifacts/expanded_classical/GB/configuration.json'))
    for attack in ATTACKS:
        manifest_path=Path('artifacts/augmented_training')/attack/'manifest.json'
        copy(manifest_path)
        manifest=json.loads((PROJECT/manifest_path).read_text())
        required.add(manifest['clean_records'])
        required.update(s['path'] for s in manifest['shards'])
        folder=Path('artifacts/adversarial_defenses')/attack
        for name in ['training_manifest.json','configuration.json']:
            if (PROJECT/folder/name).exists():copy(folder/name)
        checkpoint='model.ubj' if (PROJECT/folder/'complete.json').exists() else 'partial.ubj'
        source=PROJECT/folder/checkpoint
        if source.exists():
            # Copy only a stable source and verify the portable model loads.
            for attempt in range(10):
                before=sha(source);copy(folder/checkpoint)
                if before==sha(source)==sha(bundle/folder/checkpoint):break
                time.sleep(.2)
            else:raise RuntimeError(f'Checkpoint changing: retry export for {attack}')
            model=xgb.Booster();model.load_model(bundle/folder/checkpoint)
            models.append(dict(attack=attack,checkpoint=checkpoint,rounds=model.num_boosted_rounds(),sha256=before))
            if checkpoint=='model.ubj':copy(folder/'complete.json')
            # History is optional; truncate to verified checkpoint round on resume.
            if (PROJECT/folder/'history.json').exists():copy(folder/'history.json')
    from augmented_training import NAMES
    for name in NAMES:required.add(f'artifacts/encoded_inputs/{name}/features.npy')
    entries=[]
    for relative in sorted(required):
        path=PROJECT/relative
        assert path.is_file(),f'Missing required training data: {path}'
        entries.append(dict(path=relative,bytes=path.stat().st_size))
    write(bundle/'migration_inventory.json',dict(models=models,files=entries,total_data_bytes=sum(e['bytes'] for e in entries),
        note='Exact frozen defense training inputs; not latest attack-generation state. Size inventory is not a content checksum.'))
    (bundle/'transfer-files.txt').write_text('\n'.join(e['path'] for e in entries)+'\n',encoding='utf-8')
    (bundle/'.gitignore').write_text('.venv/\n__pycache__/\nartifacts/\nmigration/\n*.log\n',encoding='utf-8')
    with zipfile.ZipFile(destination/'code_and_checkpoints.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(destination))
    print(json.dumps(dict(bundle=str(destination),models=models,data_bytes=sum(e['bytes'] for e in entries),data_files=len(entries)),indent=2),flush=True)

if __name__=='__main__':main()
