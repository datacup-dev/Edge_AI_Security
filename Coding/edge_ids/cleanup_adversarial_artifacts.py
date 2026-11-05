"""Delete only allowlisted adversarial-stage files; preserve clean IDS artifacts.

On the lab server: python3 cleanup_adversarial_artifacts.py --apply
Default mode prints the deletion plan without changing anything.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

PROJECT=Path(__file__).resolve().parent
ARTIFACT_DIRS=['mixed_training_attacks','separate_test_attacks','separate_validation_attacks',
    'adversarial_defenses','augmented_training','augmented_training_latest','augmented_test',
    'augmented_validation','server_attack_generation']
SCRIPTS=['adv_mixed_algorithms.py','adv_mixed_audit.py','adv_mixed_check.py','adv_mixed_control.py',
    'adv_mixed_data.py','adv_mixed_prepare.py','adv_mixed_run.py','adv_test_control.py','adv_test_preview.py',
    'adv_validation_control.py','adv_validation_preview.py','adv_validation_serial.py',
    'augmented_test.py','augmented_training.py','augmented_validation.py','check_defense_iterator.py',
    'download_training_data.py','gpu_resume_smoke.py','prepare_attack_server_inputs.py','prepare_server_migration.py',
    'publish_training_data.py','server_attack_generation.py','train_adversarial_gb.py','verify_server_migration.py']
DOCS=['ADVERSARIAL_DEFENSE_TRAINING.md','AUGMENTED_TEST.md','AUGMENTED_TRAINING.md','AUGMENTED_VALIDATION.md',
    'LAB_SERVER_SETUP.md','MIXED_TRAINING_ATTACKS.md','MIXED_TRAINING_ATTACK_REPORT.md',
    'SERVER_ATTACK_GENERATION.md','SERVER_MIGRATION.md','TEST_ATTACKS.md','TEST_ATTACK_REPORT.md',
    'TRAINING_DATA_TRANSFER.md','VALIDATION_ATTACKS.md','VALIDATION_ATTACK_REPORT.md']
OTHER=['requirements-mixed-attacks-lock.txt','requirements-server-attacks.txt','requirements-server-gpu.txt',
    'requirements-server.txt','run_lab_gpu.sh','run_lab_server.sh','attack_generation_inputs.json',
    'attack-generation-inputs.tar.gz','attack-generation-inputs.sha256','migration_inventory.json','transfer-files.txt']
PROTECTED=['adapted_experiment','classical_native','documentation','encoded_inputs','expanded_classical',
    'expanded_neural','extended_report','knn_index_development','mixed_training','native_splits',
    'pilot_feature_tokens','report']

def protected_hashes():
    paths=[PROJECT/'artifacts/overall_baseline.json',PROJECT/'artifacts/expanded_classical/GB/model.ubj']
    for name in PROTECTED:
        directory=PROJECT/'artifacts'/name
        if directory.exists():
            paths.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.pt','.ubj'])
    result={}
    for path in set(paths):
        if not path.exists():continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        result[path.relative_to(PROJECT).as_posix()]=h.hexdigest()
    return result

def targets():
    entries=[PROJECT/'artifacts'/n for n in ARTIFACT_DIRS]
    entries += [PROJECT/n for n in SCRIPTS+DOCS+OTHER]
    entries += [PROJECT/n for n in ['migration','training-data-download','.venv-attacks']]
    entries += [PROJECT/'github_upload'/n for n in ['Edge_AI_Security_code.zip','upload_manifest.json']]
    cache=PROJECT/'__pycache__'
    if cache.exists():entries.extend(p for p in cache.iterdir() if any(p.name.startswith(Path(n).stem+'.') for n in SCRIPTS))
    for p in entries:
        resolved=p.resolve()
        assert resolved!=PROJECT and PROJECT in resolved.parents,f'Target escapes project: {p}'
        assert not any((PROJECT/'artifacts'/n).resolve()==resolved for n in PROTECTED)
    return [p for p in entries if p.exists()]

def stop_linux_jobs():
    stopped=[]
    if os.name=='nt':return stopped
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            args=(process/'cmdline').read_bytes().split(b'\0')
            cwd=(process/'cwd').resolve()
            for raw in args[1:]:
                arg=raw.decode(errors='replace')
                if Path(arg).name not in SCRIPTS:continue
                script=Path(arg) if Path(arg).is_absolute() else cwd/arg
                if script.resolve()==PROJECT/Path(arg).name:
                    os.kill(int(process.name),signal.SIGTERM);stopped.append(int(process.name));break
        except (OSError,PermissionError):continue
    if stopped:
        time.sleep(2)
        for pid in stopped:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
    return stopped

def clean_notebook():
    path=PROJECT/'preview.ipynb'
    if not path.exists():return 0
    book=json.loads(path.read_text(encoding='utf-8'));before=len(book['cells'])
    prefixes=('both-class-attacks-','separate-test-attacks-','separate-validation-attacks-')
    book['cells']=[c for c in book['cells'] if not c.get('id','').startswith(prefixes)]
    if len(book['cells'])!=before:path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return before-len(book['cells'])

def main(apply=False):
    entries=targets()
    for p in entries:print(p.relative_to(PROJECT),flush=True)
    if not apply:
        print('Dry run only. Pass --apply for the requested deletion.');return
    before=protected_hashes();stopped=stop_linux_jobs();removed=[]
    for p in entries:
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
        removed.append(p.relative_to(PROJECT).as_posix())
        print('REMOVED',removed[-1],flush=True)
    cells=clean_notebook()
    # Delete only the known temporary export restored during this migration.
    temporary=Path('/tmp/edge_ids_restore')
    if os.name!='nt' and temporary.is_dir() and (temporary/'edge_ids/migration_inventory.json').is_file():
        assert temporary.resolve()==Path('/tmp/edge_ids_restore')
        if temporary.stat().st_uid==os.getuid():shutil.rmtree(temporary)
    cache=Path.home()/'edge_ids_cache'
    if os.name!='nt' and cache.is_dir() and not cache.is_symlink() and cache.stat().st_uid==os.getuid():
        for name in ['FGSM','CW','PGD','JSMA','DeepFool','BIM','MIM','Transfer','Boundary','HopSkipJump','ZOO','NES','SPSA','Square','GAN']:
            target=cache/name
            if target.is_symlink():target.unlink()
            elif target.is_dir():
                assert cache.resolve() in target.resolve().parents
                shutil.rmtree(target)
        if not any(cache.iterdir()):cache.rmdir()
    after=protected_hashes();assert before==after,'Protected artifact hash mismatch'
    report=dict(status='completed',removed=removed,stopped_pids=stopped,removed_notebook_cells=cells,
        protected_hashes=after,original_datasets_touched=False)
    (PROJECT/'adversarial_cleanup_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Cleanup complete; protected classifier artifacts unchanged.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args();main(args.apply)
