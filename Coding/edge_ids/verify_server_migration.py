"""Validate portable model checksums and required data presence before resume."""
import json
import hashlib
from pathlib import Path
import xgboost as xgb

root=Path(__file__).resolve().parent
inventory=json.loads((root/'migration_inventory.json').read_text())
errors=[]
for entry in inventory['files']:
    p=root/entry['path']
    if not p.is_file() or p.stat().st_size!=entry['bytes']:errors.append(entry['path'])
for entry in inventory['models']:
    p=root/'artifacts/adversarial_defenses'/entry['attack']/entry['checkpoint']
    if not p.is_file():errors.append(str(p));continue
    if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:errors.append(str(p));continue
    model=xgb.Booster();model.load_model(p)
    assert model.num_boosted_rounds()==entry['rounds']
for attack in ['FGSM','CW','PGD','JSMA','DeepFool','BIM','MIM','Transfer','Boundary','HopSkipJump','ZOO','NES','SPSA','Square','GAN']:
    d=root/'artifacts/adversarial_defenses'/attack
    if (d/'training_manifest.json').exists():
        assert (d/'training_manifest.json').read_bytes()==(root/'artifacts/augmented_training'/attack/'manifest.json').read_bytes()
if errors:raise SystemExit('Missing/changed inputs: '+str(errors[:20]))
print('Verified required file sizes and portable model hashes/rounds. Data transfer checksums should also be verified.')
