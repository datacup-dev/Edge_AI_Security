"""Restore the validation-selected baseline for preprocessed native inputs."""
import json
from pathlib import Path

import torch

from adapted_models import AdaptedModel


def load_baseline(device=None):
    project = Path(__file__).resolve().parent
    experiment = project/'artifacts/adapted_experiment'
    selection = experiment/'baseline.json'
    if not selection.exists():
        raise RuntimeError('Baseline selection is not complete')
    summary = json.loads(selection.read_text())
    names = ['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']
    features = [json.loads((project/'artifacts/encoded_inputs'/n/'preprocessing.json').read_text())['features'] for n in names]
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    kind = summary['selected']['model']
    model = AdaptedModel(kind,features).to(device)
    state = torch.load(experiment/kind/'best.pt',map_location=device,weights_only=True)
    assert state['kind'] == kind and state['feature_counts'] == list(map(len,features))
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, dict(selection=summary,domains=names,features=features,
                       input_contract='Native numeric inputs transformed with the matching saved training-only preprocessing; not raw flows')


@torch.inference_mode()
def attack_probability(model, preprocessed_rows, domain):
    names = ['DataSense','Edge-IIoTset','CICIoT2023','X-IIoTID']
    code = names.index(domain)
    device = next(model.parameters()).device
    x = torch.as_tensor(preprocessed_rows,dtype=torch.float32,device=device)
    if x.ndim != 2 or x.shape[1] != model.encoders[code].weight.shape[0]:
        raise ValueError('Input must be [observations, native features] in the saved feature order')
    if not torch.isfinite(x).all():
        raise ValueError('Preprocessed inputs must be finite')
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        logits = model(x,code)
    return torch.sigmoid(logits.float()).cpu().numpy()
