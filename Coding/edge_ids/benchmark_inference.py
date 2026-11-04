"""Warm, preprocessed-input inference timing; not end-to-end edge latency."""
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from adapted_models import AdaptedModel


def main():
    project = Path(__file__).resolve().parent
    experiment = project / 'artifacts/adapted_experiment'
    progress = json.loads((experiment / 'progress.json').read_text())
    if progress['stage'] != 'complete':
        raise RuntimeError('Finish model selection and test evaluation before timing inference')
    names = ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']
    feature_lists, examples = [], []
    for name in names:
        source = project / 'artifacts/encoded_inputs' / name
        feature_lists.append(json.loads((source/'preprocessing.json').read_text())['features'])
        training = np.flatnonzero(np.load(source/'split.npy') == 0)[:512]
        examples.append(np.asarray(np.load(source/'features.npy', mmap_mode='r')[training]).copy())
    candidates = json.loads((experiment/'baseline.json').read_text())['candidates']
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    rows = []
    for entry in candidates:
        kind = entry['model']
        checkpoint = torch.load(experiment/kind/'best.pt', map_location='cpu', weights_only=True)
        for device_name in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            device = torch.device(device_name)
            model = AdaptedModel(kind, feature_lists).to(device).eval()
            model.load_state_dict(checkpoint['state_dict'])
            for code, name in enumerate(names):
                for batch_size in [1, 512]:
                    x = torch.from_numpy(examples[code][:batch_size]).to(device)
                    durations = []
                    with torch.inference_mode():
                        for iteration in range(25):
                            if device_name == 'cuda':
                                torch.cuda.synchronize()
                            start = time.perf_counter()
                            with torch.autocast(device_type=device_name, dtype=torch.float16,
                                                enabled=device_name=='cuda'):
                                probability = torch.sigmoid(model(x, code).float())
                            if device_name == 'cuda':
                                torch.cuda.synchronize()
                            duration = time.perf_counter()-start
                            if iteration >= 5:
                                durations.append(duration)
                    rows.append(dict(model=kind, dataset=name, device=device_name,
                                     batch_size=batch_size, repetitions=20,
                                     median_ms=float(np.median(durations)*1000),
                                     p95_ms=float(np.quantile(durations,.95)*1000),
                                     observations_per_second=float(batch_size/np.median(durations))))
            del model
            if device_name == 'cuda':
                torch.cuda.empty_cache()
    target = project/'artifacts/report'
    target.mkdir(exist_ok=True)
    with (target/'inference_latency.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scope = dict(scope='Warm model+sigmoid inference on already preprocessed resident tensors; training examples used only as timing inputs',
                 excluded=['packet extraction','scaling','host-to-device transfer','queueing','window collection'],
                 platform=platform.platform(), torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 cpu_threads=4, warmups=5, repetitions=20,
                 caveat='Workstation timing, not validated edge-device or end-to-end real-time performance.')
    (target/'inference_latency_scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    print(f'Wrote {len(rows)} inference timing records',flush=True)


if __name__ == '__main__':
    main()
