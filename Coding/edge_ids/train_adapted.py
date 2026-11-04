"""Full-data mixed training with per-domain encoders; validation-only selection."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from adapted_models import AdaptedModel, GROUPS, group_for
from evaluate import evaluate_predictions

PROJECT = Path(__file__).resolve().parent
NAMES = ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']
KINDS = ['MLP', 'CNN1D', 'LSTM', 'GRU', 'BiLSTM']


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows readers/antivirus can briefly block replacing an open file.
            if attempt == 29:
                if path.name == 'progress.json':
                    return  # A stale progress update must not abort training.
                raise
            time.sleep(.1)


def load_domains():
    domains = []
    for name in NAMES:
        directory = PROJECT / 'artifacts/encoded_inputs' / name
        info = json.loads((directory / 'preprocessing.json').read_text())
        x = np.load(directory / 'features.npy', mmap_mode='r')
        y = np.load(directory / 'labels.npy')
        split = np.load(directory / 'split.npy')
        assert len(x) == len(y) == len(split) == info['rows']
        domains.append(dict(name=name, x=x, y=y, features=info['features'],
                            indices=[np.flatnonzero(split == code) for code in range(3)]))
    return domains


@torch.inference_mode()
def predict(model, domain, code, rows, device, batch_size=None):
    model.eval()
    batch_size = batch_size or 8192
    probability = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start+batch_size]
        x = torch.from_numpy(np.asarray(domain['x'][selected])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logits = model(x, code)
        probability[start:start+len(selected)] = torch.sigmoid(logits.float()).cpu().numpy()
    return probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--microbatch-size', type=int, default=8192)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--models', nargs='+', default=KINDS, choices=KINDS)
    args = parser.parse_args()
    output = PROJECT / 'artifacts/adapted_experiment'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    domains = load_domains()
    counts = [d['x'].shape[1] for d in domains]
    feature_lists = [d['features'] for d in domains]
    configuration = dict(seed=42, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                         models=args.models, feature_counts=counts, epochs=args.epochs,
                         batch_size=args.batch_size, early_stopping_patience=args.patience,
                         microbatch_size=args.microbatch_size,
                         precision='CUDA float16 autocast with GradScaler; float32 master parameters',
                         optimizer='Adam', learning_rate=.001, dropout=.3,
                         loss='unweighted BCEWithLogitsLoss', hidden_widths=[128,64,32],
                         encoder='Per-domain scalar feature tokens, width 8, affine+tanh; mean pooled into eight documented feature groups',
                         feature_groups=GROUPS,
                         feature_group_assignment={d['name']:{f:GROUPS[group_for(f)] for f in d['features']} for d in domains},
                         recurrent_input='Eight fixed-order feature-group tokens, NOT time sequences',
                         mixed_sampling='All original training rows once per epoch, shuffled domain-homogeneous batches at natural proportions',
                         baseline_selection='Mean macro-F1 across four validation sets; mean AP then fewer parameters break ties',
                         threshold=.5, adaptation='Authorized per-domain encoders; not paper eight-feature harmonization',
                         exclusions='No 2D/3D reshaping: no spatial axes available',
                         paper_deviations='Dataset-specific encoders; feature-sequence recurrent models; batch8192 for GPU throughput; validation early stopping; CNN/RNN extensions')
    save_json(output / 'configuration.json', configuration)
    # Model selection is completed before ANY model test results are computed.
    candidates = []
    for kind in args.models:
        directory = output / kind
        directory.mkdir(exist_ok=True)
        existing = directory / 'selection.json'
        if existing.exists():
            candidates.append(json.loads(existing.read_text()))
            print('Using completed training', kind, flush=True)
            continue
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
        model = AdaptedModel(kind, feature_lists).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        grad_scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
        microbatch_size = args.microbatch_size
        loss_function = torch.nn.BCEWithLogitsLoss()
        best, best_ap, stale, history = -1, -1, 0, []
        resume = directory / 'last.pt'
        if resume.exists():
            previous = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(previous['model'])
            optimizer.load_state_dict(previous['optimizer'])
            best, best_ap, stale, history = (previous[k] for k in ['best','best_ap','stale','history'])
            torch.set_rng_state(previous['torch_rng'].cpu())
            if 'grad_scaler' in previous:
                grad_scaler.load_state_dict(previous['grad_scaler'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([state.cpu() for state in previous['cuda_rng']])
        for epoch in range(len(history), args.epochs):
            start_time = time.perf_counter()
            rng = np.random.default_rng(42+epoch)
            indices = [rng.permutation(d['indices'][0]) for d in domains]
            schedule = [(code, start) for code, rows in enumerate(indices)
                        for start in range(0, len(rows), args.batch_size)]
            rng.shuffle(schedule)
            total_loss, seen = 0., 0
            model.train()
            for batch, (code, start) in enumerate(schedule):
                rows = indices[code][start:start+args.batch_size]
                domain = domains[code]
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss = torch.zeros((), device=device)
                    x = y = loss = None
                    try:
                        for micro_start in range(0, len(rows), microbatch_size):
                            selected = rows[micro_start:micro_start+microbatch_size]
                            x = torch.from_numpy(np.asarray(domain['x'][selected])).to(device)
                            y = torch.from_numpy(domain['y'][selected].astype(np.float32)).to(device)
                            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                                loss = loss_function(model(x, code), y)
                            grad_scaler.scale(loss * (len(selected)/len(rows))).backward()
                            batch_loss += loss.detach() * len(selected)
                        loss_value = float(batch_loss)
                        if not np.isfinite(loss_value):
                            raise ValueError(f'Nonfinite loss: {kind}/{epoch}/{batch}')
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        break
                    except torch.cuda.OutOfMemoryError:
                        if microbatch_size <= 128:
                            raise
                        microbatch_size //= 2
                        optimizer.zero_grad(set_to_none=True)
                        del x, y, loss
                        torch.cuda.empty_cache()
                        print(kind, 'reduced physical microbatch to', microbatch_size, flush=True)
                total_loss += loss_value
                seen += len(rows)
                if batch % 100 == 0:
                    save_json(output / 'progress.json', dict(stage='training', model=kind,
                              epoch=epoch+1, batch=batch+1, batches=len(schedule),
                              rows_seen=seen, rows_per_epoch=sum(len(i) for i in indices),
                              microbatch_size=microbatch_size,
                              elapsed_seconds=time.perf_counter()-start_time))
            assert seen == sum(len(i) for i in indices)
            validation = {}
            for code, domain in enumerate(domains):
                rows = domain['indices'][1]
                p = predict(model, domain, code, rows, device)
                y = domain['y'][rows]
                validation[domain['name']] = dict(
                    macro_f1=float(f1_score(y, p>=.5, average='macro', zero_division=0)),
                    average_precision=float(average_precision_score(y, p)))
            score = float(np.mean([s['macro_f1'] for s in validation.values()]))
            ap = float(np.mean([s['average_precision'] for s in validation.values()]))
            record = dict(epoch=epoch+1, train_loss=total_loss/seen, validation=validation,
                          mean_validation_macro_f1=score, mean_validation_ap=ap,
                          elapsed_seconds=time.perf_counter()-start_time, training_rows=seen)
            history.append(record)
            save_json(directory / 'history.json', history)
            print(kind, json.dumps(record), flush=True)
            if (score, ap) > (best, best_ap):
                best, best_ap, stale = score, ap, 0
                torch.save(dict(state_dict=model.state_dict(), kind=kind,
                                feature_counts=counts, epoch=epoch+1), directory / 'best.pt')
            else:
                stale += 1
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            best=best, best_ap=best_ap, stale=stale, history=history,
                            grad_scaler=grad_scaler.state_dict(),
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []), resume)
            if stale >= args.patience:
                break
        selection = dict(model=kind, validation_macro_f1=best, validation_ap=best_ap,
                         parameters=sum(p.numel() for p in model.parameters()),
                         epochs_run=len(history))
        save_json(existing, selection)
        candidates.append(selection)
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history])
        axes[0].set(xlabel='Epoch', ylabel='Training BCE')
        axes[1].plot([r['epoch'] for r in history], [r['mean_validation_macro_f1'] for r in history])
        axes[1].set(xlabel='Epoch', ylabel='Mean validation macro-F1')
        fig.tight_layout()
        fig.savefig(directory / 'learning_curves.png', dpi=160)
        plt.close(fig)
    baseline = max(candidates, key=lambda c:(c['validation_macro_f1'],c['validation_ap'],-c['parameters']))
    save_json(output / 'baseline.json', dict(selected=baseline, candidates=candidates,
              selection_uses_test=False, status='Selected using validation only; adapted experiment'))
    for candidate in candidates:
        kind = candidate['model']
        directory = output / kind
        checkpoint = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
        model = AdaptedModel(kind, feature_lists).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        for code, domain in enumerate(domains):
            target = directory / 'test' / domain['name']
            if all((target / file).exists() for file in ['metrics.json','roc_pr.png','source_rows.npy']):
                continue
            save_json(output / 'progress.json', dict(stage='test_evaluation', model=kind, dataset=domain['name']))
            rows = domain['indices'][2]
            probability = predict(model, domain, code, rows, device)
            evaluate_predictions(domain['y'][rows], probability, target)
            np.save(target / 'source_rows.npy', rows)
    # Reference model: probability is just each domain's training attack rate.
    for domain in domains:
        rows = domain['indices'][2]
        prior = float(domain['y'][domain['indices'][0]].mean())
        evaluate_predictions(domain['y'][rows], np.full(len(rows), prior, dtype=np.float32),
                             output / 'domain_prior_reference' / domain['name'])
    save_json(output / 'progress.json', dict(stage='complete', baseline=baseline,
              models=len(candidates), test_evaluations=len(candidates)*4))
    from report import main as refresh_report
    refresh_report()
    print('COMPLETE', json.dumps(baseline), flush=True)


if __name__ == '__main__':
    main()
