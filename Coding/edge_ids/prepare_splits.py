"""Create binary labels and stratified row assignments; no feature harmonization implied.

Run: python Coding/edge_ids/prepare_splits.py
Artifacts reference original rows (zero-based, excluding CSV header).
Split codes: 0=train, 1=validation, 2=test. Label codes: 0=benign, 1=attack.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def binary(values, mapping):
    text = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(text.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized labels: {unknown}")
    return text.map(mapping).to_numpy(dtype=np.uint8)


def csv_labels(path, column, mapping):
    return np.concatenate([
        binary(chunk[column], mapping)
        for chunk in pd.read_csv(path, usecols=[column], dtype=str, chunksize=250_000)
    ])


def datasense(root, seconds):
    archive = next((root / 'DataSense').glob('*.tar.xz'))
    labels, sources = [], []
    offset = 0
    with tarfile.open(archive, 'r|xz') as outer:
        for member in outer:
            if not member.name.endswith(f'_samples_{seconds}sec.csv.tar.xz'):
                continue
            with tarfile.open(fileobj=outer.extractfile(member), mode='r|xz') as inner:
                for item in inner:
                    if not item.isfile() or not item.name.endswith('.csv'):
                        continue
                    reader = csv.DictReader(
                        line.decode('utf-8-sig') for line in inner.extractfile(item)
                    )
                    result = bytearray()
                    for row in reader:
                        value = row['label1'].strip().lower()
                        if value not in {'benign', 'attack'}:
                            raise ValueError(f'Unknown DataSense label: {value}')
                        result.append(int(value == 'attack'))
                    y = np.frombuffer(result, dtype=np.uint8).copy()
                    sources.append(dict(path=str(archive.relative_to(root)),
                                        member=member.name, csv_member=item.name,
                                        offset=offset, rows=len(y)))
                    offset += len(y)
                    labels.append(y)
    if len(sources) != 2:
        raise ValueError('Expected one benign and one attack DataSense CSV')
    return np.concatenate(labels), sources


def prepare(root, name, seconds):
    if name == 'DataSense':
        return datasense(root, seconds)
    if name == 'X-IIoTID':
        files = [root / name / 'X-IIoTID dataset.csv']
        column, mapping = 'class3', {'normal': 0, 'attack': 1}
    elif name == 'Edge-IIoTset':
        # These releases overlap; select DNN only, never concatenate ML and DNN.
        files = [root / 'Edge-IIoTset dataset' / 'Selected dataset for ML and DL'
                 / 'DNN-EdgeIIoT-dataset.csv']
        column, mapping = 'Attack_label', {'0': 0, '1': 1, '0.0': 0, '1.0': 1}
    else:
        files = sorted((root / 'CICIoT2023' / 'CSV').rglob('*.csv'))
        column, mapping = None, None
    parts, sources, offset = [], [], 0
    for path in files:
        if column:
            y = csv_labels(path, column, mapping)
        else:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                first = next(csv.reader(handle))[0]
            n = sum(len(c) for c in pd.read_csv(path, usecols=[first], chunksize=250_000))
            # Current release is organized by scenario, not an embedded row label.
            y = np.full(n, int(path.parent.name != 'Benign_Final'), dtype=np.uint8)
        sources.append(dict(path=str(path.relative_to(root)), offset=offset,
                            rows=len(y), bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns))
        parts.append(y)
        offset += len(y)
    return np.concatenate(parts), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=[
        'DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--datasense-seconds', type=int, default=1, choices=range(1, 11))
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project.parents[1] / 'Datasets'
    out = project / 'artifacts' / 'native_splits'
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print(f'Reading {name}', flush=True)
        y, sources = prepare(root, name, args.datasense_seconds)
        index = np.arange(len(y))
        train, holdout = train_test_split(index, test_size=.30, stratify=y,
                                          random_state=args.seed)
        val, test = train_test_split(holdout, test_size=.50, stratify=y[holdout],
                                    random_state=args.seed)
        split = np.full(len(y), 255, dtype=np.uint8)
        counts = {}
        for code, (key, selected) in enumerate(zip(
                ['train', 'validation', 'test'], [train, val, test])):
            assert np.all(split[selected] == 255)
            split[selected] = code
            counts[key] = dict(rows=len(selected),
                               benign=int((y[selected] == 0).sum()),
                               attack=int((y[selected] == 1).sum()))
        assert np.all(split < 3)
        # Counts are within rounding of requested class proportions.
        for cls in (0, 1):
            total = int((y == cls).sum())
            for code, ratio in enumerate([.7, .15, .15]):
                assert abs(int(((y == cls) & (split == code)).sum()) - ratio*total) <= 2
        np.savez_compressed(out / f'{name}.npz', labels=y, split=split)
        metadata = dict(dataset=name, seed=args.seed, counts=counts, sources=sources,
                        label_codes={'benign': 0, 'attack': 1},
                        split_codes={'train': 0, 'validation': 1, 'test': 2},
                        label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                        split_sha256=hashlib.sha256(split.tobytes()).hexdigest(),
                        status='native row splits only; not harmonized flow features',
                        datasense_window_seconds=args.datasense_seconds if name == 'DataSense' else None,
                        label_source={'DataSense': 'label1', 'Edge-IIoTset': 'Attack_label',
                                      'X-IIoTID': 'class3',
                                      'CICIoT2023': 'scenario folder; Benign_Final=benign, others=attack'}[name],
                        limitations=[
                            'Row stratification does not isolate sessions, captures, devices or duplicates.',
                            'DataSense windows and packet aggregates are not equivalent to network flows.',
                            'No features have been imputed, scaled, selected or pooled.',
                            'CICIoT2023 labels describe scenarios; per-flow attack attribution is unverified.',
                        ])
        (out / f'{name}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(name, json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
