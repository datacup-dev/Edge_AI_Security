"""Combine native training records into a shuffled, source-backed mixed index.

This combines membership, labels and provenance, not incompatible feature schemas.
The original CSV/archive records remain the backing data for each indexed row.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    project = Path(__file__).resolve().parent
    source_dir = project / 'artifacts' / 'native_splits'
    output = project / 'artifacts' / 'mixed_training'
    output.mkdir(parents=True, exist_ok=True)
    names = ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']
    metadata = [json.loads((source_dir / f'{name}.json').read_text()) for name in names]
    count = sum(m['counts']['train']['rows'] for m in metadata)
    dtype = np.dtype([('dataset_id', 'u1'), ('source_row', '<u4'), ('label', 'u1')])
    mixed = np.empty(count, dtype=dtype)
    cursor = 0
    for code, (name, meta) in enumerate(zip(names, metadata)):
        with np.load(source_dir / f'{name}.npz') as data:
            split, labels = data['split'], data['labels']
        assert hashlib.sha256(split.tobytes()).hexdigest() == meta['split_sha256']
        assert hashlib.sha256(labels.tobytes()).hexdigest() == meta['label_sha256']
        rows = np.flatnonzero(split == 0)
        assert len(rows) == meta['counts']['train']['rows']
        assert rows.max() <= np.iinfo(np.uint32).max
        part = mixed[cursor:cursor + len(rows)]
        part['dataset_id'] = code
        part['source_row'] = rows
        part['label'] = labels[rows]
        cursor += len(rows)
        print(f'{name}: included {len(rows):,} training records', flush=True)
    assert cursor == count
    np.random.default_rng(42).shuffle(mixed)
    target = output / 'records.npy'
    np.save(target, mixed, allow_pickle=False)
    del mixed
    saved = np.load(target, mmap_mode='r', allow_pickle=False)
    summary = {}
    for code, (name, meta) in enumerate(zip(names, metadata)):
        part = saved[saved['dataset_id'] == code]
        with np.load(source_dir / f'{name}.npz') as data:
            split, labels = data['split'], data['labels']
        rows = part['source_row']
        assert np.array_equal(np.sort(rows), np.flatnonzero(split == 0))
        assert np.all(split[rows] == 0)
        assert np.array_equal(part['label'], labels[rows])
        summary[name] = dict(dataset_id=code, rows=len(part),
                             benign=int((part['label'] == 0).sum()),
                             attack=int((part['label'] == 1).sum()),
                             fraction=len(part) / count,
                             source_manifest=f'../native_splits/{name}.json',
                             source_manifest_sha256=digest(source_dir / f'{name}.json'))
    report = dict(
        status='Mixed training membership complete; common feature matrix pending',
        rows=count, seed=42, record_file='records.npy', sha256=digest(target),
        datasets=summary,
        benign=sum(v['benign'] for v in summary.values()),
        attack=sum(v['attack'] for v in summary.values()),
        representation='source-backed native-record index, not a numeric feature matrix',
        source_row_definition='Zero-based global CSV record within the dataset; use source manifest offsets to locate file/member and local row.',
        sampling='All training rows, natural proportions; no balancing or downsampling.',
        validation='Exact training membership, uniqueness and labels verified after saving; no validation/test rows included.',
        limitations=['Source records have incompatible observation units and feature schemas.',
                     'Shared feature preprocessing and model training have not been performed.'])
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['rows', 'benign', 'attack', 'status']}), flush=True)


if __name__ == '__main__':
    main()
