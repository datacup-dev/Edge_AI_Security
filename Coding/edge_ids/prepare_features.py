"""Dataset-specific numeric inputs; not harmonized flow features."""
import csv
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1] / 'Datasets'
OUTPUT = PROJECT / 'artifacts' / 'encoded_inputs'
EDGE = ['arp.opcode', 'arp.hw.size', 'http.content_length',
        'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
        'tcp.connection.synack', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack',
        'tcp.len', 'tcp.srcport', 'udp.time_delta', 'dns.qry.name.len',
        'dns.qry.qu', 'dns.qry.type', 'dns.retransmission', 'mqtt.conflag.cleansess',
        'mqtt.hdrflags', 'mqtt.len', 'mqtt.msgtype', 'mqtt.proto_len',
        'mqtt.topic_len', 'mqtt.ver', 'mbtcp.len']
X_FEATURES = ['Scr_port', 'Des_port', 'Duration', 'Scr_bytes', 'Des_bytes',
              'missed_bytes', 'is_syn_only', 'Is_SYN_ACK', 'is_pure_ack',
              'is_with_payload', 'FIN or RST', 'Bad_checksum', 'is_SYN_with_RST',
              'Scr_pkts', 'Scr_ip_bytes', 'Des_pkts', 'Des_ip_bytes',
              'total_bytes', 'total_packet', 'paket_rate', 'byte_rate',
              'Scr_packts_ratio', 'Des_pkts_ratio', 'Scr_bytes_ratio', 'Des_bytes_ratio']


def number(frame, features):
    result = np.empty((len(frame), len(features)), np.float32)
    for j, col in enumerate(features):
        if pd.api.types.is_numeric_dtype(frame[col]):
            result[:, j] = frame[col].to_numpy(dtype=np.float32)
            continue
        values = frame[col].astype(str).str.strip().str.lower()
        boolean = values.map({'true': '1', 'false': '0'})
        values = values.where(boolean.isna(), boolean)
        hexmask = values.str.startswith('0x', na=False)
        if hexmask.any():
            values.loc[hexmask] = values.loc[hexmask].map(lambda v: str(int(v, 16)))
        result[:, j] = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float32)
    result[~np.isfinite(result)] = np.nan
    # Fixed, non-learned transform reduces scale extremes while retaining sign.
    return np.sign(result) * np.log1p(np.abs(result))


def sources(name, meta):
    if name != 'DataSense':
        for item in meta['sources']:
            path = ROOT / item['path']
            yield from pd.read_csv(path, dtype=np.float32 if name == 'CICIoT2023' else str, chunksize=100_000,
                                   usecols=(EDGE if name == 'Edge-IIoTset' else
                                            X_FEATURES if name == 'X-IIoTID' else None))
        return
    members = {item['member'] for item in meta['sources']}
    with tarfile.open(ROOT / meta['sources'][0]['path'], 'r|xz') as outer:
        for member in outer:
            if member.name not in members:
                continue
            with tarfile.open(fileobj=outer.extractfile(member), mode='r|xz') as inner:
                for item in inner:
                    if not item.isfile() or not item.name.endswith('.csv'):
                        continue
                    reader = csv.DictReader(line.decode('utf-8-sig') for line in inner.extractfile(item))
                    rows = []
                    for row in reader:
                        rows.append(row)
                        if len(rows) == 50_000:
                            yield pd.DataFrame(rows)
                            rows = []
                    if rows:
                        yield pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ['DataSense', 'Edge-IIoTset', 'CICIoT2023', 'X-IIoTID']:
        meta = json.loads((PROJECT / 'artifacts/native_splits' / f'{name}.json').read_text())
        with np.load(PROJECT / 'artifacts/native_splits' / f'{name}.npz') as archive:
            split, labels = archive['split'], archive['labels']
        print('Preparing', name, flush=True)
        directory = OUTPUT / name
        directory.mkdir(exist_ok=True)
        completed = directory / 'preprocessing.json'
        if completed.exists():
            previous = json.loads(completed.read_text())
            if previous['split_sha256'] == meta['split_sha256'] and previous['rows'] == len(split):
                print(name, 'using completed numeric inputs', flush=True)
                continue
        scaler, offset, raw, features = StandardScaler(), 0, None, None
        missing_train = None
        for chunk in sources(name, meta):
            if raw is None:
                if name == 'DataSense':
                    excluded = {'network_ips_all', 'network_ips_src', 'network_ips_dst',
                                'network_macs_all', 'network_macs_src', 'network_macs_dst',
                                'network_ports_all', 'network_ports_src', 'network_ports_dst',
                                'network_protocols_all', 'network_protocols_src', 'network_protocols_dst'}
                    features = [c for c in chunk.columns if c.startswith('network_') and c not in excluded]
                else:
                    features = EDGE if name == 'Edge-IIoTset' else X_FEATURES if name == 'X-IIoTID' else list(chunk.columns)
                assert not any('label' in f.lower() or 'class' in f.lower() for f in features)
                raw = np.lib.format.open_memmap(directory / 'features.npy', mode='w+',
                                               dtype=np.float32, shape=(len(split), len(features)))
                missing_train = np.zeros(len(features), dtype=np.int64)
            block = number(chunk, features)
            end = offset + len(block)
            raw[offset:end] = block
            train = block[split[offset:end] == 0]
            if len(train):
                missing_train += np.isnan(train).sum(axis=0)
                # Train-only missing -> zero in transformed space, a fixed policy.
                scaler.partial_fit(np.nan_to_num(train, nan=0.0))
            offset = end
            if offset % 1_000_000 < len(block):
                print(name, offset, '/', len(split), flush=True)
        assert offset == len(split), (name, offset, len(split))
        for start in range(0, len(split), 100_000):
            stop = min(start + 100_000, len(split))
            block = np.nan_to_num(raw[start:stop], nan=0.0)
            raw[start:stop] = scaler.transform(block).astype(np.float32)
            assert np.isfinite(raw[start:stop]).all()
        raw.flush()
        np.save(directory / 'labels.npy', labels)
        np.save(directory / 'split.npy', split)
        info = dict(dataset=name, rows=len(split), features=features,
                    scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                    train_missing_per_feature=missing_train.tolist(),
                    transform='signed log1p; missing/nonfinite -> zero; per-domain training-only standard scaling',
                    excluded='Labels, raw IP/MAC identifiers/lists, timestamps, payload/text and host alerts/logs.',
                    representation='dataset-specific native numeric observations; not common flow features',
                    split_sha256=meta['split_sha256'], label_sha256=meta['label_sha256'])
        (directory / 'preprocessing.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print(name, raw.shape, 'complete', flush=True)


if __name__ == '__main__':
    main()
