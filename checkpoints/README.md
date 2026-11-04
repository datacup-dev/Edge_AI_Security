# GPU-enabled checkpoint bundle

`code_and_checkpoints.zip` is the exact GPU-enabled export from `20261104_121037`. It includes eight completed 300-round defenses: FGSM, CW, PGD, JSMA, DeepFool, BIM, MIM and Transfer; frozen training manifests; GPU-enabled code; and a transfer inventory. It contains no Boundary tree checkpoint because that export was taken during matrix preparation. The models use the original provisional attack snapshots.

The approximately 10.7 GB of training arrays and shards are not in this ZIP; transfer the files listed in `edge_ids/transfer-files.txt` separately.

Those exact files are provided as seven compressed parts in the
[training-data release](https://github.com/datacup-dev/Edge_AI_Security/releases/tag/training-snapshot-20261104-121037).
Use [the server downloader and guide](../Coding/edge_ids/TRAINING_DATA_TRANSFER.md)
to fetch and verify them.

From the repository root on Linux:

```bash
(cd checkpoints && sha256sum -c SHA256SUMS)
python3 -m zipfile -e checkpoints/code_and_checkpoints.zip /tmp/edge_ids_restore
cp -a /tmp/edge_ids_restore/edge_ids/. Coding/edge_ids/
```

Restore into a fresh clone before starting training; copying this snapshot into an active run can replace newer checkpoints. See `Coding/edge_ids/LAB_SERVER_SETUP.md` for environment setup, data verification and GPU launch.
