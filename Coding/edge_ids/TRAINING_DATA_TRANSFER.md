# Checkpoint training data on GitHub

Repository: https://github.com/datacup-dev/Edge_AI_Security

Release tag: `training-snapshot-20261104-121037`. The publisher keeps the release
in draft until all assets are uploaded and their GitHub SHA-256 digests match.
The release contains exactly the original checkpoint snapshot inputs, not newer
full-pool generation outputs. It contains 74,451 source files totaling
10,737,816,833 bytes before archive compression. Validation/test inputs are not
included. Eight completed models are in `checkpoints/code_and_checkpoints.zip`.

On the Ubuntu server, clone or update the repository, then run from
`Coding/edge_ids` after the release is published:

```bash
python3 download_training_data.py --destination training-data-download
(cd training-data-download && sha256sum -c SHA256SUMS)
cat training-data-download/training-data.tar.gz.part* | tar -xz -C .
```

The downloader uses only Python's standard library. It downloads public release
assets, checks GitHub's published digests and the package part digests, skips
already verified downloads, and attempts to resume interrupted downloads. No
GitHub token is needed for this public release. Git clone alone does not fetch
release assets. Verify all parts before extraction, and restore only into a
fresh migration directory to avoid replacing newer artifacts.

Restore the checkpoint ZIP as described in `checkpoints/README.md` first, and
then verify and launch using the configured environment:

```bash
.venv/bin/python verify_server_migration.py
CUDA_VISIBLE_DEVICES=0 bash run_lab_gpu.sh
```

Install dependencies and create the virtual environment as described in
LAB_SERVER_SETUP.md. The server GPU preflight precedes full training; full-data
VRAM fit and performance remain to be tested on the server. Restoring more data
does not alter existing frozen snapshots or make their provisional models final.

The repository's checkpoint ZIP is the exact historical export. Its bundled
Python files can overwrite later helpers if restored afterward; run `git restore
Coding/edge_ids` from the repository root after unpacking to retain the latest
tracked code while preserving untracked restored artifacts. Use this command
only on a fresh clone with no local code edits to preserve.
