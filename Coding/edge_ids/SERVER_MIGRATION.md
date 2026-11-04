# Resume defense training on the lab server

Update: the server's RTX 3090 configuration is now known. The optional GPU
launcher and CUDA preflight are documented in LAB_SERVER_SETUP.md. The older
CPU-only discussion below describes the initial migration profile. Use the
latest exported bundle to obtain the GPU-enabled iterator and launcher.

Unfinished adversarial generation does not prevent migration. Existing defenses
use fixed snapshots, so they resume with exactly those records. They remain
provisional when their snapshot is partial. Adding newly generated records
requires a separate run from scratch; do not replace manifests under checkpoints.

## What is saved

`prepare_server_migration.py` exports source code, requirements, all 15 frozen
training manifests, baseline configuration, and stable load-verified UBJ models.
Completed defenses retain `complete.json` and are skipped on resume. Interrupted
boosting resumes the last saved 25-round checkpoint, which may be behind the
displayed progress. Matrix preparation has no tree checkpoint and restarts.
The export does not stop local jobs. Avoid running the same defense on both
machines after confirming the server works. A final cutover requires stopping
the local scheduler/worker and exporting again if newer checkpoints are wanted.

The output is `migration/<timestamp>/code_and_checkpoints.zip` and an unpacked
`edge_ids` directory. `migration_inventory.json` records verified model hashes,
round counts, and required data file sizes. It covers defense training only,
not resuming unfinished attack-generation jobs or later validation/test evaluation.
Data arrays are not included in this ZIP. Disk caches and scheduler locks are
excluded and are rebuilt on the server.

## GitHub and large data

Put source code, documentation, requirements and small metadata in a private
GitHub repository. The exported `.gitignore` excludes artifacts by default.
Transfer the checkpoint ZIP and required data via lab storage, SFTP or another
large-file channel. Git cloning alone will not provide the excluded artifacts.
No upload is performed by the exporter. Do not commit the virtual environment,
raw datasets, generated shards or external-memory caches into ordinary Git.

Copy every project-relative file listed in `edge_ids/transfer-files.txt` from
the workstation project root into the server project root, preserving paths.
These are the four encoded feature arrays, mixed training membership index,
and the exact training shards referenced by all frozen manifests. The exporter
checks that all files exist; its size inventory is not a full content hash audit.
Use a transfer tool with checksum verification. Keep the original workstation
copies until server validation and training resume succeed.

## Server setup (Linux example)

Unpack the ZIP into the cloned project directory so the bundled Python files
and `artifacts` directory sit together. If the repository preserves the original
layout, this is `Coding/edge_ids`. Run from that directory. Use Python 3.13 as
on the workstation and compatible package wheels:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-server.txt
python verify_server_migration.py
python -u train_adversarial_gb.py --threads 16 --matrix external
```

The thread count is an example to adjust to the server allocation. External
memory uses local SSD and minimizes RAM pressure. A server with ample RAM may
use `--matrix memory` to reduce disk traffic, after assessing available memory.
Both modes use CPU histogram training with the original tree hyperparameters.
GPU training is not enabled by this migration: it needs a compatible GPU setup,
VRAM capacity assessment and an appropriate GPU data iterator. More hardware
does not guarantee a particular speedup. Check server specs before choosing.
Keep one scheduler running. The bundle contains no stale scheduler lock.

Progress is `artifacts/adversarial_defenses/status.json`, each defense's
`progress.json`, and `<attack>.log`. Resume rebuilds quantile cuts from the same
training snapshot, then continues the saved trees to a total of 300 rounds.
There is no optimizer state for these gradient-boosted trees. Device, threads,
package versions and numeric execution can affect bitwise reproducibility.
