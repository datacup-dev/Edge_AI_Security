# Adversarial generation on the lab server

Publication status: the input archive is prepared locally, but public upload
requires explicit approval for its frozen weights, labels and model probabilities.
The download commands below apply only after that approval and publication.

This is a fresh server-generation workflow, not a continuation of workstation
cursors. It resumes its own server-created shards on rerun. Workstation shards
and the fixed training snapshots used by completed defenses are preserved.
Copying a workstation cursor without all its previous shards would falsely
claim missing coverage and is not supported. Exact local-run continuation
requires a separate transfer of all previously committed shards and state.

The existing training-data release already supplies encoded feature arrays and
mixed membership. Also download `attack-generation-inputs.tar.gz` from the same
GitHub release, with `attack-generation-inputs.sha256`. That extra archive
contains original frozen GB, distilled MLP, original MLP, GAN, training bounds,
split labels and metadata, held-out cohort indices, clean probabilities and a
per-file SHA-256 manifest. It contains no defended model fitting and no generated
attack outputs. All data roles retain original labels and target `1-label`.

From `Coding/edge_ids`, after updating the repository:

```bash
python3 - <<'PY'
import urllib.request
base='https://github.com/datacup-dev/Edge_AI_Security/releases/download/training-snapshot-20261104-121037/'
for name in ['attack-generation-inputs.tar.gz','attack-generation-inputs.sha256']:
    urllib.request.urlretrieve(base+name,name)
PY
sha256sum -c attack-generation-inputs.sha256
tar -xzf attack-generation-inputs.tar.gz
python3.13 -m venv .venv-attacks
.venv-attacks/bin/python -m pip install -r requirements-server-attacks.txt
CUDA_VISIBLE_DEVICES=0 .venv-attacks/bin/python -u server_attack_generation.py --role train --method NES
```

Use a persistent lab terminal/session. The above command generates NES candidates
for every original mixed-training record, both classes, starting at zero in the
new server output directory. It does not use validation/test records for fitting.
If interrupted, rerun the same command to continue committed server shards.
An abrupt termination can leave an `<attack>.lock`; remove only that lock after
verifying no matching generation process is still running.

At preparation time, the five incomplete workstation training attacks were
HopSkipJump, ZOO, NES, SPSA and Square. You can run each separately. Start with
one process due shared RAM and CPU pressure; further concurrency must be based
on actual available resources. Gradient/proxy and GAN operations use PyTorch
CUDA when available. The frozen GB oracle remains CPU-based, so score/query
methods are not automatically accelerated by GPU availability.

Held-out examples (each role and dataset stays separate):

```bash
CUDA_VISIBLE_DEVICES=0 .venv-attacks/bin/python -u server_attack_generation.py --role validation --method PGD
CUDA_VISIBLE_DEVICES=0 .venv-attacks/bin/python -u server_attack_generation.py --role test --method Square --dataset CICIoT2023
```

Omit `--dataset` to process all four held-out datasets. Options are FGSM, CW,
PGD, JSMA, DeepFool, BIM, MIM, Transfer, Boundary, HopSkipJump, ZOO, NES, SPSA,
Square and GAN. DIGFuPAS remains excluded.

Outputs are `artifacts/server_attack_generation/<role>/<method>/...`; per-job
`cursor.json` records committed coverage, and `parameters.json` records settings.
`<method>_run.json` is the process-level status. Each run checks the frozen input
hashes before generating. Completion of generation still requires a full quality
audit before scientific conclusions; this runner marks that audit as pending.
Existing workstation audit tools use original paths and must be adapted before
claiming a full server-run audit. The same record IDs, seeds, epsilon, projection
rules and attack settings are used; cross-device bitwise reproduction is not
guaranteed. Existing proxy-gradient transfer variants and tabular Square
adaptations retain their original method limitations.

Local generation is not stopped by preparing or downloading these files.
Coordinate the cutover after confirming the server job runs correctly.
