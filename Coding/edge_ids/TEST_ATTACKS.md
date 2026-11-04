# Separate adversarial test datasets

This adds held-out generation to the existing both-class mixed-training run.
Training outputs remain under `artifacts/mixed_training_attacks/`. All test
outputs go to the independent `artifacts/separate_test_attacks/` tree.
No existing training samples or classifier results are replaced.

## Scope and separation

Following the latest both-class choice, every original test record is included:

| Dataset | Test records | Benign | Attack |
|---|---:|---:|---:|
| DataSense | 34,079 | 20,520 | 13,559 |
| Edge-IIoTset | 332,881 | 242,347 | 90,534 |
| CICIoT2023 | 7,016,505 | 164,729 | 6,851,776 |
| X-IIoTID | 123,126 | 63,213 | 59,913 |
| Total | 7,506,591 | 490,809 | 7,015,782 |

Fifteen attacks over four separate datasets form **60 jobs** and request
**112,598,865 candidate records**. The test cohorts use original split==2;
membership is checked for disjointness against the original mixed training pool.
No final subsampling or combining the four test sets is performed.

True labels are retained, with target=1-label. Initially wrong GB predictions
are kept unchanged and excluded from new attack-success counts. Changed,
unsuccessful, unchanged, seed-failed and query-exhausted cases remain visible.
Separate files do not guarantee a distinct successful vector for every source.

## Frozen components and validation rule

The audited GB baseline, distilled MLP, independent original MLP and label-
conditioned GAN are reused from the training run without fitting or updating.
Model SHA256 values are recorded before test generation and rechecked by workers
and audits. All preprocessing ranges, mutable-feature masks, epsilon=0.3,
algorithms and per-record budgets are reused unchanged. The surrogate and GAN
have fixed three/five-epoch training checkpoints; they are not selected or tuned
with test data. The original baseline remains validation-selected.

Only each individual test candidate is optimized through the declared oracle.
This expected test-time adversarial search is not training on held-out records.
Test candidates must not later be used for defense training. These original
test sets were already examined in the classifier benchmark; they are reused
evaluation sets rather than a new untouched external holdout.

All fifteen methods and exact settings are in
[the shared attack protocol](MIXED_TRAINING_ATTACKS.md): FGSM, C&W (`CW`), PGD,
JSMA, DeepFool, BIM, MIM, Transfer, Boundary, HopSkipJump, ZOO, NES, SPSA, Square
and GAN. Gradient attacks transfer through the frozen differentiable proxy;
they are not direct GB gradients. Square is an explicit tabular adaptation.
Feature-space constraints do not certify packet realizability or semantic
functionality. Finite-budget failure is not a robustness certificate.

## Output layout and identifiers

```text
artifacts/separate_test_attacks/
  cohorts/<dataset>.npy
  cohorts/<dataset>_probability.npy
  FGSM/<dataset>/shards/*.npz
  CW/<dataset>/shards/*.npz
  PGD/<dataset>/shards/*.npz
  ... one independent subtree for every listed attack ...
```

Each dataset remains separate inside each attack. The existing training folders
remain untouched. Test shards use source_split='test', source_dataset_id and
source_row for original provenance; cohort_position indexes that dataset's test
cohort, and mixed_position=-1 explicitly prevents interpreting a test record as
a training-index member. Original and target labels, probabilities, eligibility,
changed/success flags, norms, query counts, seed failures, budget exhaustion and
generation seed are saved. Native values use the same compact width64 format
and reconstruct the unchanged 157-coordinate GB input adapter.

The same attack engine handles both roles but receives explicit separate cohort,
path, seed and split settings. Six test worker lanes rotate across all datasets,
processing one cycle per dataset before returning; no dataset must finish before
another starts. GPU/score chunks remain 4096/512 and ART per-record chunks64.
These are resumable batches, not sample limits. Test worker failures are logged
separately from training worker failures.

## Checks and reporting

Every clean test prediction is checked against independently saved original GB
test results before generation. Initial shard audits verify exact test source
rows/labels, split identifiers, no training positions, frozen component hashes,
protected fields, budget, new-success definitions and fresh GB outputs. A full
every-shard audit follows all 60 jobs. Partial coverage is not full verification.

Historical GPU-margin/sigmoid outputs and current CPU probabilities can differ
slightly through floating-point accumulation. Preparation checks every record
with absolute probability tolerance 0.000002 **and identical threshold decisions**.
The observed maxima and zero decision differences are recorded per dataset in
configuration.json; this tolerance does not alter the model or attack settings.

[Live test report](TEST_ATTACK_REPORT.md),
[per-class CSV](artifacts/separate_test_attacks/progress.csv),
[status](artifacts/separate_test_attacks/status.json) and
[configuration and model hashes](artifacts/separate_test_attacks/configuration.json)
record progress and outcomes. Both attack directions are reported separately.
Partial prefixes cannot rank final attack strength. No defended IDS or
end-to-end edge performance claim is produced by sample generation.

```powershell
# Existing fitted components must already exist; do not launch duplicates.
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_test_control.py prepare
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_test_control.py launch
# Read-only audit:
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_test_control.py audit --partial
```
