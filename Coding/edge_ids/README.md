# Edge IDS mixed-domain experiment

Separate GB defense retraining is documented in
[ADVERSARIAL_DEFENSE_TRAINING.md](ADVERSARIAL_DEFENSE_TRAINING.md). Live status is
`artifacts/adversarial_defenses/status.json`; partial training snapshots produce
provisional defenses, with no validation/test fitting.

Clean + adversarial validation datasets have 60 separate snapshots, one per
attack and original validation dataset. See
[AUGMENTED_VALIDATION.md](AUGMENTED_VALIDATION.md) for counts and
`augmented_validation.batches` usage. Training and test records are excluded;
incomplete attack generation is marked partial.

Clean + adversarial test datasets now have 60 independent snapshots, one per
attack and original test dataset. See [AUGMENTED_TEST.md](AUGMENTED_TEST.md) for
counts and `augmented_test.batches` usage. Training and validation are excluded;
incomplete attack generation is marked partial.

Per-attack clean + adversarial training datasets are described in
[AUGMENTED_TRAINING.md](AUGMENTED_TRAINING.md). Their snapshot manifests live in
`artifacts/augmented_training/<attack>/manifest.json`; use `augmented_training.batches`
to read the concatenated features and original labels. Partial generation is
explicitly marked, and validation/test samples are excluded.

Validation attack outputs have their own [live report](VALIDATION_ATTACK_REPORT.md)
and [process record](VALIDATION_ATTACKS.md), separate from both training and test
outputs and generated using frozen training-fitted attack components.

Held-out attack generation now has an independent [test report](TEST_ATTACK_REPORT.md)
and [test process record](TEST_ATTACKS.md). It reuses frozen training-fitted
components; adversarial training and test outputs remain separate.

Current generation covers **all mixed-training records from both classes**, with
fifteen separate attack outputs. See the [live report](MIXED_TRAINING_ATTACK_REPORT.md)
and [complete process and settings](MIXED_TRAINING_ATTACKS.md).

The expanded comparison is tracked in [Extended report](EXTENDED_EXPERIMENT_REPORT.md),
with full settings in [Extended methods](EXTENDED_EXPERIMENT_METHOD.md).
The five-model results below describe the initial phase. The current overall
baseline is separate from that phase's MLP: `artifacts/overall_baseline.json`
is available only after the expanded comparison and artifact audit finish.

Start with [Complete process documentation](FULL_PROCESS_DOCUMENTATION.md) for
the full workflow, exact feature/epoch catalogs, all results, development history,
limitations, artifact inventory and reproduction instructions.

[Training parameters and validation rule](TRAINING_PARAMETERS.md) provides the
explicit optimizer, learning rate, dropout, epoch budget, stopping criteria and
actual checkpoint epochs for each model.

This stage prepares binary labels and reproducible, stratified 70/15/15 row
assignments for the original dataset exports. A mixed training index now combines
all training memberships and binary labels. Shared-classifier training with
dataset-specific encoders is complete for five neural candidates. Common
flow-feature harmonization remains pending.

Run from the workspace root:

```powershell
python Coding/edge_ids/prepare_splits.py
```

The default random seed is 42. The first stratified split reserves 30% of rows;
the second divides that holdout equally into validation and test. Integer
rounding can make validation and test differ by one row. No resampling or
class balancing is applied.

## Selected releases and label provenance

| Dataset | Records used | Label mapping |
|---|---|---|
| DataSense | Matched one-second attack and benign CSVs inside the supplied nested archive | `label1`: benign=0, attack=1 |
| Edge-IIoTset | `DNN-EdgeIIoT-dataset.csv` only | `Attack_label`: 0=benign, 1=attack |
| CICIoT2023 | All supplied scenario CSVs | `Benign_Final` folder=0, other supplied scenario folders=1 |
| X-IIoTID | `X-IIoTID dataset.csv` | `class3`: Normal=0, Attack=1 |

Other DataSense window resolutions are excluded to avoid representing the same
traffic repeatedly at different resolutions. Edge ML, DNN, and raw CSV releases
are not combined because they overlap. CICIoT2023 folder labels are scenario
labels, not verified per-flow attack annotations.

## Artifact format

Each dataset has an NPZ and JSON under `artifacts/native_splits/`.
NPZ `labels` contains binary labels; NPZ `split` contains 0=train,
1=validation, 2=test. Both arrays follow the source CSV row order.
The JSON records ordered source files/archive members, their row offsets,
label provenance, split counts, random seed, and array SHA-256 digests.
CSV row positions are zero-based and exclude headers. Original feature data
stays in its original files; it is not copied into four large exports.

```python
import numpy as np
with np.load('Coding/edge_ids/artifacts/native_splits/X-IIoTID.npz') as data:
    labels = data['labels']
    train_rows = np.flatnonzero(data['split'] == 0)
    validation_rows = np.flatnonzero(data['split'] == 1)
    test_rows = np.flatnonzero(data['split'] == 2)
```

## Scientific scope

These are native-record splits, not harmonized network-flow splits. Packet
exports, connection summaries, and device windows remain different units.
Do not concatenate their feature columns or fill incompatible fields with zeros.
No feature imputation, normalization, selection, or learned preprocessing has
been fitted. Once a common representation is verified, any learned statistics
must be fitted exclusively on training rows. Validation and test remain held out.

Stratification preserves label proportions; it does not prevent duplicate,
capture, device, session, or temporally related observations from crossing splits.
These requested random splits therefore do not establish independence from those
sources of leakage. If conversion changes the observation unit to flows, split
assignments must be recreated or reconciled at that unit before evaluation.

The user subsequently authorized dataset-specific encoders. The adapted neural
experiment is documented in `EXPERIMENT_METHOD.md`; it uses native per-domain
features, not the paper's shared eight-feature flow representation. Its actual
training settings are in `artifacts/adapted_experiment/configuration.json`.

The paper's mixed-domain scenario uses compatible NetFlow features and natural
dataset proportions. The authorized adapted experiment has completed mixed
training and separate test evaluation; common flow-level harmonization remains
pending. Validation and test datasets remain separate.

Completed results: [experiment report](EXPERIMENT_REPORT.md),
[executed preview notebook](preview.ipynb), [run status](RUN_STATUS.md), and
[methods and limitations](EXPERIMENT_METHOD.md). MLP/DNN was selected using
validation only. All five models were evaluated on all four test sets.

## Mixed training index

```powershell
python Coding/edge_ids/combine_training.py
```

This includes every training row once at natural dataset proportions, shuffling
with seed 42. The source-backed index contains 35,030,746 records and occupies
about 210 MB. It references original feature data rather than copying incompatible
CSV schemas into a single table.

```python
import numpy as np
records = np.load('Coding/edge_ids/artifacts/mixed_training/records.npy',
                  mmap_mode='r', allow_pickle=False)
batch = records[:1024]
dataset_ids = batch['dataset_id']
source_rows = batch['source_row']
labels = batch['label']
```

Dataset IDs are 0=DataSense, 1=Edge-IIoTset, 2=CICIoT2023, 3=X-IIoTID.
`source_row` is the global row within that dataset's ordered native sources.
Use the source offsets in `artifacts/native_splits/<dataset>.json` to find the
backing CSV/archive member and local row. `manifest.json` records counts,
proportions, hashes and provenance. After saving, the script checks exact training
membership, no duplicate source rows, binary label consistency, and exclusion of
all validation/test rows. This index is not a shared numeric model input.
