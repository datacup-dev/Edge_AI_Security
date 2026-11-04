# Run status

## Separate GB defense retraining

Started a sequential run of 15 independent GB defenses using fixed augmented
training snapshots. FGSM begins with 38,831,834 rows; generation-incomplete
snapshots yield provisional defenses. No validation/test fitting or defense
evaluation is performed in this step. This entry records launch, not completion.
Live status: `artifacts/adversarial_defenses/status.json` and each attack's
`progress.json` / log. [Method and settings](ADVERSARIAL_DEFENSE_TRAINING.md).

<!-- validation-attacks-start -->
## Separate adversarial validation generation

Status: **running**; 78,836,717/112,598,835 candidates; 51/60 jobs complete.
All four original validation sets, both classes. Frozen training-fitted attack components; no held-out fitting.
[Live report](VALIDATION_ATTACK_REPORT.md). [Protocol](VALIDATION_ATTACKS.md).
<!-- validation-attacks-end -->

<!-- test-attacks-start -->
## Separate adversarial test generation

Status: **failed_worker**; 101,643,897/112,598,865 candidates; 55/60 jobs complete.
All four original test sets, both classes. Frozen training-fitted attack components; no held-out fitting.
[Live report](TEST_ATTACK_REPORT.md). [Protocol](TEST_ATTACKS.md).
<!-- test-attacks-end -->

<!-- both-class-attacks-start -->
## Both-class mixed-training adversarial generation

Status: **running**; saved 420,412,932/525,461,190 records; 10/15 methods complete.
All original mixed-training records, both classes. Fifteen separate outputs. No held-out outputs.
[Live report](MIXED_TRAINING_ATTACK_REPORT.md). [Process and parameters](MIXED_TRAINING_ATTACKS.md).
<!-- both-class-attacks-end -->

## Expanded comparison

All 11 new candidates completed; baseline selection uses validation only. Artifact status: verified_complete.

[Expanded report](EXTENDED_EXPERIMENT_REPORT.md). [Methods and parameters](EXTENDED_EXPERIMENT_METHOD.md).
Completed new candidates: 11/11; recorded new test evaluations: 44/44.

Best new model: **GB**. Overall baseline: **GB**.
Overall mean validation macro-F1: 0.962042; prior MLP: 0.943363.
Selection uses validation only. Separate four-dataset test metrics and confusion/ROC figures are linked in the report.

SVM uses the user-selected exact degree-2 polynomial kernel; KNN retains the full training reference population with exact search. The sections below preserve the initial five-model phase.

Full audit trail and process record: [Complete process documentation](FULL_PROCESS_DOCUMENTATION.md).

Exact training settings: [Training parameters and validation rule](TRAINING_PARAMETERS.md).

Independent native-record splits completed on 2026-10-31 using seed 42.
Labels: benign=0, attack=1. Row assignments: train=0, validation=1, test=2.

| Dataset | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| DataSense (1 second) | 159,033 | 34,079 | 34,079 | 227,191 |
| Edge-IIoTset (DNN release) | 1,553,440 | 332,880 | 332,881 | 2,219,201 |
| CICIoT2023 | 32,743,690 | 7,016,505 | 7,016,505 | 46,776,700 |
| X-IIoTID | 574,583 | 123,125 | 123,126 | 820,834 |

Completed:

- Binary labels and stratified 70/15/15 splits for the selected releases.
- All selected rows assigned exactly once to one split.
- Per-class split counts checked against requested ratios, allowing integer rounding.
- Saved row assignments, per-class counts, source offsets, seed and array hashes.
- Original datasets preserved without modification.

Mixed training membership completed on 2026-10-31:

- Combined all 35,030,746 training records into a shuffled, source-backed index.
- Labels: 2,290,435 benign and 32,740,311 attack.
- Natural dataset proportions preserved; no balancing or downsampling.
- Verified exact membership and uniqueness against each training split after saving.
- Validation and test rows excluded; their original splits remain unchanged.
- Saved `artifacts/mixed_training/records.npy` and `manifest.json`.

This index combines labels and references to original records. It is not yet a
common flow-feature matrix. Common flow extraction and harmonization remain
pending; the authorized adapted encoder training is recorded below.

Not completed:

- Common flow extraction/semantic feature harmonization.
- A common flow-feature matrix (native per-domain numeric preprocessing is complete).
- Adversarial training and explanation methods.
- End-to-end extraction, preprocessing, window-delay and edge-device validation.

## Authorized adapted neural experiment

On 2026-10-31 the user authorized dataset-specific encoders feeding shared
classifiers. Numeric input preparation has completed for all four datasets:
DataSense 64 features, Edge-IIoTset 25, CICIoT2023 39, X-IIoTID 25.
Each domain's scaling statistics use its training rows only. This is not a
common-flow representation.

Trained candidates: MLP/DNN, CNN1D, LSTM, GRU, BiLSTM. Recurrent models operate
on feature tokens, not chronological traffic. No CNN2D/3D variants are claimed.
The isolated CUDA environment uses the workstation's RTX 4060.

Final comparison uses the corrected eight-group dataset encoders:

| Candidate | Training status | Best mean validation macro-F1 |
|---|---|---:|
| MLP/DNN | Complete, 20 epochs | 0.943363 |
| CNN1D | Complete, 20 epochs | 0.942221 |
| LSTM | Complete, 14 epochs (early stopped) | 0.933605 |
| GRU | Complete, 11 epochs (early stopped) | 0.932789 |
| BiLSTM | Complete, 9 epochs (early stopped) | 0.926434 |

All five final training runs and all 20 separate test evaluations are complete.
Development trials in `artifacts/pilot_feature_tokens` are not part of this final
comparison.

Selected provisional baseline: **MLP/DNN**, 21,137 trainable parameters,
checkpoint epoch 19 of 20, mean validation macro-F1 **0.943363**. The decision
was frozen before test evaluation; test metrics did not determine the baseline.

| Baseline test dataset | Accuracy | Macro-F1 | ROC-AUC | Attack recall | False-positive rate |
|---|---:|---:|---:|---:|---:|
| DataSense | 0.898823 | 0.889688 | 0.909903 | 0.767903 | 0.014669 |
| Edge-IIoTset | 0.995951 | 0.994897 | 0.999551 | 0.995571 | 0.003908 |
| CICIoT2023 | 0.991933 | 0.913589 | 0.997779 | 0.995425 | 0.153325 |
| X-IIoTID | 0.973409 | 0.973380 | 0.996349 | 0.965867 | 0.019442 |

DataSense attack miss rate is 23.21%; CICIoT2023 benign false-positive rate is
15.33%. This is a provisional experimental baseline, not deployment validation.

Verified exact held-out membership, labels, probability ranges, thresholds and
confusion counts for all 20 evaluations. All completed epochs used 35,030,746
training observations. A separate loader reproduced saved probabilities exactly
for 32 observations per dataset. An idempotent completion run exited successfully
without retraining or repeating model test inference.

Saved 80 warm workstation CPU/GPU inference timing configurations. These exclude
extraction, scaling, transfer, queueing and window collection, and do not establish
end-to-end real-time or edge-device performance.

Results: `EXPERIMENT_REPORT.md`, `preview.ipynb`,
`artifacts/report/test_metrics.csv`, `artifacts/report/inference_latency.csv`.
Checkpoints, predictions, source rows, metrics, confusion/ROC/PR figures, learning
curves and verification records are in `artifacts/adapted_experiment/`.

Canonical completion record: `artifacts/adapted_experiment/progress.json`.
Selection record: `baseline.json`; audit: `verification.json`.
See EXPERIMENT_METHOD.md for the authorized adaptation and its limitations.

The saved splits refer to supplied native records, not reconstructed flows.
They do not guarantee isolation of duplicates, captures, sessions or overlapping
observations. Conversion to a different observation unit may require new splits.
See README.md for release selection and label provenance, and
`artifacts/native_splits/*.json` for class counts and source records.
