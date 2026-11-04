# Separate adversarial validation datasets

Execution update: concurrent workers exhausted Windows paging memory. Saved
shards were retained, and validation generation resumed with
`adv_validation_serial.py`, which rotates all attack/dataset jobs one at a time.
Memory allocation failures retain the saved cursor and retry after 30 seconds.
Earlier failure traces remain in `failure_history`. The initial partial audit
passed for all 60 jobs; it does not certify completion. The serial scheduler
runs the full audit only after every job finishes.

This stage processes every record in the four original validation partitions,
from both classes. It reuses the frozen GB baseline, GB-distilled MLP, original
supervised MLP and conditional GAN already fitted on training data. It does not
fit or tune those components on validation records.

The original baseline was selected using clean validation results in the earlier
classifier study; that historical selection is unchanged. This new stage only
crafts individual adversarial validation candidates with the previously fixed
attack settings. It does not select stronger budgets based on test outcomes or
retrain a defended IDS. Validation candidates may support a later explicitly
authorized defense-selection stage; test candidates remain evaluation-only.

## Scope and separate outputs

All original split==1 records are used, with no final downsampling. Train,
validation and test memberships are distinct. Fifteen methods × four datasets
produce 60 independent jobs and **112,598,835 requested candidate records**
(7,506,589 validation sources per method).

The fifteen outputs are FGSM, C&W (`CW`), PGD, JSMA, DeepFool, BIM, MIM,
Transfer, Boundary, HopSkipJump, ZOO, NES, SPSA, Square and GAN. DIGFuPAS is
excluded. Each attack/dataset has its own path:

```text
artifacts/separate_validation_attacks/<attack>/<dataset>/shards/*.npz
```

Training outputs remain under `artifacts/mixed_training_attacks/`, and test
outputs remain under `artifacts/separate_test_attacks/`. They are not pooled or
overwritten. Validation cohorts are saved separately in cohorts/<dataset>.npy.
Source labels are retained; the target is 1-label. Original GB errors are kept
unchanged and excluded from newly successful flips. Saved candidates include
failures, unchanged cases, seed failures and query-budget exhaustion, rather
than falsely counting every output as a successful adversarial example.

## Components, settings and checks

The [shared attack process](MIXED_TRAINING_ATTACKS.md) records every optimizer,
learning rate, dropout setting, epoch/iteration limit, access type, mask and
budget. They are unchanged for this stage. The surrogate and GAN were fitted
for three/five fixed full-training epochs, with no new validation early stopping.
Their weights and the original MLP are hashed before validation generation;
workers and audits recheck those hashes. GB weights are also checked unchanged.

Gradient methods transfer through the frozen differentiable proxy, with actual
success measured on GB. Boundary/HopSkipJump see only hard labels; ZOO/NES/SPSA/
Square see scores. Square is explicitly a tabular block adaptation. The final
Linf bound is 0.3 standardized native units; protected fields and domain routing
stay fixed. Native feature-space feasibility does not establish packet
realizability or semantic label invariance. Failed finite-budget or proxy attacks
do not certify model robustness.

The original validation rows/labels are verified against encoded split masks,
and disjointness from the mixed training pool is checked. Where test cohorts
exist, they are verified to be original split==2, excluding overlap with split1.
For every validation source, clean GB predictions from the native dense adapter
are compared against the independent original sparse adapter. The check requires
absolute probability agreement within 0.000002 and identical class decisions.
Observed differences are recorded, not used to tune attack strength.

Validation shards use source_split='validation', source_dataset_id, source_row
and cohort_position. mixed_position=-1 prevents confusing a validation row with
an original training-index position. Original/target labels, compact native
candidate values, probabilities, success flags, L0/L2/Linf, query counts, seed
failures and cap exhaustion are saved. Native vectors use the shared width64
storage convention and reconstruct the same unchanged 157-coordinate GB adapter.

The held-out engine is shared with test generation, but role, split code,
output directory, metadata and random seeds are explicitly distinct. Six worker
lanes rotate across the four datasets with resumable chunks. No duplicate
supervisor should be started while its lock file exists.

Initial saved-shard audits check memberships, source roles, labels, component
hashes, protection/budget, success definitions and fresh GB outcomes. The final
audit covers every shard after all 60 jobs finish. Partial audits do not certify
full generation. Partial rates cannot rank final attack strength because methods
can cover different prefixes. Both label directions are reported separately.

[Live validation report](VALIDATION_ATTACK_REPORT.md),
[per-class metrics](artifacts/separate_validation_attacks/progress.csv),
[status](artifacts/separate_validation_attacks/status.json), and
[cohort counts, adapter checks and model hashes](artifacts/separate_validation_attacks/configuration.json)
provide current evidence. No model-fitting or real-time edge claim follows from
creating these samples.

```powershell
# From workspace root; do not launch duplicate workers.
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_validation_control.py prepare
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_validation_control.py launch
# Read-only initial audit:
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_validation_control.py audit --partial
```
