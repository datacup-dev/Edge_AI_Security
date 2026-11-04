# Separate adversarial outputs from all mixed-training records

The subsequent [test-generation stage](TEST_ATTACKS.md) is stored separately and
reuses frozen components. This document describes the original training run;
its training membership and outputs are unchanged by that addition.

Run scope: 2026-11-01, Asia/Seoul. This is a new run after the user deleted the
previous attack files. Deleted candidates and attack models are not reused.
[Live progress](MIXED_TRAINING_ATTACK_REPORT.md) is updated while workers run.

## Exact scope, populations and labels

The user explicitly selected **all mixed-training records from both classes**.
Every one of the following 15 methods processes the exact same original mixed
training membership, without final subsampling:

FGSM, C&W (folder `CW`), PGD, JSMA, DeepFool, BIM, MIM, Transfer, Boundary,
HopSkipJump, ZOO, NES, SPSA, Square and GAN. DIGFuPAS remains excluded.

| Source | Training records | Benign | Attack |
|---|---:|---:|---:|
| DataSense | 159,033 | 95,760 | 63,273 |
| Edge-IIoTset | 1,553,440 | 1,130,950 | 422,490 |
| CICIoT2023 | 32,743,690 | 768,734 | 31,974,956 |
| X-IIoTID | 574,583 | 294,991 | 279,592 |
| Total | 35,030,746 | 2,290,435 | 32,740,311 |

The authoritative index is `artifacts/mixed_training/records.npy`. Its SHA256 is
checked against its original manifest, and each dataset's complete training row
membership and labels are checked against the encoded split. Its shuffled order
is preserved in every attack output. Fifteen times 35,030,746 means
**525,461,190 requested candidate records**. No validation/test records are
included, and this request does not generate held-out adversarial test sets.

True labels are retained. The target is **1-original_label**: benign→attack and
attack→benign. GB must originally classify the record correctly for it to count
as an eligible new success. Initially wrong records remain unchanged and flagged
not attempted. Success requires eligibility, a representable feature change and
an opposite-class GB prediction. Failed, unchanged and duplicate candidates are
retained. Saved output coverage is not a count of successful adversarial examples.

## Frozen baseline and feature representation

The target is the audited overall GB/XGBoost baseline selected by validation,
loaded from `artifacts/overall_baseline.json`. It is never retrained or modified
in this stage. Its training settings remain in EXTENDED_EXPERIMENT_METHOD.md.

The datasets still have distinct native schemas and observation units. This is
not a new common physical-flow feature conversion. Native widths are
64/25/39/25. To query GB, native values are inserted into its original 153-feature
block namespace plus four domain markers. All unavailable blocks and inactive
markers remain missing/NaN; the active marker stays 1. Dense predictions are
checked against the original sparse adapter, including observed zero values.
Source domain identity is fixed and cannot be attacked.

## Constraints and output storage

All final candidates use a declared **standardized native Linf budget of 0.3**.
This is not a raw byte/packet budget or percentage. Mutable fields are in the
packet-count, size and timing/rate groups, excluding port/protocol/flag/checksum/
header and endpoint-cardinality fields and training-constant columns. Exact
feature names, masks and training-only ranges are saved in configuration.json.

Values are projected to the intersection of the local box and training ranges,
while retaining an original outlier as a feasible endpoint. Float32 endpoints
round inward before serialization. Protected coordinates are unchanged. This
feature-space policy does not certify integer counts, aggregate identities,
protocol functionality, semantic label invariance or packet realizability.

Each attack has its own independent directory:

```text
artifacts/mixed_training_attacks/
  FGSM/shards/*.npz
  CW/shards/*.npz
  PGD/shards/*.npz
  JSMA/shards/*.npz
  DeepFool/shards/*.npz
  BIM/shards/*.npz
  MIM/shards/*.npz
  Transfer/shards/*.npz
  Boundary/shards/*.npz
  HopSkipJump/shards/*.npz
  ZOO/shards/*.npz
  NES/shards/*.npz
  SPSA/shards/*.npz
  Square/shards/*.npz
  GAN/shards/*.npz
```

To avoid repeatedly copying clean data across 525 million candidates, shards
store `x_adversarial_native` as float32 [N,64] with trailing NaN padding, plus
`mixed_position` identifying the original shared clean record. This compact
format is not a truncated model representation: a domain's full native feature
list is retained, and the exact GB namespace is reconstructed when queried.
Do not feed the padded 64-column array directly into the 157-column GB model.

Each file also stores labels, target_labels, clean_probability,
adversarial_probability, eligible/attempted/changed/success, L0/L2/Linf,
per-record GB-generation query counts, query-budget exhaustion,
initialization_failed, attack name, source_split=train and generation seed.
Dataset IDs and source rows are recovered unambiguously from mixed_position.
The `load_shard` helper in adv_mixed_data.py reconstructs clean and candidate
native rows and the full source record metadata. Labels remain original labels
for any later supervised use; target labels are not replacement ground truth.

## Access to GB and training-only preparation

Tree decisions do not supply useful ordinary input gradients. FGSM/C&W/PGD/
JSMA/DeepFool/BIM/MIM therefore transfer from a GB-distilled MLP. Transfer uses
the independent original supervised MLP instead. Final outcomes always use GB,
not the surrogate. These are not claimed as direct GB white-box gradient attacks.

Boundary/HopSkipJump receive only hard decisions, including seed selection.
ZOO/NES/SPSA/Square use probabilities. The GAN learns bounded perturbations
conditioned on the desired opposite label. No validation/test records fit these
components, and neither held-out metrics nor attack outcomes select settings.

| Component | Optimizer / LR | Betas | Fitting dropout | Batch | Epoch limit and selection |
|---|---|---|---:|---:|---|
| GB-distilled MLP | Adam / 0.0001 | 0.9 / 0.999 | 0.3 | 8192 | 3 full training epochs; fixed final checkpoint |
| Conditional GAN generator | Adam / 0.0001 | 0.5 / 0.9 | 0 | 4096 | 5 full training epochs; fixed final checkpoint |
| Conditional GAN discriminator | Adam / 0.0002 | 0.5 / 0.999 | 0 | 4096 | One discriminator update per generator update |

All use epsilon=1e-8, weight_decay=0, amsgrad=False, float32, no learning-rate
schedule and no gradient clipping. Preparation starts from seed42; epoch
permutations use 42+epoch. Both MLPs and GAN generators are frozen in eval mode
during generation, so surrogate fitting dropout is disabled for attacks.

The proxy warm-starts the prior supervised MLP and distills cached GB soft
probabilities on all 35,030,746 original training records each epoch. BCE sums
over native encoder groups and divides by actual mixed batch size. The original
warm-start model was validation-selected in the earlier study; this new fit uses
a fixed three epochs without validation selection or early stopping. Its new
fidelity has not been independently evaluated, so failed transfer is not proof
of GB robustness.

The GAN also trains on all mixed-training rows from both classes, with target
1-label. Native input, target one-hot2 and Gaussian noise16 feed hidden widths
128/64 and a tanh epsilon-scaled perturbation output. Its 64/32 discriminator is
conditioned on the original true label for realism. Generator loss is target
softplus(-signed_proxy_logit) +0.05 realism loss +0.01 mean squared perturbation.
Discriminator loss is real/fake logistic loss. Four native branches are jointly
optimized in mixed batches. Histories and resumable optimizer/RNG states are
saved separately under surrogate/ and GAN_model/. No defended IDS is trained.

## Attack variants and fixed input-search budgets

| Attack | Settings |
|---|---|
| FGSM | One targeted sign step, epsilon 0.3; gradient uses opposite-label log probability |
| C&W | Targeted L2 plus signed-logit hinge, additional local box; three constant searches; initial constant 0.1; 60 Adam iterations/search, lr 0.01, confidence 0 |
| PGD | 20 projected updates of 0.03, random start |
| BIM | 20 projected updates of 0.03, no random start |
| MIM | 20 updates of 0.03, decay1, L1-normalized momentum |
| JSMA | Positive two-coordinate target-probability saliency, theta 0.05, gamma 0.2 of all native input coordinates; immutable fields excluded; local projection |
| DeepFool | Binary signed-logit boundary linearization, 20 updates, overshoot 0.02, local projection |
| Transfer | Targeted 20-step PGD on original MLP, random start, step 0.03 |
| NES | 20 updates, 16 antithetic Gaussian directions, sigma 0.05, step 0.03; accept GB target-probability improvements |
| SPSA | Same budget with Rademacher directions |
| Square | Explicit tabular block-search adaptation, 100 updates, initial mutable-coordinate fraction 0.8; signed epsilon patches, greedy acceptance |
| Boundary | ART targeted BoundaryAttack, 15 iterations, five trials, sample_size8, provided local target seed |
| HopSkipJump | ART targeted L2 attack, 10 iterations, max_eval64, init_eval16, provided local target seed |
| ZOO | ART coordinate Adam, 20 iterations, two constant searches, lr 0.02, finite difference 0.02, eight coordinate probes; resize/importance disabled |
| GAN | Frozen label-conditioned perturbation generator from five-epoch full mixed-training fit |

C&W uses Adam defaults betas=(0.9,0.999), epsilon=1e-8, weight_decay=0 for input
tanh variables. ART ZOO's coordinate Adam uses the same betas/epsilon. These are
input-search iterations, not classifier training epochs. Additional box/mask
constraints make these constrained variants. Square is explicitly adapted to
tabular coordinate blocks; no image geometry is asserted. No parameter search or
attack-strength selection uses held-out data.

Boundary/HopSkipJump start with **64 vectorized random local candidates per
eligible record**, using hard decisions only. If no feasible opposite-class seed
is found, that record is retained with initialization_failed=True; it is not
dropped or counted as an adversarial success. Otherwise ART refines the provided
seed. This avoids per-record repeated random initialization and improves full-
population throughput without changing the requested source population.

Boundary/HopSkipJump/ZOO have at most **4,096 GB-record queries per source**,
including initialization. On cap exhaustion, keep the closest feasible target
candidate already found; otherwise keep the original (ZOO can retain its best
score candidate). Exhaustion is explicit. ART ZOO uses translated delta inputs
with scalar bounds +/-0.3 to handle coordinate-noise clipping; the native-space
oracle restores constraints. The installed library is not patched.

These finite budgets, flat tree scores, seed failures and surrogate mismatch all
limit attack strength. Failure is not a robustness certificate. Raw source
labels are preserved without a claim that network functionality is validated.
See [official ART documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html).

## Metrics, checks, resumption and status

Results are reported separately by original label: benign→attack and attack→
benign. Each has eligible counts, changed counts, new successes/ASR, queries,
caps, initialization failures and norms. Existing GB errors are excluded from
new success denominators. These are **training-set vulnerability measurements**,
not held-out generalization/robustness. Partial prefixes cannot rank final attack
strength, and the majority attack class must not hide benign-direction results.

Preflight checks both true classes for every method in each dataset, compares
the GB input adapter to its original sparse representation and checks known
linear-boundary direction/sign behavior independently. Small correctness cases
are not a sampled final experiment. Saved-output audits verify labels, opposite
targets, exact mixed positions, source membership, mask/budget/NaN policy,
norms and fresh GB predictions. Partial audits check first saved shards; the
final audit streams every shard and checks complete coverage without gaps.

Worker lanes are gradient, score, Boundary, HopSkipJump, ZOO and GAN. Chunks are
4096 for gradient/GAN, 512 for score and 64 for per-record query methods. Seeds
depend on method index and mixed offset. Atomic shard/cursor commits and Windows
file-lock retries allow resumption. A cursor is progress, not a sample limit.
Source hashes, dependency versions, baseline/index hashes, parameters, logs and
checkpoints are recorded. Bitwise GPU retraining is not guaranteed.

```powershell
# From workspace root; do not start duplicates while supervisor.lock exists.
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_mixed_prepare.py
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_mixed_check.py
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_mixed_control.py launch
# Read-only snapshots:
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_mixed_audit.py --partial
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/adv_mixed_control.py preview
```

No prior deleted attack artifacts are reused. Original datasets, splits and
classifier/baseline weights remain intact. Full completion is claimed only when
all fifteen methods cover all 35,030,746 sources and the final audit passes.

Recorded preparation evidence is available in
[preparation_summary.json](artifacts/mixed_training_attacks/preparation_summary.json),
including optimizer parameter groups read from saved checkpoints. The complete
[training curves](artifacts/mixed_training_attacks/preparation_curves.png) show all
three surrogate and five GAN epochs; the fixed final checkpoints were used even
when GAN training loss was non-monotonic. These are not validation curves or a
claim of generator attack success.
