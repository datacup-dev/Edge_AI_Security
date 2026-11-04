# Methods for the expanded comparison

This extends the initial MLP/CNN/recurrent comparison. The original MLP and its
results remain in `artifacts/adapted_experiment`; current overall selection is
recorded separately in `artifacts/overall_baseline.json` after the final audit.
See [live report](EXTENDED_EXPERIMENT_REPORT.md) for completion and results.

## Data and relation to the paper

The local [reference paper](../../journal.pone.0346801.pdf), pages 16–17,
provides the mixed-domain principle retained here: pool source training records at
their natural proportions and evaluate each dataset's test partition separately.
The paper's common-flow input representation is **not** reproduced. Our four
exports have different observation units and feature definitions. DataSense
contains one-second summaries; the other native exports are not established as
equivalent flows. Dataset-specific adapters were explicitly authorized.

Seed-42 stratified binary splits and training-fitted numeric preprocessing are
unchanged. All 35,030,746 training records remain available, with no domain
downsampling. Four validation sets and four test sets retain their original row
memberships. No validation/test rows fit scalers, quantile cuts or KNN indices.
The 153 native features have widths 64, 25, 39 and 25. Direct identifiers and
label columns are excluded; known dataset/schema identity routes each record.

Neural models learn a separate native encoder for each dataset, pooling scalar
embeddings into eight feature groups with width eight. Classical pipelines keep
all 153 native coordinates in disjoint blocks and append four dataset markers,
for 157 dimensions. Inactive blocks are abstract zeros for LR, SVM, NB, DT, RF
and KNN; GB treats them as unavailable, while preserving observed native zeros
as explicit sparse entries. Thus comparisons concern complete pipelines rather
than isolated classifier effects. Dataset identity can influence predictions.

## Models and training parameters

| Model | Architecture or settings | Training objective |
|---|---|---|
| Autoencoder | Group embedding 64 to 32 to 16; native decoder 16 to 64 to native width; classifier 16 to 32 to 1 | Binary cross entropy plus 0.1 reconstruction MSE on all classes |
| GNN | Eight feature-group nodes; three own/neighbor layers 8 to 32 to 32 to 16; head 128 to 64 to 32 to 1 | Binary cross entropy |
| Transformer | Eight tokens; model width 64, four heads, feed-forward width 128, two layers, learned group positions, mean pooling, 32-unit head | Binary cross entropy |
| GAN classifier | Per-domain conditional generator: noise 32 plus class indicator 2, hidden widths 64/128; discriminator: group embedding to 128/64 and benign/attack/fake logits | Discriminator real CE + 0.5 fake CE; generator conditional benign/attack CE |
| LR | Native 157-dimensional linear logit; no hidden layers or dropout | Unweighted binary cross entropy plus squared-weight norm/(2N), excluding intercept |
| SVM | Exact degree-2 polynomial kernel; gamma=1/157, coef0=1, C=1 | Mean hinge loss + squared mapped-weight norm/(2 C N); unregularized intercept |
| KNN | Five uniform-vote neighbors, exact pooled Euclidean distance, eps=0, KD-tree leafsize 128 | Retain all training reference records |
| DT | Depth 16, minimum leaf 50, sqrt feature candidates, seed 42 | sklearn classification tree |
| RF | 64 trees, depth 12, minimum leaf 50, sqrt feature candidates, bootstrap=True, max_samples=None, two CPU workers, seed 42 | Bootstrap forest over the full training population |
| NB | Gaussian conditional likelihoods, float64 accumulators, variance smoothing 1e-9 | Full-data sufficient statistics and empirical class priors |
| GB | XGBoost histogram trees, 300 rounds, depth 8, eta 0.1, max_bin 128, row fraction 1, column fraction 0.8, seed 42 | Binary logistic loss, training-only quantile cuts |

Neural models use Adam at learning rate 0.001, dropout 0.3, batches of 8192,
maximum 20 epochs and patience three. CUDA float16 autocast and GradScaler use
float32 master parameters. Each completed epoch consumes every original training
record once, at natural dataset proportions. GAN uses separate Adam optimizers
for its discriminator and generators. GAN inference uses logit_attack minus
logit_benign, conditioning the binary probability on the two real classes.
GAN-generated vectors are auxiliary training data, not adversarial attacks.

Adam uses PyTorch defaults for the remaining settings: betas (0.9, 0.999),
epsilon 1e-8, weight decay 0 and amsgrad=False. There is no learning-rate schedule
or gradient clipping. Batch schedules shuffle domain-homogeneous batches; seed
42 and the epoch index determine each epoch's shuffle. Record-level randomness
does not make native records into temporal sequences.

This is one seeded run per candidate, not a repeated-seed significance study.
Saved versions, memberships and checkpoints support reproduction; GPU kernels
and floating-point execution do not guarantee bitwise-identical retraining.

LR and SVM use Adam at 0.001, batch 8192, at most 20 full-data epochs, patience
three, and float32 without autocast. Neither finite stochastic optimizer provides
a global-optimality certificate. The SVM kernel map is exact independently of
optimization accuracy. Squared diagonal terms, sqrt(2)-weighted off-diagonal
terms and the constant/linear terms represent the complete kernel. Only
identically zero cross-domain monomials and proportional coordinates are merged;
there is no kernel approximation or support-point sampling.

GNN adjacency is an explicitly chosen feature-relation prior, visualized in the
report. Nodes are not endpoints/devices and edges are not observed traffic links.
Transformer positions identify feature groups, not time. The autoencoder is a
supervised bottleneck model rather than benign-only anomaly detection. Resource
caps and dropout differ from some settings in the paper; this is an adapted
experiment, not a replication of its architecture or performance claims.

GB attempts GPU histogram construction/training with four CPU helper threads;
memory errors fall back to CPU and are recorded in configuration. No row subset
is substituted. The final fixed 300-round model is validated without test tuning.
Its training-loss curve is descriptive, not an early-stopping validation curve.

## Exact KNN and computational checks

For a query in domain a and a training vector from a different domain b, disjoint
native blocks give squared distance norm(query)^2 + norm(reference)^2 + 2.
Consequently only the five minimum-norm references from each foreign domain can
belong to a global top five. An exact within-domain KD-tree search supplies its
five candidates; sorting their union yields pooled neighbors over the entire
35-million-record reference population. It does not restrict votes to one domain.

Constant training coordinates are omitted from the search index with their exact
query-dependent squared-distance contribution restored afterwards. A training-only
orthogonal covariance rotation retaining all 39 CICIoT2023 directions was tried
without whitening or truncation. Although it preserves Euclidean distance,
observed query throughput was worse. Final evaluation therefore resumes the
original native-coordinate index, including its 2,629,632 saved CIC validation
predictions. The slower rotated development index is archived separately and
excluded from final model ranking. Index selection uses runtime, not labels or
accuracy. Cross-domain norms always use original native coordinates.
Floating-point roundoff and ties can affect which equally distant references are
returned; tree search uses eps=0, with no approximate-neighbor tolerance.

CPU parallelism is benchmarked on identical unlabelled validation features at
4, 8 and 16 workers. Neighbor IDs, distances and votes must match exactly across
worker counts. This feature-only latency trial selects scheduling, not accuracy
or model hyperparameters. The saved worker benchmark documents its query rows.
Prediction cursors flush each completed 8192-row block. Resuming loads the saved
rotation and references together, preserving prior predictions.

Synthetic independent checks compare pooled KNN against brute-force distances,
the polynomial-map Gram matrix against its direct kernel, sparse adapters against
dense blocks and NB against sklearn GaussianNB. These are correctness checks,
not sampled final model runs. Their results are saved under expanded_classical.

## Checkpoint selection, metrics and audit

Neural/LR/SVM checkpoints maximize the pair (mean of four validation macro-F1
scores, mean of four validation average-precision scores). Domains receive equal
weight in this validation criterion despite unequal training sizes. Each final
candidate is frozen before its own four test evaluations. The new winner uses
the same validation pair; exact ties among new candidates prefer smaller saved
deployment artifacts. Compare this winner with the prior MLP pair, retaining MLP
on an exact pair tie. No test metric selects the baseline.

All binary decisions use probability >=0.5 (equivalently nonnegative signed
margin where applicable). Outputs include labels, probabilities, predictions,
source row IDs, confusion counts, normalized confusion plots, precision, recall,
F1, accuracy, balanced accuracy, MCC, ROC-AUC, average precision, log loss and
Brier score as defined by the evaluation implementation. ROC/AP use raw signed
margins when available, avoiding sigmoid saturation ties. SVM sigmoid scores are
uncalibrated; probability metrics must be interpreted accordingly. Probabilities
and clipping precision can differ between pipelines.

The final audit checks 44 new model/test combinations, labels, original test
membership/order, finite probability ranges, threshold decisions, confusion
counts, accuracy, full epoch totals, validation aggregation and baseline choice.
It also checks every KNN reference vector and label against the original training
membership, including the stored rotation and constant-coordinate treatment.
This audit is not a second inference pass over every test row. A separate loader
check compares selected saved test probabilities with fresh inference samples.

Original test results were already inspected in the preceding stage. This is a
follow-on comparison on reused holdouts. Random record splitting does not isolate
duplicate records, captures, devices or time; these effects remain limitations.
This stage establishes neither unseen-domain transfer nor adversarial robustness
nor a real-time edge deployment. No SHAP/IG/counterfactual or defense result is
implied by the classifier comparison.

## Reproduction and saved artifacts

Run commands from the workspace root with the project virtual environment:

```powershell
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/train_cpu_classical.py
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/train_gradient_boosting.py
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/report_expanded.py
& Coding/edge_ids/.venv/Scripts/python.exe Coding/edge_ids/verify_expanded.py
```

On the current workstation, run full KNN and GB sequentially to avoid exhausting
32 GiB RAM. The initial `finish_expanded.py` controller was replaced by
`complete_remaining.py` to run GB first, then resume exact KNN from its saved
cursor, refresh the report, audit artifacts, check the overall loader and execute
the notebook. Do not launch a second controller or duplicate
training worker. Completed markers allow scripts to skip finished models.

`expanded_neural` and `expanded_classical` contain configurations, histories,
checkpoints, candidate selection records, separate four-domain predictions and
plots. `extended_report` contains combined metrics, validation ranking, figures,
code hashes and verification. `overall_baseline.py` loads only an audited overall
selection. Its inputs must already follow the saved native preprocessing and
feature order; it is not a raw packet extractor. KNN deployment requires its
complete reference corpus and index reconstruction, not just its JSON metadata.
