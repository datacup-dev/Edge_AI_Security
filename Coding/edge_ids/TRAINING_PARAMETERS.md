# Training parameters and validation rule

For the subsequent autoencoder/GNN/transformer/GAN and classical candidates,
see [Expanded methods and parameters](EXTENDED_EXPERIMENT_METHOD.md).
Current overall selection is in the [Extended report](EXTENDED_EXPERIMENT_REPORT.md).

This reference records the completed five-model adapted mixed-domain experiment.
It summarizes the actual code and saved configuration; it is not a new run.

## Shared training settings

| Parameter | Actual setting |
| --- | --- |
| Optimizer | Adam |
| Initial and fixed learning rate | 0.001 |
| Adam beta1 / beta2 | [0.9, 0.999] |
| Adam epsilon | 1e-08 |
| Weight decay | 0 |
| AMSGrad | False |
| Dropout probability | 0.3 |
| Maximum training epochs | 20 |
| Early-stopping patience | 3 |
| Effective batch size | 8192 |
| Initial physical microbatch size | 8192 |
| Loss | Unweighted BCEWithLogitsLoss |
| Positive class | attack = 1; benign = 0 |
| Prediction threshold | Attack probability >= 0.5 |
| Python / NumPy / Torch seed | 42 |
| Training observations per completed epoch | 35030746 |
| Sampling | Natural domain proportions; shuffled domain-homogeneous batches |
| Class weights / resampling | None |
| Learning-rate scheduler | None; no learning-rate decay configured |
| Gradient clipping | None configured |
| Label smoothing / data augmentation | None configured |
| Training device | NVIDIA GeForce RTX 4060 |
| Precision | CUDA float16 autocast with GradScaler; float32 master parameters |
| CPU threads | 4 |
| cuDNN deterministic / benchmark | True / False |

All Adam defaults, including foreach/fused/capturable/differentiable flags, are
recorded in [training_parameters.json](artifacts/documentation/training_parameters.json).
Null foreach/fused values mean that no explicit choice was made; library behavior
follows the recorded PyTorch build. They do not prove a particular kernel was used.

Physical microbatches halve if the implemented GPU out-of-memory fallback is
triggered; the effective batch remains 8192 through weighted gradient accumulation.
Final partial batches contain the remaining rows. No rows are dropped to force
a full batch. A fallback is implemented, not claimed as an observed final event.

## Dropout and model dimensions

- Encoder: learned per-feature affine map to width 8, tanh, and fixed mean pooling
  into eight groups. No encoder dropout. Every candidate has four encoders and
  one shared classifier; the resulting representation is [batch, 8, 8].
- MLP/DNN: flatten to 64; hidden layers 128, 64, 32, ReLU and dropout 0.3 after
  each hidden layer; one output logit.
- CNN1D: channels 8->128->64->32, kernels 3, padding 1. Dropout 0.3 after the
  first two convolution/ReLU stages; adaptive average pooling. Head 32->64->32->1,
  with ReLU/dropout 0.3 after head widths 64 and 32.
- LSTM and GRU: input width 8, hidden width 128, two recurrent layers, dropout 0.3
  between recurrent layers. Head 128->64->32->1 with ReLU/dropout after 64 and 32.
- BiLSTM: same recurrent depth/width per direction; concatenate both final hidden
  states into 256 values. Head 256->64->32->1 with the same head dropout.

Dropout is disabled by model.eval() for validation/test inference. Encoder weights
and biases initialize from normal noise scaled by 0.02; other modules use their
installed PyTorch initialization defaults under seed 42. Shared settings do not
mean identical parameter counts or identical numbers of dropout layers.
Recurrent inputs are feature-group sequences, not chronological traffic.

## Exact validation and stopping rule

After every completed epoch, predict each of the four validation sets separately
using threshold 0.5. Compute binary macro-F1 and attack average precision (AP)
for each dataset. Let F_d and A_d be these values for the four datasets:

```
F = (F_DataSense + F_Edge + F_CIC + F_X) / 4
A = (A_DataSense + A_Edge + A_CIC + A_X) / 4
```

Each dataset has equal weight in validation selection, even though training uses
natural row proportions. Compare `(F, A)` lexicographically to the best recorded
pair. A higher F improves the checkpoint; exactly equal F and higher A also
improves it. No separate min_delta is configured. Improvement saves best.pt and
resets the non-improvement counter to zero. Otherwise increment the counter.
Stop after three consecutive non-improvements, or at the maximum of 20 epochs.
This is not early stopping on validation loss or on test accuracy.

After every candidate finishes, rank models by highest F, then highest A, then
fewest trainable parameters. Restore the best checkpoint, not automatically the
last epoch. Freeze the baseline choice before test evaluation. Test thresholds
are not optimized. The final baseline is provisional within this fixed-seed,
dataset-specific-encoder experiment.

## Actual completed runs

| Model | Parameters | Epoch limit | Actual epochs | Best checkpoint epoch | Best mean validation macro-F1 | Associated mean AP |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 21137 | 20 | 20 | 19 | 0.94336292 | 0.97644702 |
| CNN1D | 40689 | 20 | 20 | 20 | 0.94222059 | 0.9757397 |
| LSTM | 215569 | 20 | 14 | 11 | 0.9336052 | 0.9730162 |
| GRU | 164881 | 20 | 11 | 8 | 0.93278919 | 0.97244409 |
| BiLSTM | 557585 | 20 | 9 | 6 | 0.92643375 | 0.96975216 |

Selected baseline: **MLP**, with mean validation
macro-F1 **0.94336292**. The MLP ran 20
epochs but its selected checkpoint is epoch 19.

## Input preprocessing and evaluation contract

Per-domain numeric parsing -> signed log1p -> fixed zero imputation -> per-domain
standard scaling fitted on training rows only. Feature counts are 64, 25, 39 and
25 for DataSense, Edge-IIoTset, CICIoT2023 and X-IIoTID, respectively. Labels,
timestamps and raw identities are not predictors; derived identity counts may
be present in the saved native feature lists. Domain identity routes the encoder.
This is not a common physical flow-feature representation.

All four test sets are evaluated separately for every candidate (20 evaluations).
Confusion counts, both-class precision/recall/F1/support, macro/weighted F1,
accuracy, balanced accuracy, MCC, specificity, false-positive/negative rates,
ROC-AUC, AP, log loss and Brier score are saved. AP is not trapezoidal PR area.
Inference batches use at most 8192 rows in the trainer. Model-only timing is a
separate experiment at batches 1/512 with five warmups and 20 repetitions.

## Sources and full procedure

- [Saved run configuration](artifacts/adapted_experiment/configuration.json)
- [Recorded optimizer defaults and machine environment](artifacts/documentation/environment.json)
- [Machine-readable parameter record](artifacts/documentation/training_parameters.json)
- [Training implementation](train_adapted.py) and [architectures](adapted_models.py)
- [Full process documentation](FULL_PROCESS_DOCUMENTATION.md)
- [Complete results](EXPERIMENT_REPORT.md)

The batch size, encoders and recurrent/CNN extensions differ from the paper.
No 2-D/3-D spatial variants, adversarial defenses or end-to-end edge timing are
claimed by these training results. See the full procedure for all limitations.
