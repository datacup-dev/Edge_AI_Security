# Extended mixed-domain model comparison

All 11 new candidates completed; baseline selection uses validation only. Artifact status: verified_complete.

All models use the original four native train/validation/test memberships. The real training population has 35,030,746 rows; natural domain proportions are preserved. This extends the authorized dataset-specific-encoder study, not the paper's common-flow representation.

User choices: exact KNN and a degree-2 polynomial kernel SVM. KNN has no approximate-search setting; the kernel map has no Nystroem/random-feature approximation. The SVM uses finite stochastic primal optimization, not a global-optimum certificate.

[Protocol and planned settings](EXTENDED_EXPERIMENT_PLAN.md). [Full expanded methods and parameters](EXTENDED_EXPERIMENT_METHOD.md). [All recorded test metrics](artifacts/extended_report/all_test_metrics.csv). [Validation comparison](artifacts/extended_report/validation_comparison.csv).

## Validation ranking

| Model | Mean validation macro-F1 | Mean validation AP | Checkpoint/reference bytes |
|---|---:|---:|---:|
| GB | 0.962042 | 0.985641 | 2,168,929 |
| KNN | 0.950304 | 0.974192 | 10,736,476,216 |
| RF | 0.946305 | 0.979078 | 5,280,633 |
| MLP | 0.943363 | 0.976447 | 95,769 |
| Transformer | 0.934062 | 0.972859 | 312,079 |
| Autoencoder | 0.929756 | 0.972271 | 99,385 |
| GNN | 0.928887 | 0.969497 | 83,497 |
| SVM | 0.914836 | 0.948192 | 104,735 |
| GAN | 0.913244 | 0.966660 | 346,197 |
| DT | 0.907797 | 0.958132 | 336,873 |
| LR | 0.894225 | 0.945983 | 3,505 |
| NB | 0.457749 | 0.608977 | 9,860 |

Best new candidate: **GB**. Overall baseline compared with the prior MLP: **GB**.

The old MLP checkpoint/results are preserved. Overall selection compares validation pairs; new test results do not select the winner. These test datasets were already inspected in the prior stage, so this is a follow-on benchmark on reused holdouts.


## Separate held-out test results

| Model | Dataset | Macro-F1 | Accuracy | Attack recall | FPR | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| GB | DataSense | 0.939528 | 0.943602 | 0.859650 | 0.000926 | 0.958807 | 0.942822 |
| GB | Edge-IIoTset | 0.995581 | 0.996491 | 0.997471 | 0.003875 | 0.999579 | 0.997813 |
| GB | CICIoT2023 | 0.921402 | 0.992677 | 0.995855 | 0.139526 | 0.998136 | 0.999955 |
| GB | X-IIoTID | 0.990183 | 0.990189 | 0.991705 | 0.011248 | 0.999487 | 0.999474 |
| KNN | DataSense | 0.912842 | 0.919129 | 0.817538 | 0.013743 | 0.930680 | 0.912358 |
| KNN | Edge-IIoTset | 0.994491 | 0.995620 | 0.998100 | 0.005306 | 0.998155 | 0.990248 |
| KNN | CICIoT2023 | 0.905967 | 0.991096 | 0.994639 | 0.156269 | 0.984159 | 0.999276 |
| KNN | X-IIoTID | 0.986931 | 0.986940 | 0.987015 | 0.013130 | 0.995673 | 0.993485 |
| RF | DataSense | 0.924422 | 0.929957 | 0.828601 | 0.003070 | 0.935719 | 0.933657 |
| RF | Edge-IIoTset | 0.985962 | 0.988951 | 0.969801 | 0.003895 | 0.997443 | 0.987063 |
| RF | CICIoT2023 | 0.903840 | 0.991294 | 0.995857 | 0.198496 | 0.997316 | 0.999936 |
| RF | X-IIoTID | 0.970460 | 0.970494 | 0.962345 | 0.021783 | 0.993463 | 0.993660 |
| MLP | DataSense | 0.889688 | 0.898823 | 0.767903 | 0.014669 | 0.909903 | 0.908775 |
| MLP | Edge-IIoTset | 0.994897 | 0.995951 | 0.995571 | 0.003908 | 0.999551 | 0.997718 |
| MLP | CICIoT2023 | 0.913589 | 0.991933 | 0.995425 | 0.153325 | 0.997779 | 0.999947 |
| MLP | X-IIoTID | 0.973380 | 0.973409 | 0.965867 | 0.019442 | 0.996349 | 0.996477 |
| Transformer | DataSense | 0.880036 | 0.890050 | 0.755439 | 0.021004 | 0.895258 | 0.903377 |
| Transformer | Edge-IIoTset | 0.994845 | 0.995911 | 0.994809 | 0.003677 | 0.999553 | 0.997731 |
| Transformer | CICIoT2023 | 0.909026 | 0.991301 | 0.994492 | 0.141426 | 0.997512 | 0.999941 |
| Transformer | X-IIoTID | 0.952964 | 0.953032 | 0.940347 | 0.034945 | 0.987083 | 0.988491 |
| Autoencoder | DataSense | 0.883495 | 0.893043 | 0.762519 | 0.020712 | 0.902824 | 0.903958 |
| Autoencoder | Edge-IIoTset | 0.981210 | 0.985277 | 0.956028 | 0.003796 | 0.998249 | 0.994974 |
| Autoencoder | CICIoT2023 | 0.902881 | 0.990961 | 0.995005 | 0.177243 | 0.997302 | 0.999935 |
| Autoencoder | X-IIoTID | 0.951070 | 0.951107 | 0.949059 | 0.046952 | 0.987986 | 0.986795 |
| GNN | DataSense | 0.879651 | 0.890901 | 0.735379 | 0.006335 | 0.898525 | 0.902732 |
| GNN | Edge-IIoTset | 0.980405 | 0.984661 | 0.953454 | 0.003681 | 0.996124 | 0.991454 |
| GNN | CICIoT2023 | 0.907861 | 0.991204 | 0.994484 | 0.145251 | 0.997462 | 0.999939 |
| GNN | X-IIoTID | 0.947262 | 0.947363 | 0.928446 | 0.034708 | 0.979937 | 0.981762 |
| SVM | DataSense | 0.883186 | 0.893923 | 0.742385 | 0.005945 | 0.894209 | 0.892879 |
| SVM | Edge-IIoTset | 0.971433 | 0.977812 | 0.928889 | 0.003912 | 0.961844 | 0.951959 |
| SVM | CICIoT2023 | 0.893350 | 0.990095 | 0.994615 | 0.197895 | 0.976586 | 0.999064 |
| SVM | X-IIoTID | 0.912211 | 0.912212 | 0.939612 | 0.113758 | 0.963892 | 0.949407 |
| GAN | DataSense | 0.870575 | 0.883535 | 0.712663 | 0.003558 | 0.877602 | 0.889471 |
| GAN | Edge-IIoTset | 0.981447 | 0.985451 | 0.957640 | 0.004159 | 0.998098 | 0.994243 |
| GAN | CICIoT2023 | 0.866229 | 0.985806 | 0.988849 | 0.140765 | 0.994284 | 0.999861 |
| GAN | X-IIoTID | 0.934590 | 0.934839 | 0.897201 | 0.029488 | 0.980732 | 0.981575 |
| DT | DataSense | 0.927792 | 0.932745 | 0.843056 | 0.007992 | 0.947728 | 0.934550 |
| DT | Edge-IIoTset | 0.988881 | 0.991225 | 0.978185 | 0.003903 | 0.998487 | 0.993699 |
| DT | CICIoT2023 | 0.887327 | 0.989518 | 0.994276 | 0.208397 | 0.996599 | 0.999914 |
| DT | X-IIoTID | 0.827788 | 0.828079 | 0.808689 | 0.153544 | 0.921710 | 0.901858 |
| LR | DataSense | 0.881154 | 0.889932 | 0.776827 | 0.035331 | 0.894692 | 0.903517 |
| LR | Edge-IIoTset | 0.971233 | 0.977656 | 0.928657 | 0.004040 | 0.972094 | 0.956610 |
| LR | CICIoT2023 | 0.867453 | 0.987771 | 0.993589 | 0.254218 | 0.995337 | 0.999888 |
| LR | X-IIoTID | 0.854039 | 0.854214 | 0.842071 | 0.134276 | 0.924817 | 0.921163 |
| NB | DataSense | 0.416982 | 0.617624 | 0.038941 | 0.000000 | 0.610564 | 0.627900 |
| NB | Edge-IIoTset | 0.495270 | 0.748183 | 0.074105 | 0.000000 | 0.536663 | 0.333900 |
| NB | CICIoT2023 | 0.583973 | 0.975112 | 0.995747 | 0.883190 | 0.988760 | 0.999730 |
| NB | X-IIoTID | 0.339237 | 0.513401 | 0.000000 | 0.000000 | 0.533830 | 0.476520 |

![Test comparison](artifacts/extended_report/test_macro_f1.png)

## Architecture meaning and limitations

- Autoencoder: supervised bottleneck classifier with a weighted native-feature reconstruction objective; not benign-only anomaly detection.
- GNN: message passing over eight feature-group nodes and a predefined relation graph; no inferred device or flow topology.
- Transformer: attention over feature groups, not a chronological packet sequence.
- GAN: conditional generator and benign/attack/fake discriminator. Binary classification conditions on the real classes; generated vectors are not adversarial network attacks.
- LR and polynomial SVM: full native-coordinate models, optimized with Adam and validation early stopping. Finite iterations do not certify optimum convergence.
- DT/RF: sparse zero-coded native namespace. GB: unavailable blocks are missing, but observed native zero values remain explicit. These are comparisons of complete pipelines, not a pure classifier-only ablation.
- NB: Gaussian independence model over full native blocks; weak performance is retained rather than omitted. Its implementation matched a dense sklearn reference in synthetic checks.
- SVM sigmoid scores are uncalibrated. ROC/AP use saved raw margins; log loss/Brier use the stated probability link. Numerical probability precision and clipping can differ by pipeline.
- KNN keeps the full reference population. Constant-coordinate removal has an exact query-distance offset; it is not a PCA or sample reduction.
- Random row splits do not isolate duplicate records, captures, devices or time. Native observation units differ. No unseen-domain transfer, adversarial robustness or end-to-end edge timing is established.

![Feature graph](artifacts/extended_report/gnn_feature_graph.png)

## All confusion and ROC/PR figures

| Model | Dataset | Confusion counts and normalized rates | ROC/PR |
|---|---|---|---|
| GB | DataSense | [Confusion](artifacts/expanded_classical/GB/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/GB/test/DataSense/roc_pr.png) |
| GB | Edge-IIoTset | [Confusion](artifacts/expanded_classical/GB/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/GB/test/Edge-IIoTset/roc_pr.png) |
| GB | CICIoT2023 | [Confusion](artifacts/expanded_classical/GB/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/GB/test/CICIoT2023/roc_pr.png) |
| GB | X-IIoTID | [Confusion](artifacts/expanded_classical/GB/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/GB/test/X-IIoTID/roc_pr.png) |
| KNN | DataSense | [Confusion](artifacts/expanded_classical/KNN/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/KNN/test/DataSense/roc_pr.png) |
| KNN | Edge-IIoTset | [Confusion](artifacts/expanded_classical/KNN/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/KNN/test/Edge-IIoTset/roc_pr.png) |
| KNN | CICIoT2023 | [Confusion](artifacts/expanded_classical/KNN/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/KNN/test/CICIoT2023/roc_pr.png) |
| KNN | X-IIoTID | [Confusion](artifacts/expanded_classical/KNN/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/KNN/test/X-IIoTID/roc_pr.png) |
| RF | DataSense | [Confusion](artifacts/expanded_classical/RF/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/RF/test/DataSense/roc_pr.png) |
| RF | Edge-IIoTset | [Confusion](artifacts/expanded_classical/RF/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/RF/test/Edge-IIoTset/roc_pr.png) |
| RF | CICIoT2023 | [Confusion](artifacts/expanded_classical/RF/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/RF/test/CICIoT2023/roc_pr.png) |
| RF | X-IIoTID | [Confusion](artifacts/expanded_classical/RF/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/RF/test/X-IIoTID/roc_pr.png) |
| MLP | DataSense | [Confusion](artifacts/adapted_experiment/MLP/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/adapted_experiment/MLP/test/DataSense/roc_pr.png) |
| MLP | Edge-IIoTset | [Confusion](artifacts/adapted_experiment/MLP/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/adapted_experiment/MLP/test/Edge-IIoTset/roc_pr.png) |
| MLP | CICIoT2023 | [Confusion](artifacts/adapted_experiment/MLP/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/adapted_experiment/MLP/test/CICIoT2023/roc_pr.png) |
| MLP | X-IIoTID | [Confusion](artifacts/adapted_experiment/MLP/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/adapted_experiment/MLP/test/X-IIoTID/roc_pr.png) |
| Transformer | DataSense | [Confusion](artifacts/expanded_neural/Transformer/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Transformer/test/DataSense/roc_pr.png) |
| Transformer | Edge-IIoTset | [Confusion](artifacts/expanded_neural/Transformer/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Transformer/test/Edge-IIoTset/roc_pr.png) |
| Transformer | CICIoT2023 | [Confusion](artifacts/expanded_neural/Transformer/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Transformer/test/CICIoT2023/roc_pr.png) |
| Transformer | X-IIoTID | [Confusion](artifacts/expanded_neural/Transformer/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Transformer/test/X-IIoTID/roc_pr.png) |
| Autoencoder | DataSense | [Confusion](artifacts/expanded_neural/Autoencoder/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Autoencoder/test/DataSense/roc_pr.png) |
| Autoencoder | Edge-IIoTset | [Confusion](artifacts/expanded_neural/Autoencoder/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Autoencoder/test/Edge-IIoTset/roc_pr.png) |
| Autoencoder | CICIoT2023 | [Confusion](artifacts/expanded_neural/Autoencoder/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Autoencoder/test/CICIoT2023/roc_pr.png) |
| Autoencoder | X-IIoTID | [Confusion](artifacts/expanded_neural/Autoencoder/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/Autoencoder/test/X-IIoTID/roc_pr.png) |
| GNN | DataSense | [Confusion](artifacts/expanded_neural/GNN/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GNN/test/DataSense/roc_pr.png) |
| GNN | Edge-IIoTset | [Confusion](artifacts/expanded_neural/GNN/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GNN/test/Edge-IIoTset/roc_pr.png) |
| GNN | CICIoT2023 | [Confusion](artifacts/expanded_neural/GNN/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GNN/test/CICIoT2023/roc_pr.png) |
| GNN | X-IIoTID | [Confusion](artifacts/expanded_neural/GNN/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GNN/test/X-IIoTID/roc_pr.png) |
| SVM | DataSense | [Confusion](artifacts/expanded_classical/SVM/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/SVM/test/DataSense/roc_pr.png) |
| SVM | Edge-IIoTset | [Confusion](artifacts/expanded_classical/SVM/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/SVM/test/Edge-IIoTset/roc_pr.png) |
| SVM | CICIoT2023 | [Confusion](artifacts/expanded_classical/SVM/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/SVM/test/CICIoT2023/roc_pr.png) |
| SVM | X-IIoTID | [Confusion](artifacts/expanded_classical/SVM/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/SVM/test/X-IIoTID/roc_pr.png) |
| GAN | DataSense | [Confusion](artifacts/expanded_neural/GAN/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GAN/test/DataSense/roc_pr.png) |
| GAN | Edge-IIoTset | [Confusion](artifacts/expanded_neural/GAN/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GAN/test/Edge-IIoTset/roc_pr.png) |
| GAN | CICIoT2023 | [Confusion](artifacts/expanded_neural/GAN/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GAN/test/CICIoT2023/roc_pr.png) |
| GAN | X-IIoTID | [Confusion](artifacts/expanded_neural/GAN/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_neural/GAN/test/X-IIoTID/roc_pr.png) |
| DT | DataSense | [Confusion](artifacts/expanded_classical/DT/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/DT/test/DataSense/roc_pr.png) |
| DT | Edge-IIoTset | [Confusion](artifacts/expanded_classical/DT/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/DT/test/Edge-IIoTset/roc_pr.png) |
| DT | CICIoT2023 | [Confusion](artifacts/expanded_classical/DT/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/DT/test/CICIoT2023/roc_pr.png) |
| DT | X-IIoTID | [Confusion](artifacts/expanded_classical/DT/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/DT/test/X-IIoTID/roc_pr.png) |
| LR | DataSense | [Confusion](artifacts/expanded_classical/LR/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/LR/test/DataSense/roc_pr.png) |
| LR | Edge-IIoTset | [Confusion](artifacts/expanded_classical/LR/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/LR/test/Edge-IIoTset/roc_pr.png) |
| LR | CICIoT2023 | [Confusion](artifacts/expanded_classical/LR/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/LR/test/CICIoT2023/roc_pr.png) |
| LR | X-IIoTID | [Confusion](artifacts/expanded_classical/LR/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/LR/test/X-IIoTID/roc_pr.png) |
| NB | DataSense | [Confusion](artifacts/expanded_classical/NB/test/DataSense/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/NB/test/DataSense/roc_pr.png) |
| NB | Edge-IIoTset | [Confusion](artifacts/expanded_classical/NB/test/Edge-IIoTset/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/NB/test/Edge-IIoTset/roc_pr.png) |
| NB | CICIoT2023 | [Confusion](artifacts/expanded_classical/NB/test/CICIoT2023/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/NB/test/CICIoT2023/roc_pr.png) |
| NB | X-IIoTID | [Confusion](artifacts/expanded_classical/NB/test/X-IIoTID/confusion_matrix.png) | [ROC/PR](artifacts/expanded_classical/NB/test/X-IIoTID/roc_pr.png) |

## Training histories and reproducibility

[Detailed settings and methods](EXTENDED_EXPERIMENT_METHOD.md). [Executed preview notebook](preview.ipynb).

- GB: [learning curves](artifacts/expanded_classical/GB/learning_curves.png), [recorded history](artifacts/expanded_classical/GB/history.json).
- MLP: [learning curves](artifacts/adapted_experiment/MLP/learning_curves.png), [recorded history](artifacts/adapted_experiment/MLP/history.json).
- Transformer: [learning curves](artifacts/expanded_neural/Transformer/learning_curves.png), [recorded history](artifacts/expanded_neural/Transformer/history.json).
- Autoencoder: [learning curves](artifacts/expanded_neural/Autoencoder/learning_curves.png), [recorded history](artifacts/expanded_neural/Autoencoder/history.json).
- GNN: [learning curves](artifacts/expanded_neural/GNN/learning_curves.png), [recorded history](artifacts/expanded_neural/GNN/history.json).
- SVM: [learning curves](artifacts/expanded_classical/SVM/learning_curves.png), [recorded history](artifacts/expanded_classical/SVM/history.json).
- GAN: [learning curves](artifacts/expanded_neural/GAN/learning_curves.png), [recorded history](artifacts/expanded_neural/GAN/history.json).
- LR: [learning curves](artifacts/expanded_classical/LR/learning_curves.png), [recorded history](artifacts/expanded_classical/LR/history.json).

[Full artifact audit](artifacts/extended_report/verification.json).

[Fresh loader consistency check](artifacts/extended_report/loader_verification.json).

[KNN CPU worker trial](artifacts/expanded_classical/KNN/worker_benchmark.json). [Index-layout decision](artifacts/knn_index_development/index_layout_decision.json).
