# Adapted mixed-domain experiment report

Historical initial five-model phase. See the [Extended comparison](EXTENDED_EXPERIMENT_REPORT.md)
for subsequent candidates and the current overall baseline.

Full workflow and audit record: [Complete process documentation](FULL_PROCESS_DOCUMENTATION.md).

User-authorized dataset-specific encoders; not the paper’s shared NetFlow representation.

Current stage: **complete**. Completed model/test evaluations: 20 of 20.

![Training composition](artifacts/report/training_composition.png)

All training rows are used at natural proportions. Select checkpoints and the baseline using validation only. Recurrent variants process feature tokens, not time. See [methods](EXPERIMENT_METHOD.md).

Selected provisional baseline: **MLP**.

| Model | Mean validation macro-F1 | Mean validation AP | Parameters | Epochs |
|---|---:|---:|---:|---:|
| MLP | 0.943363 | 0.976447 | 21,137 | 20 |
| CNN1D | 0.942221 | 0.975740 | 40,689 | 20 |
| LSTM | 0.933605 | 0.973016 | 215,569 | 14 |
| GRU | 0.932789 | 0.972444 | 164,881 | 11 |
| BiLSTM | 0.926434 | 0.969752 | 557,585 | 9 |

The baseline was selected before test evaluation. Test scores do not determine this choice.

| Model | Test dataset | Accuracy | Macro-F1 | Attack recall | False-positive rate | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | CICIoT2023 | 0.991376 | 0.909815 | 0.994530 | 0.139805 | 0.997556 | 0.999942 |
| BiLSTM | DataSense | 0.888964 | 0.879602 | 0.766723 | 0.030263 | 0.894885 | 0.901068 |
| BiLSTM | Edge-IIoTset | 0.996356 | 0.995410 | 0.996996 | 0.003883 | 0.999409 | 0.997284 |
| BiLSTM | X-IIoTID | 0.918872 | 0.918866 | 0.935406 | 0.096800 | 0.976754 | 0.978329 |
| CNN1D | CICIoT2023 | 0.991853 | 0.915682 | 0.994504 | 0.118449 | 0.997817 | 0.999948 |
| CNN1D | DataSense | 0.895243 | 0.885022 | 0.750350 | 0.009016 | 0.906761 | 0.906736 |
| CNN1D | Edge-IIoTset | 0.996017 | 0.994978 | 0.995195 | 0.003677 | 0.999571 | 0.997786 |
| CNN1D | X-IIoTID | 0.973572 | 0.973543 | 0.966385 | 0.019616 | 0.996576 | 0.996588 |
| GRU | CICIoT2023 | 0.991604 | 0.912084 | 0.994679 | 0.136321 | 0.997661 | 0.999944 |
| GRU | DataSense | 0.889580 | 0.878047 | 0.731470 | 0.005945 | 0.898815 | 0.899957 |
| GRU | Edge-IIoTset | 0.995905 | 0.994837 | 0.994709 | 0.003648 | 0.999577 | 0.997806 |
| GRU | X-IIoTID | 0.947477 | 0.947346 | 0.922421 | 0.028776 | 0.988828 | 0.989024 |
| LSTM | CICIoT2023 | 0.991405 | 0.905871 | 0.995706 | 0.187490 | 0.997575 | 0.999942 |
| LSTM | DataSense | 0.886000 | 0.874218 | 0.728815 | 0.010136 | 0.896615 | 0.901288 |
| LSTM | Edge-IIoTset | 0.996077 | 0.995054 | 0.995372 | 0.003660 | 0.999460 | 0.997690 |
| LSTM | X-IIoTID | 0.959213 | 0.959144 | 0.943485 | 0.025881 | 0.991181 | 0.991807 |
| MLP | CICIoT2023 | 0.991933 | 0.913589 | 0.995425 | 0.153325 | 0.997779 | 0.999947 |
| MLP | DataSense | 0.898823 | 0.889688 | 0.767903 | 0.014669 | 0.909903 | 0.908775 |
| MLP | Edge-IIoTset | 0.995951 | 0.994897 | 0.995571 | 0.003908 | 0.999551 | 0.997718 |
| MLP | X-IIoTID | 0.973409 | 0.973380 | 0.965867 | 0.019442 | 0.996349 | 0.996477 |

![Test macro-F1 comparison](artifacts/report/test_comparison.png)

[All recorded test metrics](artifacts/report/test_metrics.csv). Per-model predictions, source rows and confusion/ROC/PR figures are in `artifacts/adapted_experiment/<model>/test/<dataset>/`.

For the selected baseline, DataSense attack miss rate is 23.21%; CICIoT2023 benign false-positive rate is 15.33%.

[Domain-prior reference metrics](artifacts/report/domain_prior_metrics.csv). This reference uses training label prevalence and no input traffic features.

[Workstation inference timings](artifacts/report/inference_latency.csv). Warm, preprocessed tensors only; excludes extraction, scaling, transfer, queueing and window collection. This is not an end-to-end edge latency measurement.

[Completed artifact audit](artifacts/adapted_experiment/verification.json): all 20 model/test evaluations matched the original held-out memberships and labels.

## Interpretation limits

Native observation units remain different. Scenario labels and random row splits do not establish independence of captures, sessions, devices or duplicates. Dataset ID routes encoders; compare against the saved domain-prior reference. These results do not establish unseen-domain transfer, temporal modelling, adversarial robustness or real-time deployment.
