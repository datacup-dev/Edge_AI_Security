# Combined clean and adversarial test datasets

Fifteen attacks times four datasets form 60 separate evaluation datasets. Each contains the complete original clean test set followed by the committed test candidates for that attack and dataset. Training and validation records are excluded. Original class labels are retained, and failed or unchanged candidates remain included. The four native schemas remain separate.

These are fixed zero-copy snapshots, not duplicated feature files. Rerun `augmented_test.py` to include newly generated shards. Partial generation remains explicitly marked. This step performs no training, model selection or evaluation.

Verification checks all clean cohort rows against the original split code 2 and labels, every committed shard range for contiguous coverage, and the first/last shard provenance. It is not an exhaustive new attack-quality audit. The batch reader checks provenance for every shard as it is consumed.

Use `from augmented_test import batches`, then `batches(attack, dataset, batch_size=4096)`. It yields `(native_features, labels, dataset_ids, source_rows, is_adversarial)`. Features have 64 columns with native features first and NaN padding. `is_adversarial` identifies a generated candidate, not necessarily a successful attack. For valid evaluation, report clean and adversarial metrics separately using that flag as well as any combined metrics.

| Attack | Dataset | Clean | Adversarial | Combined | Status |
|---|---|---:|---:|---:|---|
| FGSM | DataSense | 34,079 | 34,079 | 68,158 | complete |
| CW | DataSense | 34,079 | 34,079 | 68,158 | complete |
| PGD | DataSense | 34,079 | 34,079 | 68,158 | complete |
| JSMA | DataSense | 34,079 | 34,079 | 68,158 | complete |
| DeepFool | DataSense | 34,079 | 34,079 | 68,158 | complete |
| BIM | DataSense | 34,079 | 34,079 | 68,158 | complete |
| MIM | DataSense | 34,079 | 34,079 | 68,158 | complete |
| Transfer | DataSense | 34,079 | 34,079 | 68,158 | complete |
| Boundary | DataSense | 34,079 | 3,776 | 37,855 | partial |
| HopSkipJump | DataSense | 34,079 | 34,079 | 68,158 | complete |
| ZOO | DataSense | 34,079 | 34,079 | 68,158 | complete |
| NES | DataSense | 34,079 | 34,079 | 68,158 | complete |
| SPSA | DataSense | 34,079 | 34,079 | 68,158 | complete |
| Square | DataSense | 34,079 | 34,079 | 68,158 | complete |
| GAN | DataSense | 34,079 | 34,079 | 68,158 | complete |
| FGSM | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| CW | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| PGD | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| JSMA | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| DeepFool | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| BIM | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| MIM | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| Transfer | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| Boundary | Edge-IIoTset | 332,881 | 3,776 | 336,657 | partial |
| HopSkipJump | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| ZOO | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| NES | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| SPSA | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| Square | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| GAN | Edge-IIoTset | 332,881 | 332,881 | 665,762 | complete |
| FGSM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| CW | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| PGD | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| JSMA | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| DeepFool | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| BIM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| MIM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| Transfer | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| Boundary | CICIoT2023 | 7,016,505 | 3,712 | 7,020,217 | partial |
| HopSkipJump | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| ZOO | CICIoT2023 | 7,016,505 | 3,441,664 | 10,458,169 | partial |
| NES | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| SPSA | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| Square | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| GAN | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| FGSM | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| CW | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| PGD | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| JSMA | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| DeepFool | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| BIM | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| MIM | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| Transfer | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| Boundary | X-IIoTID | 123,126 | 3,712 | 126,838 | partial |
| HopSkipJump | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| ZOO | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| NES | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| SPSA | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| Square | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
| GAN | X-IIoTID | 123,126 | 123,126 | 246,252 | complete |
