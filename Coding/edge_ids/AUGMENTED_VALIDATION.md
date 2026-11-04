# Combined clean and adversarial validation datasets

Fifteen attacks times four datasets form 60 separate evaluation datasets. Each contains the complete original clean validation set followed by the committed validation candidates for that attack and dataset. Training and test records are excluded. Original class labels are retained, and failed or unchanged candidates remain included. The four native schemas remain separate.

These are fixed zero-copy snapshots, not duplicated feature files. Rerun `augmented_validation.py` to include newly generated shards. Partial generation remains explicitly marked. This step performs no training, model selection or evaluation.

Verification checks all clean cohort rows against the original split code 1 and labels, every committed shard range for contiguous coverage, and the first/last shard provenance. It is not an exhaustive new attack-quality audit. The batch reader checks provenance for every shard as it is consumed.

Use `from augmented_validation import batches`, then `batches(attack, dataset, batch_size=4096)`. It yields `(native_features, labels, dataset_ids, source_rows, is_adversarial)`. Features have 64 columns with native features first and NaN padding. `is_adversarial` identifies a generated candidate, not necessarily a successful attack. For valid evaluation, report clean and adversarial metrics separately using that flag as well as any combined metrics.

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
| Boundary | DataSense | 34,079 | 34,079 | 68,158 | complete |
| HopSkipJump | DataSense | 34,079 | 34,079 | 68,158 | complete |
| ZOO | DataSense | 34,079 | 34,079 | 68,158 | complete |
| NES | DataSense | 34,079 | 34,079 | 68,158 | complete |
| SPSA | DataSense | 34,079 | 34,079 | 68,158 | complete |
| Square | DataSense | 34,079 | 34,079 | 68,158 | complete |
| GAN | DataSense | 34,079 | 34,079 | 68,158 | complete |
| FGSM | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| CW | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| PGD | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| JSMA | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| DeepFool | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| BIM | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| MIM | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| Transfer | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| Boundary | Edge-IIoTset | 332,880 | 284,992 | 617,872 | partial |
| HopSkipJump | Edge-IIoTset | 332,880 | 284,096 | 616,976 | partial |
| ZOO | Edge-IIoTset | 332,880 | 284,096 | 616,976 | partial |
| NES | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| SPSA | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| Square | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| GAN | Edge-IIoTset | 332,880 | 332,880 | 665,760 | complete |
| FGSM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| CW | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| PGD | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| JSMA | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| DeepFool | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| BIM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| MIM | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| Transfer | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| Boundary | CICIoT2023 | 7,016,505 | 284,992 | 7,301,497 | partial |
| HopSkipJump | CICIoT2023 | 7,016,505 | 284,096 | 7,300,601 | partial |
| ZOO | CICIoT2023 | 7,016,505 | 284,160 | 7,300,665 | partial |
| NES | CICIoT2023 | 7,016,505 | 2,271,744 | 9,288,249 | partial |
| SPSA | CICIoT2023 | 7,016,505 | 2,271,744 | 9,288,249 | partial |
| Square | CICIoT2023 | 7,016,505 | 2,271,744 | 9,288,249 | partial |
| GAN | CICIoT2023 | 7,016,505 | 7,016,505 | 14,033,010 | complete |
| FGSM | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| CW | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| PGD | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| JSMA | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| DeepFool | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| BIM | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| MIM | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| Transfer | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| Boundary | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| HopSkipJump | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| ZOO | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| NES | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| SPSA | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| Square | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
| GAN | X-IIoTID | 123,125 | 123,125 | 246,250 | complete |
