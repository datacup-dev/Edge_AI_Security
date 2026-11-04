# Separate attacks on all mixed-training records

Status: **running**. 420,412,932/525,461,190 records saved; 10/15 attack datasets complete.

Every method processes the same 35,030,746 original mixed-training records, including 2,290,435 benign and 32,740,311 attack records. No validation/test records or final subsampling are included. Each method has its own folder; candidates retain their original label and target the opposite class.

[Full process and settings](MIXED_TRAINING_ATTACKS.md). [Live per-class metrics](artifacts/mixed_training_attacks/progress.csv). [Status](artifacts/mixed_training_attacks/status.json).

Saved records include unsuccessful, unchanged and initially misclassified cases. Success is a new opposite-class GB prediction on a changed record GB originally classified correctly. Partial rates describe different prefixes and cannot rank final attack strength. These are training-set vulnerability results, not held-out robustness.

![Live coverage and per-class ASR](artifacts/mixed_training_attacks/progress.png)

| Attack | Source → target | Saved / requested | New successes / eligible | Seed failures | Query cap exhausted |
|---|---|---:|---:|---:|---:|
| FGSM | 0 → 1 | 2,290,435 / 2,290,435 | 448,659 / 2,177,104 | 0 | 0 |
| FGSM | 1 → 0 | 32,740,311 / 32,740,311 | 166,577 / 32,596,618 | 0 | 0 |
| CW | 0 → 1 | 2,290,435 / 2,290,435 | 197,209 / 2,177,104 | 0 | 0 |
| CW | 1 → 0 | 32,740,311 / 32,740,311 | 90,981 / 32,596,618 | 0 | 0 |
| PGD | 0 → 1 | 2,290,435 / 2,290,435 | 512,533 / 2,177,104 | 0 | 0 |
| PGD | 1 → 0 | 32,740,311 / 32,740,311 | 184,770 / 32,596,618 | 0 | 0 |
| JSMA | 0 → 1 | 2,290,435 / 2,290,435 | 41,744 / 2,177,104 | 0 | 0 |
| JSMA | 1 → 0 | 32,740,311 / 32,740,311 | 44,400 / 32,596,618 | 0 | 0 |
| DeepFool | 0 → 1 | 2,290,435 / 2,290,435 | 408,314 / 2,177,104 | 0 | 0 |
| DeepFool | 1 → 0 | 32,740,311 / 32,740,311 | 165,494 / 32,596,618 | 0 | 0 |
| BIM | 0 → 1 | 2,290,435 / 2,290,435 | 419,069 / 2,177,104 | 0 | 0 |
| BIM | 1 → 0 | 32,740,311 / 32,740,311 | 179,838 / 32,596,618 | 0 | 0 |
| MIM | 0 → 1 | 2,290,435 / 2,290,435 | 452,083 / 2,177,104 | 0 | 0 |
| MIM | 1 → 0 | 32,740,311 / 32,740,311 | 208,042 / 32,596,618 | 0 | 0 |
| Transfer | 0 → 1 | 2,290,435 / 2,290,435 | 545,952 / 2,177,104 | 0 | 0 |
| Transfer | 1 → 0 | 32,740,311 / 32,740,311 | 186,432 / 32,596,618 | 0 | 0 |
| NES | 0 → 1 | 1,156,061 / 2,290,435 | 245,650 / 1,098,888 | 0 | 0 |
| NES | 1 → 0 | 16,535,587 / 32,740,311 | 104,416 / 16,462,993 | 0 | 0 |
| SPSA | 0 → 1 | 1,156,026 / 2,290,435 | 239,659 / 1,098,854 | 0 | 0 |
| SPSA | 1 → 0 | 16,535,110 / 32,740,311 | 98,735 / 16,462,517 | 0 | 0 |
| Square | 0 → 1 | 1,156,026 / 2,290,435 | 424,081 / 1,098,854 | 0 | 0 |
| Square | 1 → 0 | 16,535,110 / 32,740,311 | 211,909 / 16,462,517 | 0 | 0 |
| Boundary | 0 → 1 | 2,290,435 / 2,290,435 | 816,365 / 2,177,104 | 1,360,739 | 0 |
| Boundary | 1 → 0 | 32,740,311 / 32,740,311 | 359,857 / 32,596,618 | 32,236,761 | 0 |
| HopSkipJump | 0 → 1 | 838,952 / 2,290,435 | 299,241 / 797,422 | 498,181 | 89,383 |
| HopSkipJump | 1 → 0 | 12,010,264 / 32,740,311 | 131,684 / 11,957,400 | 11,825,716 | 4,564 |
| ZOO | 0 → 1 | 272,916 / 2,290,435 | 29,529 / 259,513 | 0 | 0 |
| ZOO | 1 → 0 | 3,909,420 / 32,740,311 | 8,275 / 3,892,102 | 0 | 0 |
| GAN | 0 → 1 | 2,290,435 / 2,290,435 | 204,135 / 2,177,104 | 0 | 0 |
| GAN | 1 → 0 | 32,740,311 / 32,740,311 | 232,025 / 32,596,618 | 0 | 0 |
