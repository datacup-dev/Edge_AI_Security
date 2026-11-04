# Augmented training datasets

Each dataset concatenates all original mixed training records with one attack’s committed training candidates. Validation and test records are excluded. These are zero-copy datasets; the batch reader resolves original feature arrays and saved adversarial shards without duplicating them. Labels retain their original class. Unchanged/failed candidates are included and therefore can duplicate clean examples.

Snapshots are fixed at build time. Rerun `augmented_training.py` as generation advances. A partial dataset must not be presented as a completed full-pool experiment. No defended model has been trained by this step.

| Attack | Clean | Adversarial | Combined | Status |
|---|---:|---:|---:|---|
| FGSM | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| CW | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| PGD | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| JSMA | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| DeepFool | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| BIM | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| MIM | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| Transfer | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| Boundary | 35,030,746 | 35,030,746 | 70,061,492 | complete |
| HopSkipJump | 35,030,746 | 12,494,080 | 47,524,826 | partial |
| ZOO | 35,030,746 | 4,069,504 | 39,100,250 | partial |
| NES | 35,030,746 | 17,243,648 | 52,274,394 | partial |
| SPSA | 35,030,746 | 17,243,648 | 52,274,394 | partial |
| Square | 35,030,746 | 17,243,648 | 52,274,394 | partial |
| GAN | 35,030,746 | 35,030,746 | 70,061,492 | complete |

Use `batches(attack, batch_size=4096)` from `augmented_training.py`. It yields `(native_features, labels, dataset_ids, source_rows, is_adversarial)`; native features occupy the first F columns with NaN padding to 64. Dataset identity selects the native encoder. Iteration is clean then adversarial; training code must add reproducible shuffling, using only training records.

Latest snapshots are saved in `artifacts/augmented_training_latest`, separate from the frozen `artifacts/augmented_training` snapshots used by existing defenses. To read latest snapshots, set `augmented_training.OUT = augmented_training.PROJECT / "artifacts/augmented_training_latest"` before calling its batch reader. Existing defense inputs were not changed.
