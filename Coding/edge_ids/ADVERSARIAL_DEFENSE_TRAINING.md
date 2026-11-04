# Separate Gradient Boosting defenses

`train_adversarial_gb.py` retrains 15 independent binary XGBoost classifiers,
one per attack's clean-plus-adversarial training snapshot. Each model starts
from scratch with the baseline algorithm and tree settings. Original baseline
artifacts remain preserved. This step does not evaluate or select defenses.

Inputs are `artifacts/augmented_training/<attack>/manifest.json`, copied to each
defense directory and checked by SHA-256 on every iterator reset. Snapshots do
not silently grow as attack generation continues. All clean training records
and all committed attack candidates in that snapshot are used with original
labels, including unsuccessful and unchanged candidates. Partial snapshots
produce explicitly provisional models. GAN has a complete training snapshot;
the other attack snapshots were partial when this run was started.

Settings: binary logistic objective, histogram trees, 300 fixed boosting rounds,
learning rate 0.1, depth 8, max_bin 128, subsample 1, column subsample 0.8, seed
42. Neural optimizer and dropout are not applicable. There is no early stopping
or validation/test selection. Quantile cuts use augmented training data only.
Validation and test combinations are not used for fitting.

Models run sequentially with two CPU threads and disk-backed
`ExtMemQuantileDMatrix` to reduce contention with attack generation. CPU replaces
the original baseline's GPU execution; tree hyperparameters remain the same,
although numerical results need not be bitwise identical. Batches contain up
to 16,384 rows. The 157-column native namespace retains explicit observed zeros,
missing inactive domain blocks, and the active domain marker.

Artifacts: `artifacts/adversarial_defenses/<attack>/configuration.json`, frozen
`training_manifest.json`, `progress.json`, `history.json`, `partial.ubj` every
25 rounds, final `model.ubj`, and `complete.json`. A completion record means
training finished, not that robustness was demonstrated. Per-attack logs and
the scheduler's `status.json` are in `artifacts/adversarial_defenses`.

Restart with `python train_adversarial_gb.py`: completed defenses are skipped;
interrupted defenses rebuild their training matrix and resume saved trees.
A changed input manifest is rejected rather than mixed with an older checkpoint.
Inspect logs for failed jobs; the scheduler proceeds to the other attacks.
