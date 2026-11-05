# Adversarial-stage removal

The user requested removal of adversarial sample generation and all files created
for that stage, while preserving original datasets, clean classifier results and
the original Gradient Boosting baseline. Attack outputs, augmented datasets,
defended models, attack scripts, attack notebook cells and migration exports are
removed. The clean GAN classifier is preserved: it belongs to the original
classifier comparison and is different from the attack generator.

## Lab server

From the cloned repository root:

```bash
git pull --ff-only
cd Coding/edge_ids
python3 cleanup_adversarial_artifacts.py
python3 cleanup_adversarial_artifacts.py --apply
```

The first invocation is a dry run. `--apply` stops only matching adversarial
processes owned by the current user in this project, then deletes an explicit
project-contained allowlist. It removes downloaded data/checkpoint migration
archives and `.venv-attacks`, but preserves `.venv`, original datasets, clean
encoded inputs, split metadata, mixed training membership, original model
checkpoints, saved classifier predictions/metrics and `artifacts/expanded_classical/GB`.
The known `/tmp/edge_ids_restore` export is removed only when its migration
marker exists and the directory is owned by the current user.
Known per-attack cache subdirectories under `~/edge_ids_cache` are also removed;
other cache names and custom locations are preserved.

Tracked source deletions arrive through Git. Untracked server-generated samples
and defended models require running the cleanup helper; Git pull cannot remove
them. Keep the helper until server cleanup succeeds. Its JSON audit records
before/after hashes of protected classifier metadata and checkpoints. The
original datasets and feature arrays are not rewritten.

No server SSH access is available in this session. A workstation cleanup does
not imply the server files or processes have been removed. Send the helper's
completion output after running it there.

The dedicated adversarial release/tag and current-branch checkpoint ZIP are
removed from GitHub. Existing Git commit history is retained; this is not a
force-pushed history rewrite.
