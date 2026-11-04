# Ubuntu lab server profile

## GPU update: three RTX 3090s reported

The user subsequently identified GPU 0 and GPU 1 as RTX 3090s with 24,101 MiB
free each (about 23.5 GiB); GPU 2 has only 3,088 MiB free. This supersedes the
earlier pending-GPU assessment below. Start with one available GPU and one
defense at a time because host RAM remains shared. The launcher does not reserve
a GPU; coordinate its use and recheck free memory before starting. Two cards do
not combine into one 48 GiB device in this implementation. Multi-GPU/distributed
training and concurrent defense workers are not implemented by this launcher.

After transferring the latest bundle and required data into the server clone:

```bash
cd Coding/edge_ids
.venv/bin/python -m pip install -r requirements-server-gpu.txt
CUDA_VISIBLE_DEVICES=0 bash run_lab_gpu.sh
```

CuPy's CUDA 12 wheel with toolkit extras is specified; driver-reported CUDA 13
compatibility does not require using a CUDA 13 wheel. Validate the actual runtime
with the provided preflight. The original CPU launcher remains available.

The GPU launcher verifies the migration, then runs a 32,768-row, three-round
save/resume smoke test without overwriting defense models. It proceeds to the
full frozen snapshots using GPU `QuantileDMatrix`, attempting to retain the
quantized training representation in VRAM. Full-data fit in 24 GiB is not yet
verified. On OOM, stop that run and review memory before retrying; do not launch
duplicate workers. Optional `EDGE_IDS_MATRIX=external` uses GPU input batches
and host-memory cache, which must fit available host RAM. It is not automatically
selected given the reported 38 GiB available and full swap. The CPU disk-backed
launcher is the fallback if both GPU modes exceed the memory budget.

The GPU iterator supplies dense CuPy arrays and labels: inactive native blocks
and inactive domain markers are NaN, active observed zeros remain zero, and the
active domain marker is one. GPU CSR is unsupported by XGBoost's iterator.
The CPU iterator remains CSR with identical observed values and missingness. An asynchronous
CuPy allocator is configured. Saved CPU UBJ trees can be loaded for GPU
continuation, but bitwise reproduction across CPU/GPU is not promised. The
same fixed training manifests and 300-round limit remain in force.

GPU execution has not been tested on the remote server; no SSH access was
provided. Syntax and the unchanged CPU input path are checked locally. This
package does not claim a measured speedup or completed remote training.

References: [XGBoost external-memory requirements](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)
and [CuPy installation](https://docs.cupy.dev/en/stable/install.html).

Repository supplied by the user: https://github.com/datacup-dev/Edge_AI_Security

The supplied machine has Ubuntu 22.04.5 x86_64, 14 physical CPU cores and 28
logical CPUs, 125 GiB total RAM but only 38 GiB currently available, and full
2 GiB swap. Start one defense at a time with eight CPU threads and external
memory. This is an initial resource allocation, not a measured optimal setting.
Do not select in-memory training based on total RAM alone. Coordinate with other
lab workloads and adjust threads after observing CPU, disk and available memory.

`/mnt/Data` has 6.1 TB free, while the root NVMe partition has 331 GB free.
Keep project data on `/mnt/Data`, in a directory you can write. Place disposable
training caches on the NVMe only if its remaining capacity and lab policy permit.
`/mnt/Data` storage type and speed were not supplied. Check the cache path with
`df -h`; caches may consume tens of GB and are not included in transfer bundles.
The launcher uses `$HOME/edge_ids_cache` by default; change it if home is not
on the intended device. Check disk space before starting each large run.

The NVIDIA driver is 580.178.04 and nvidia-smi reports CUDA compatibility 13.0.
Neither value specifies the GPU model, VRAM capacity, free VRAM or installed
CUDA toolkit. The prepared launcher uses CPU; GPU acceleration is pending those
details and a compatible GPU iterator. A particular speedup is not guaranteed.

## Clone, restore and launch

These commands run on the server using your own GitHub authentication:

```bash
mkdir -p /mnt/Data/"$USER"
cd /mnt/Data/"$USER"
git clone https://github.com/datacup-dev/Edge_AI_Security.git
cd Edge_AI_Security
```

If the repo is already cloned, use that clone and preserve local changes. No
repository push has been performed by Codex. The exported bundle has the files
ready to commit, but no Git/GitHub CLI or authenticated GitHub writer was
available in the workstation session. For example, from an authenticated clone:

```bash
# Copy the exported edge_ids source files into Coding/edge_ids first.
git switch -c server-defense-resume
git add Coding/edge_ids/*.py Coding/edge_ids/*.md Coding/edge_ids/*.sh Coding/edge_ids/requirements-server.txt Coding/edge_ids/.gitignore
git diff --cached --stat
git commit -m "Add portable GB defense resume workflow"
git push -u origin server-defense-resume
```

Restore the bundle's `edge_ids` directory contents into `Coding/edge_ids`, then
copy all inventory-listed data files to paths relative to that same directory.
Do not put either archive into ordinary Git. The artifact `.gitignore` prevents
accidentally staging large data/model artifacts with the example commands.

```bash
cd Coding/edge_ids
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-server.txt
.venv/bin/python verify_server_migration.py
# Keep this command in a persistent lab terminal/session:
bash run_lab_server.sh
```

Ubuntu 22.04's default Python may not be Python 3.13. Use an existing lab-managed
Python 3.13 installation, or arrange it before these commands. The script does
not change system Python or install OS packages.

Monitor with `tail -f artifacts/adversarial_defenses/Boundary.log` initially;
the scheduler status names the current attack. Inspect per-attack progress JSON
for the latest completed round. No local jobs have been stopped by this setup.
Confirm the server is running correctly, then stop the workstation scheduler
and worker before continuing the same defense on just one machine.

See SERVER_MIGRATION.md for snapshot rules: completed models are skipped,
quantile preparation restarts, saved trees resume at their checkpoint round,
and newer full adversarial pools require separate from-scratch runs.
