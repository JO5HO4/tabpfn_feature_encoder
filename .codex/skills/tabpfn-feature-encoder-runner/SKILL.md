---
name: tabpfn-feature-encoder-runner
description: Use when running, configuring, validating, or documenting the tabpfn-feature-encoder training repo, including conda setup, runner scripts, output artifacts, tests, and Git hygiene.
---

# TabPFN Feature Encoder Runner

## Repo Basics

- Repo root: `tabpfn-feature-encoder`.
- Main 12-class source residual config: `configs/cp_encoder.yaml`.
- Particle GNN config: `configs/cp_gnn.yaml`.
- Particle transformer config: `configs/cp_transformer.yaml`.
- Launcher: `bash scripts/run_cp_encoder.sh`.
- Package CLI: `tabpfn-encoder-train train --config configs/cp_encoder.yaml`.
- Output dir is configured by `output_dir`.

## Environment

Use the existing conda env:

```bash
conda activate tabpfn
python -m pip install -e ".[train,atlas,plots]"
```

The runner falls back to `conda run --no-capture-output -n tabpfn` if the console script is not on `PATH`.

## Runner Behavior

`scripts/run_cp_encoder.sh`:

- Sets `TABPFN_MODEL_CACHE_DIR` to `$SCRATCH/tabpfn_model_cache` unless already set.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless already set.
- Reuses an existing checkpoint from `~/.cache/tabpfn` when available.
- Accepts an optional config path: `bash scripts/run_cp_encoder.sh configs/other.yaml`.
- Runs the GNN with: `bash scripts/run_cp_encoder.sh configs/cp_gnn.yaml`.
- Runs the transformer with: `bash scripts/run_cp_encoder.sh configs/cp_transformer.yaml`.

## Validation Commands

Prefer these before finalizing code changes:

```bash
conda run -n tabpfn pytest -q
conda run -n tabpfn python -m compileall src tests
```

Clean generated Python/cache files before committing:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find src -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
```

## Artifacts

Training saves:

- `metrics.json`
- `training_summary.json`
- `epoch_metrics.csv`
- `encoder_classifier.pkl`
- `run_metadata.json`
- `cp_generalization/cp_even_odd_generalization_metrics.json`
- `cp_generalization/cp_even_odd_generalization_baseline_proba.npy`
- `cp_generalization/cp_even_odd_generalization_frozen_encoder_proba.npy`
- `open_data_generalization_metrics.json` in `transfer.output_dir`
- `open_data_generalization_baseline_proba.npy` in `transfer.output_dir`
- `open_data_generalization_frozen_encoder_proba.npy` in `transfer.output_dir`

Terminal metrics print to three decimals; CSV/JSON keep full precision.
