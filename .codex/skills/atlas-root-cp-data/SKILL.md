---
name: atlas-root-cp-data
description: Use when changing ATLAS CP ROOT data loading, branch configuration, feature construction, dataset splitting, padding, or processed dataset cache behavior in tabpfn-feature-encoder.
---

# ATLAS ROOT CP Data

## Current Data Contract

- Dataset code: `src/tabpfn_feature_encoder/data/atlas_root.py`.
- Preprocessing helpers: `src/tabpfn_feature_encoder/data/preprocessing.py`.
- Config section: `dataset` in `configs/source_residual_mlp.yaml`.
- All rows from every configured ROOT file are used. Do not reintroduce `events_per_class` sampling unless explicitly requested.
- Default source training uses 12 classes and intentionally excludes `ttH_NLO.root` and `ttH_CPodd.root` for CP even/odd generalization.
- Current split is stratified 50/25/25 train/validation/test.
- Test is held out and not used during training.

## Feature Construction

Configured scalar branches are read directly. Configured particle branches are padded to fixed width using each particle's `max`.

Current feature count is 72:

- scalars: `MET_met`, `MET_phi`
- jets: 4 branches x 10
- electrons: 4 branches x 3
- muons: 4 branches x 3
- photons: 3 branches x 2

If branches or max counts change, recalculate the feature count and update `encoder.output_dim` when identity-residual training should be preserved.

## Padding And Imputation

- `padding: zero` is the current default.
- `padding: nan` is supported.
- Median imputation is fit on train features only, then applied to validation/test.
- Standardization is fit on train features only.

## Cache

Processed datasets are cached under:

```text
<raw_dir>/.tabpfn_feature_cache/
```

The cache fingerprint includes input files, branches, split fractions, padding, tree name, and seed. Changing these should create a new cache automatically.

## Tests To Update

For data changes, update or run:

```bash
bash scripts/run_tests.sh tests/test_atlas_features.py tests/test_preprocessing.py tests/test_config.py
```
