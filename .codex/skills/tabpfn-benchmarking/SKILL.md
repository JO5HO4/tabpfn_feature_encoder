---
name: tabpfn-benchmarking
description: Use when adding, reviewing, or interpreting benchmark/generalization results for tabpfn-feature-encoder, especially source 12-class encoder training, CP even/odd frozen-encoder transfer, and open-data transfer metrics.
---

# TabPFN Benchmarking

## Current Comparison

Every default run trains the encoder on the 12-class source task without TabPFN,
then freezes it for downstream TabPFN tests.

Source metrics:

- `source_val_*`
- `source_test_*`

Generalization metrics:

1. `baseline_tabpfn`: TabPFN on downstream flat features.
2. `frozen_encoder_tabpfn`: frozen source encoder output followed by TabPFN.
3. `delta`: frozen encoder minus baseline.

## Split And Context Rules

- Do not create a new split inside benchmarking.
- Use `DatasetBundle.y_train/y_val/y_test` and matching flat/graph inputs.
- Baseline TabPFN and frozen-encoder TabPFN must use the same train-context indices.
- The default context size is `batch_size * support_query_ratio`.
- Test rows are the query set; process them in chunks sized like the training
  query side of a batch.

## Output Contract

Benchmark outputs should include:

- `metrics.json`
- `cp_generalization/cp_even_odd_generalization_metrics.json`
- `open_data_generalization_metrics.json` from default training or the transfer command

Terminal output should report, at minimum:

```text
source_12_class test: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
```

Saved JSON may keep full precision; terminal metrics should stay compact.
