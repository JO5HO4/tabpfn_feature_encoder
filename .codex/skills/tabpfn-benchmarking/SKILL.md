---
name: tabpfn-benchmarking
description: Use when adding, reviewing, or interpreting nominal benchmark results for tabpfn-feature-encoder, especially baseline TabPFN, Encoder+TabPFN, and encoder-only classifier test AUC/accuracy comparisons on the same train/val/test split.
---

# TabPFN Benchmarking

## Nominal Comparison

Every default CP training run should produce final test metrics for three models
using the existing dataset split:

1. `baseline_tabpfn`: TabPFN on standardized flat features.
2. `encoder_tabpfn`: the trained feature encoder followed by TabPFN.
3. `encoder_only_classifier`: a supervised classifier using the same encoder
   architecture and encoder hyperparameters, with a linear classification head.

The final comparison is always based on the held-out test split. Validation is
only for model selection and early stopping.

## Split And Context Rules

- Do not create a new split inside benchmarking.
- Use `DatasetBundle.y_train/y_val/y_test` and matching flat/graph inputs.
- Baseline TabPFN and Encoder+TabPFN must use the same train-context indices.
- The default context size is `batch_size * support_query_ratio`.
- Test rows are the query set; process them in chunks sized like the training
  query side of a batch.

## Output Contract

Benchmark outputs should include:

- `benchmark_metrics.json`
- `benchmark_baseline_tabpfn_proba.npy`
- `benchmark_encoder_tabpfn_proba.npy`
- `benchmark_encoder_only_proba.npy`

Terminal output should report, at minimum:

```text
baseline_tabpfn: test_accuracy=..., test_roc_auc=..., test_log_loss=...
encoder_tabpfn: test_accuracy=..., test_roc_auc=..., test_log_loss=...
encoder_only_classifier: test_accuracy=..., test_roc_auc=..., test_log_loss=...
```

Saved JSON may keep full precision; terminal metrics should stay compact.
