---
name: tabpfn-benchmarking
description: Use when adding, reviewing, or interpreting benchmark/generalization results for tabpfn-feature-encoder, especially source 12-class encoder training, CP even/odd frozen-encoder transfer, and open-data transfer metrics.
---

# TabPFN Benchmarking

## Current Comparison

Every default run trains the encoder on the 12-class source task through frozen
TabPFN support/query episodes, then freezes it for source-task and downstream
TabPFN tests. TabPFN model weights are never optimized.

The default source objective uses balanced binary ECOC for the 12-class task,
with `tabpfn_max_classes: 2`, `many_class_redundancy: 4`, and rotating episodic
validation using `validation_episodes: 8`.

Use `bash scripts/run_full_workflow.sh` to produce the nominal residual MLP, GNN,
and transformer comparison runs. On multi-GPU nodes this runs configs in
parallel, one config per visible GPU. Pass config paths to restrict the workflow.

Source metrics:

- `source_val_*`
- `source_test_*`
- `source_generalization_*`
- `epoch_metrics.csv` with train/validation metrics and gradient norms

Generalization metrics:

1. `baseline_tabpfn`: TabPFN on downstream flat features.
2. `frozen_encoder_tabpfn`: frozen source encoder output followed by TabPFN.
3. `delta`: frozen encoder minus baseline.

## Split And Context Rules

- Do not create a new split inside benchmarking.
- Use `DatasetBundle.y_train/y_val/y_test` and matching flat/graph inputs.
- Baseline TabPFN and frozen-encoder TabPFN must use the same validation-context indices.
- Downstream context is sampled from the validation split, never from test.
- The context scan starts at `transfer.context_min_per_class` events per class
  and ends at the full validation split unless `transfer.context_size` caps it.
- Repeat each context size `transfer.context_repeats` times and use full test
  set evaluation for every point.
- Test rows are the query set; process them in chunks of `transfer.query_chunk_size`.
- Use `bash scripts/plot_context_comparison.sh` after all three encoder runs to
  create baseline/MLP/GNN/transformer AUC and accuracy PDFs with error bars.

## Output Contract

Benchmark outputs should include:

- `metrics.json`
- `source_generalization/source_12_class_generalization_metrics.json`
- `source_generalization/source_12_class_generalization_context_scan_metrics.csv`
- `source_generalization/source_12_class_generalization_context_scan_roc_auc.png`
- `cp_generalization/cp_even_odd_generalization_metrics.json` from default training or `transfer-cp`
- `cp_generalization/cp_even_odd_generalization_context_scan_metrics.csv`
- `cp_generalization/cp_even_odd_generalization_context_scan_roc_auc.png`
- `open_data_generalization_metrics.json` from default training or the open-data transfer command
- `open_data_generalization_context_scan_metrics.csv`
- `open_data_generalization_context_scan_roc_auc.png`
- `context_scan_comparison/*_roc_auc_comparison.pdf`
- `context_scan_comparison/*_accuracy_comparison.pdf`

Terminal output should report, at minimum:

```text
encoder_tabpfn epoch ... train_loss=..., train_accuracy=..., train_roc_auc=..., grad_norm_mean=..., grad_norm_max=..., skipped_nonfinite_updates=..., val_loss=..., val_accuracy=..., val_roc_auc=...
source_12_class test: accuracy=..., log_loss=..., roc_auc=...
source_12_class_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
source_12_class_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
```

Saved JSON may keep full precision; terminal metrics should stay compact.
