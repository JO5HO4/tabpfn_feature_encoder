---
name: tabpfn-encoder-debugging
description: Use when debugging TabPFN encoder training quality, validation AUC, probability collapse, CUDA out-of-memory, checkpoint/model-cache issues, or downstream context/query behavior in tabpfn-feature-encoder.
---

# TabPFN Encoder Debugging

## Key Known Behavior

The source encoder is trained through frozen TabPFN support/query episodes.
TabPFN weights stay frozen; gradients flow through TabPFN's differentiable input
path into the encoder only.

For the 12-class source task, training uses binary ECOC by default:

```yaml
tabpfn_max_classes: 2
many_class_redundancy: 4
```

Raw standardized features still give a useful downstream TabPFN baseline, so the
default residual MLP starts as identity residual when `output_dim == input_dim`:

```text
encoder(x) = x + 0.1 * residual_mlp(x)
```

For the current config, `output_dim: 72` matches the feature count.

## Training And Validation Protocol

- Source training uses `batch_size: 2048` support/query episodes.
- Episodes use a 50/50 support/query split from original 12-class-balanced samples,
  then apply the current ECOC column labels.
- Validation uses `encoder.validation_episodes` rotating validation support/query
  episodes for early stopping, then decodes ECOC probabilities back to 12 classes.
- CP even/odd and open-data generalization use TabPFN after source training.
- Downstream context is sampled from the downstream validation split.
- Downstream context scans from `transfer.context_min_per_class` events/class
  to the full validation split unless `transfer.context_size` caps it.
- Each context size is repeated `transfer.context_repeats` times with different
  stratified validation subsets.
- Downstream query rows are processed in chunks of `transfer.query_chunk_size`,
  currently 1024.

## Collapse Signals

If downstream generalization reports `roc_auc=0.500` and near-constant
probabilities, TabPFN probabilities have collapsed to nearly constant 0.5.

First checks:

- Confirm `encoder.type`, `encoder.output_dim`, `tabpfn_max_classes`,
  `many_class_redundancy`, `learning_rate`, and `grad_clip_norm` match the
  intended config.
- Inspect epoch `grad_norm_mean` and `grad_norm_max`. Tiny values suggest dead
  or over-clipped encoder updates.
- Inspect `cp_generalization/*_proba.npy` or transfer probability outputs.

The trainer currently stabilizes updates with:

- identity residual initialization
- residual scale `0.1`
- balanced binary ECOC columns
- original-class-balanced support/query sampling
- episodic validation
- seeded TabPFN prompts
- gradient clipping at `max_norm=1.0`

## CUDA OOM

TabPFN memory scales strongly with support/query size.

If OOM occurs:

```bash
nvidia-smi
kill <PID>
```

Then reduce:

```yaml
batch_size: 2048
```

Avoid `8192` unless memory has been checked.

## Model Cache

The runner prefers existing checkpoints in `~/.cache/tabpfn`, especially v2.5. To force one:

```bash
export TABPFN_MODEL_PATH=/path/to/model.ckpt
bash scripts/run_source_encoder.sh
```

If TabPFN attempts an interactive license flow, the checkpoint path is not being used.
