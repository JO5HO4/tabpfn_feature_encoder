---
name: tabpfn-encoder-debugging
description: Use when debugging TabPFN encoder training quality, validation AUC, probability collapse, CUDA out-of-memory, checkpoint/model-cache issues, or downstream context/query behavior in tabpfn-feature-encoder.
---

# TabPFN Encoder Debugging

## Key Known Behavior

The source encoder is trained with a direct multiclass classifier head, not with
TabPFN. This bypasses TabPFN's default 10-class limit for the 12-class source
task. Raw standardized features still give a useful downstream TabPFN baseline,
so the default encoder starts as identity residual when `output_dim == input_dim`:

```text
encoder(x) = x + 0.1 * residual_mlp(x)
```

For the current config, `output_dim: 72` matches the feature count.

## Training And Validation Protocol

- Source training uses `batch_size: 2048` supervised multiclass batches.
- Validation uses the source validation split for early stopping.
- CP even/odd and open-data generalization use TabPFN after source training.
- Downstream context size is `batch_size * support_query_ratio`, currently 1024.

## Collapse Signals

If downstream generalization reports `roc_auc=0.500` and near-constant
probabilities, TabPFN probabilities have collapsed to nearly constant 0.5.

First checks:

- Confirm `encoder.type: residual_mlp`, `residual_scale=0.1`, and `encoder.output_dim` equals actual feature count.
- Inspect `cp_generalization/*_proba.npy` or transfer probability outputs.

The trainer currently stabilizes updates with:

- identity residual initialization
- residual scale `0.1`
- gradient clipping at `max_norm=0.1`

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
