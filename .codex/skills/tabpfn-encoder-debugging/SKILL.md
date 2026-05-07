---
name: tabpfn-encoder-debugging
description: Use when debugging TabPFN encoder training quality, validation AUC, probability collapse, CUDA out-of-memory, checkpoint/model-cache issues, or support/query episode behavior in tabpfn-feature-encoder.
---

# TabPFN Encoder Debugging

## Key Known Behavior

Raw standardized features give a useful TabPFN baseline. A random encoder can destroy that signal, so the current encoder starts as identity residual when `output_dim == input_dim`:

```text
encoder(x) = x + 0.1 * residual_mlp(x)
```

For the current config, `output_dim: 72` matches the feature count.

## Training And Validation Protocol

- Train batches are split by `support_query_ratio`.
- Current `batch_size: 2048` means `1024` support and `1024` query.
- Validation uses a fixed validation context and scores the remaining validation rows as query.
- `initial val` is computed before optimizer updates; it should be close to the raw TabPFN baseline.

## Collapse Signals

If validation reports `val_roc_auc=0.500` and `val_p1_std=0.000`, TabPFN probabilities have collapsed to nearly constant 0.5.

First checks:

- Compare `initial val` to `epoch 1 val`.
- Confirm `identity_residual=True` and `residual_scale=0.1`.
- Confirm `encoder.output_dim` equals actual feature count.
- Inspect `epoch_metrics.csv` for `val_p1_std`.

The trainer currently stabilizes updates with:

- identity residual initialization
- residual scale `0.1`
- support prompt detached in `_episode_step`
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
bash scripts/run_cp_encoder.sh
```

If TabPFN attempts an interactive license flow, the checkpoint path is not being used.
