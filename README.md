# TabPFN Feature Encoder

Minimal training repo for learning a small feature encoder in front of TabPFN.

The current goal is deliberately narrow: use all configured ROOT events, train a
small encoder on support/query episodes, and feed the encoded event features into
a frozen TabPFN classifier.

## Current Design

1. Read every row from each configured ROOT file.
2. Build flat tabular features for the MLP-style encoders, or variable-particle event graphs for the GNN encoder.
3. Split the full dataset into train/validation/test with a 50/25/25 stratified split.
4. Standardize features using train-set statistics only.
5. Train on batches from the train split. Each batch is split 50/50 into support and query.
6. Fit TabPFN on the encoded support features and compute loss on the encoded query features.
7. At the end of each epoch, evaluate with a fixed validation context and the remaining validation events as query.
8. Hold out the test split for later. It is not used during training right now.

## Encoder Choice

The default encoder is now a lightweight GNN that uses every configured particle
in each event. It builds particle nodes from the jagged ROOT branches, runs a
small message-passing network, appends fixed event-summary features, and passes
a 128D hybrid output to TabPFN:

```text
particles + scalars -> GNN + event summaries -> 128D event vector -> TabPFN
```

The GNN ignores particle `max` values and uses all particles present in each
event. The `max` entries in the config are only for the flat fallback features.

The code also supports flat encoders. Set `encoder.type` to `feature_mixer` for
a stable residual linear feature mixer:

```text
encoder(x) = x + residual_scale * linear(x)
```

Set `encoder.type` to `feature_gate` for the most conservative flat encoder:

```text
encoder(x) = x * (1 + residual_scale * tanh(gate))
```

or `residual_mlp` for a more expressive flat MLP:

```text
encoder(x) = x + residual_scale * residual_mlp(x)
```

Training clips encoder gradients, keeps TabPFN frozen, detaches support/context
embeddings before fitting the TabPFN prompt, restores the best validation encoder
if an epoch hurts validation AUC, and stops early after repeated non-improving
epochs.

## Environment

From the repo root:

```bash
conda env create -f setup/environment.yml
conda activate tabpfn
python -m pip install -e ".[train,atlas,plots]"
```

For an existing `tabpfn` env:

```bash
conda env update -n tabpfn -f setup/environment.yml
conda activate tabpfn
python -m pip install -e ".[train,atlas,plots]"
```

Quick check:

```bash
python - <<'PY'
import torch
import tabpfn
import uproot
import awkward_pandas

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("tabpfn", getattr(tabpfn, "__version__", "installed"))
print("uproot", uproot.__version__)
PY
```

## Configuration

The main config is [configs/cp_encoder.yaml](configs/cp_encoder.yaml).

```yaml
output_dir: /global/cfs/projectdirs/atlas/joshua/tabpfn/runs/cp_encoder_general_gnn
cache_dir: /pscratch/sd/j/joshuaho/tabpfn
seed: 42
device: cuda

encoder:
  type: gnn
  layers: 3
  hidden_dim: 128
  output_dim: 128
  epochs: 20
  learning_rate: 0.00005
  batch_size: 2048
  support_query_ratio: 0.5
  residual_scale: 0.1
  identity_weight: 0.0
  grad_clip_norm: 0.1
  early_stopping_patience: 8
  min_delta: 0.001

dataset:
  raw_dir: /global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K
  split: {train: 0.5, val: 0.25, test: 0.25}
  labels:
    - label: 0
      files: [ttH_NLO.root]
    - label: 1
      files: [ttH_CPodd.root]
  padding: zero
  scalars: [MET_met, MET_phi]
  particles:
    - name: jet
      max: 10
      branches: [jet_pt, jet_eta, jet_phi, jet_btag]
    - name: electron
      max: 3
      branches: [ele_pt, ele_eta, ele_phi, ele_charge]
    - name: muon
      max: 3
      branches: [mu_pt, mu_eta, mu_phi, mu_charge]
    - name: photon
      max: 2
      branches: [ph_pt, ph_eta, ph_phi]

transfer:
  raw_dir: /global/cfs/projectdirs/atlas/haichen/opendata/GamGam_data
  output_dir: /global/cfs/projectdirs/atlas/joshua/tabpfn/runs/gamgam_transfer
  tree_name: mini
  context_size: 1024
  query_chunk_size: 1024
  split: {train: 0.5, val: 0.25, test: 0.25}
  labels:
    - {label: 0, name: ttH, files: [mc_341081.ttH125_gamgam.GamGam.root]}
    - {label: 1, name: ggF, files: [mc_343981.ggH125_gamgam.GamGam.root]}
    - {label: 2, name: VBF, files: [mc_345041.VBFH125_gamgam.GamGam.root]}
    - {label: 3, name: WH, files: [mc_345318.WpH125J_Wincl_gamgam.GamGam.root]}
    - {label: 4, name: ZH, files: [mc_345319.ZH125J_Zincl_gamgam.GamGam.root]}
```

Every row from every configured file is used. With the current two 100K ROOT
files, this means:

```text
total: 200000
train: 100000
validation: 50000
test: 50000
```

With `batch_size: 2048` and `support_query_ratio: 0.5`, each training step uses:

```text
support: 1024
query: 1024
```

With `type: gnn`, `output_dim: 128` is the event embedding size sent to TabPFN.
The GNN embedding is intentionally hybrid: it concatenates a learned graph
representation with fixed graph summary features, including global features,
event particle count, per-type particle counts, pooled particle statistics, and
per-particle-type pooled statistics. This is meant to preserve production-mode
information when the encoder is frozen for GamGam transfer.
With `type: feature_mixer` or `type: feature_gate`, `output_dim` must match the
number of flat input features, currently 72.

To switch back to a flat MLP-style encoder, change only the encoder block:

```yaml
encoder:
  type: residual_mlp
  layers: 4
  hidden_dim: 64
  output_dim: 72
  epochs: 20
  learning_rate: 0.00005
  batch_size: 2048
  support_query_ratio: 0.5
  residual_scale: 0.1
  identity_weight: 10.0
  grad_clip_norm: 0.1
  early_stopping_patience: 5
  min_delta: 0.0
```

Validation uses one fixed 1024-event context from the validation split, then
scores every remaining validation event as query in 1024-event chunks. This uses
almost the full validation set for metrics while keeping the TabPFN context and
query chunks in the same size regime as training.

## Data Cache

The first run reads ROOT files and saves processed `DatasetBundle` caches under:

```text
/pscratch/sd/j/joshuaho/tabpfn/cp_encoder/
/pscratch/sd/j/joshuaho/tabpfn/gamgam_production_modes/
```

Later runs load that cache and skip ROOT reading. The cache fingerprint includes
the input files, branches, split fractions, padding mode, seed, and whether graph
features were built. If those change, a new cache file is created.

When a cache does not exist yet, ROOT-to-feature creation is parallelized across
the configured input files. The worker count is
`min(os.cpu_count() or 1, number_of_files)`, so the CP task uses two workers for
the two CP ROOT files and the GamGam transfer task uses up to five workers for
the five production-mode files. During graph construction, each worker prints a
progress line every 50K events processed, plus one final line when that file is
complete.

## Training

Recommended launcher:

```bash
bash scripts/run_cp_encoder.sh
```

The launcher:

- Uses the `tabpfn` conda env automatically if the command is not already on `PATH`.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Uses an existing TabPFN checkpoint from `~/.cache/tabpfn` when available.
- Otherwise lets TabPFN download into `$SCRATCH/tabpfn_model_cache` by default.

To force a specific TabPFN checkpoint:

```bash
export TABPFN_MODEL_PATH=/path/to/model.ckpt
bash scripts/run_cp_encoder.sh
```

To use a different config:

```bash
bash scripts/run_cp_encoder.sh configs/other.yaml
```

Equivalent direct CLI:

```bash
tabpfn-encoder-train train --config configs/cp_encoder.yaml
```

## Transfer Evaluation

The transfer workflow freezes the CP-trained GNN encoder and evaluates whether
its 128D event embedding helps a new 5-class Higgs production-mode task:

```text
ttH vs ggF vs VBF vs WH vs ZH
```

It compares three TabPFN evaluations on the same GamGam train/test split:

1. `frozen_gnn_tabpfn`: CP-trained GNN encoder is frozen, GamGam events are encoded, then TabPFN is fit on encoded context events.
2. `frozen_gnn_plus_flat_tabpfn`: the frozen GNN embedding is concatenated with standardized flat features.
3. `baseline_tabpfn`: TabPFN is fit directly on padded flat GamGam features.

Run it after training the CP encoder:

```bash
bash scripts/run_gamgam_transfer.sh
```

or directly:

```bash
tabpfn-encoder-train transfer --config configs/cp_encoder.yaml
```

To evaluate a specific saved encoder:

```bash
tabpfn-encoder-train transfer \
  --config configs/cp_encoder.yaml \
  --model /path/to/encoder_best_val_auc.pkl
```

Transfer outputs are written to `transfer.output_dir`:

```text
transfer_metrics.json
frozen_gnn_test_proba.npy
frozen_gnn_plus_flat_test_proba.npy
baseline_test_proba.npy
```

## Expected Logs

A run should look like:

```text
Using TabPFN model: ...
Loading cached dataset: ...
EncoderTabPFN settings: type=gnn, device=cuda, layers=3, hidden_dim=128, output_dim=128, batch_size=2048, support_query_ratio=0.5, identity_residual=False, residual_scale=1.0, identity_weight=0.0, grad_clip_norm=0.1, early_stopping_patience=8, summary_dim=85, learned_dim=43
initial val: context=1024, query=48976, val_log_loss=..., val_accuracy=..., val_roc_auc=..., val_p1_mean=..., val_p1_std=...
epoch 1/20: train_loss=..., train_accuracy=..., train_roc_auc=..., batches=49/49
epoch 1/20 val: context=1024, query=48976, val_log_loss=..., val_accuracy=..., val_roc_auc=..., val_p1_mean=..., val_p1_std=...
restored best encoder after epoch 1 (best_val_roc_auc=...)
early stopping: no validation AUC improvement for 8 epochs
```

Metrics are printed to three decimal places in the terminal. Saved CSV/JSON files
keep full precision.

Earlier CP-only GNN checkpoints reached validation AUC around `0.66`, but their
frozen embeddings underperformed flat TabPFN on GamGam transfer. The current
default GNN is therefore summary-preserving and writes to a new output directory;
retrain it before rerunning transfer.

## Outputs

Artifacts are written to `output_dir`:

```text
run_metadata.json
metrics.json
training_summary.json
epoch_metrics.csv
encoder_tabpfn.pkl
encoder_best_val_auc.pkl
best_checkpoint.json
```

`epoch_metrics.csv` is the easiest file to inspect during development. It contains
one row per epoch with train loss/accuracy/AUC and validation loss/accuracy/AUC.
The saved `encoder_best_val_auc.pkl` is the checkpoint to load for transfer: it
contains the epoch with the highest validation AUC. `encoder_tabpfn.pkl` is kept
as the default final model artifact and is also restored to the best validation
state. Both saved models keep the trained encoder and preprocessing state on CPU
so they can be reused without a GPU session. If `device: cuda` is set on a
machine without CUDA, the trainer automatically falls back to CPU.

## Memory Notes

TabPFN memory scales strongly with support/query size. If CUDA runs out of memory:

```bash
nvidia-smi
kill <PID>
```

Then lower `encoder.batch_size` in the config. Good values to try:

```yaml
batch_size: 2048
batch_size: 1024
```

Avoid very large batches like `8192` unless you have confirmed enough GPU memory.
For transfer, keep `transfer.context_size + transfer.query_chunk_size` near the
same total batch size. The default is now `1024 + 1024 = 2048`.

## Tests

```bash
conda activate tabpfn
pytest -q
```
