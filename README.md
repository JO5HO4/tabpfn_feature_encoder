# TabPFN Feature Encoder

Minimal training repo for learning a small feature encoder in front of TabPFN.

The current goal is deliberately narrow: pretrain a small encoder on a 12-class
ATLAS source task, then freeze that encoder and test whether it helps TabPFN on
held-out downstream tasks.

## Current Design

1. Read every row from each configured ROOT file.
2. Build flat tabular features for the MLP-style encoders, or variable-particle event graphs for the GNN/transformer encoders.
3. Split the full dataset into train/validation/test with a 50/25/25 stratified split.
4. Standardize features using train-set statistics only.
5. Train the encoder through frozen TabPFN support/query episodes.
6. Use ECOC subtasks when the source task has more classes than TabPFN accepts directly.
7. Restore the best validation-AUC encoder and report source test metrics.
8. Freeze the encoder and run TabPFN context scans on the source task, held-out CP even/odd, and open data.

## Framework Diagram

Source training optimizes only the encoder. TabPFN is frozen, but gradients flow
through its differentiable input path back into the encoder:

```text
                         source episode
                  sampled from 12-class ATLAS data

        support features X_s                      query features X_q
                |                                        |
                v                                        v
        trainable encoder E_theta                trainable encoder E_theta
                |                                        |
                v                                        v
        encoded support Z_s                      encoded query Z_q
                |                                        |
                |                                        v
        support labels y_s -------------> frozen TabPFN prompt
                                                         |
                                                         v
                                                query logits / scores
                                                         |
                                                         v
                                           loss(logits, query labels y_q)
                                                         |
                                                         v
                                          backprop updates E_theta only

                         TabPFN weights stay frozen
```

For the 12-class source task, the trainer wraps that episode in an ECOC
many-class layer:

```text
12 source labels
      |
      v
ECOC codebook
      |
      v
multiple binary ECOC subtasks, controlled by encoder.tabpfn_max_classes: 2
      |
      v
frozen TabPFN support/query loss per subtask
      |
      v
decoded validation/test probabilities in the original 12-class space
```

After source training, the encoder is frozen and used as a feature map for
TabPFN context scans:

```text
downstream context X_context        downstream query X_query
              |                               |
              v                               v
        frozen encoder                  frozen encoder
              |                               |
              v                               v
      encoded context Z_context        encoded query Z_query
              |                               |
              |                               v
      context labels y_context ---> TabPFN prediction
                                      |
                                      v
                          baseline vs frozen-encoder metrics
```

## Encoder Choice

The default encoder is deliberately simple: a flat residual MLP initialized as
the identity. With the current CP feature set, `output_dim: 72` matches the raw
flat feature count, so the initial encoder is exactly the standardized TabPFN
input and training can only learn a small residual correction:

```text
encoder(x) = x + residual_scale * residual_mlp(x)
```

`residual_mlp` is strict: `output_dim` must equal the flat input feature count.
Use `mlp` only if you intentionally want a pure learned projection.

The repo also includes a lightweight GNN that uses every configured particle in
each event. It builds particle nodes from the jagged ROOT branches, runs a small
message-passing network, appends fixed event-summary features, and passes a 72D
hybrid output to TabPFN:

```text
particles + scalars -> GNN + event summaries -> 72D event vector -> TabPFN
```

Run it with [configs/source_gnn.yaml](configs/source_gnn.yaml). The GNN ignores particle
`max` values and uses all particles present in each event. The `max` entries in
the config are used by the flat residual default.

The repo also includes a particle transformer encoder. It uses the same graph
inputs and fixed event summaries as the GNN, but replaces message passing with
self-attention over all particles in an event:

```text
particles + scalars -> particle transformer + event summaries -> 72D event vector -> TabPFN
```

Run it with [configs/source_transformer.yaml](configs/source_transformer.yaml), or set
`encoder.type: transformer` in another config. `hidden_dim` must be divisible by
`attention_heads`.

The code also supports simpler flat residual variants. Set `encoder.type` to
`feature_mixer` for a stable residual linear feature mixer:

```text
encoder(x) = x + residual_scale * linear(x)
```

Set `encoder.type` to `feature_gate` for the most conservative per-feature
residual scale:

```text
encoder(x) = x * (1 + residual_scale * tanh(gate))
```

Source training clips encoder gradients, keeps TabPFN weights frozen, restores
the best validation-AUC encoder, and stops early after repeated non-improving
epochs. Gradients flow through TabPFN's differentiable input path into the
encoder only. For 12-class source training, the task is decomposed into balanced
binary ECOC subtasks and decoded back to the original class space; this gives
the encoder many simpler losses instead of nearly full 10-way surrogate tasks.
Validation is also episodic: each epoch rotates through validation support/query
episodes with the same 50/50 split used in training, then decodes the query
scores back to the original 12-class space.

## Fair Comparison Defaults

The nominal MLP, GNN, and transformer configs are sized to be comparable before
running the full workflow. They all emit the same 72-dimensional TabPFN feature
vector and have nearly matched trainable encoder parameter counts:

```text
encoder              output_dim   hidden_dim   layers   trainable params
residual_mlp              72           64         4          17,672
gnn                       72           40         1          17,904
transformer               72           28         1          18,100
```

These counts include only encoder parameters. TabPFN is frozen during source
training and is shared across model comparisons.

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

The main config is [configs/source_residual_mlp.yaml](configs/source_residual_mlp.yaml).

```yaml
output_dir: /global/cfs/projectdirs/atlas/joshua/tabpfn/runs/source_residual_mlp
cache_dir: /pscratch/sd/j/joshuaho/tabpfn
seed: 42
device: cuda

encoder:
  type: residual_mlp
  layers: 4
  hidden_dim: 64
  output_dim: 72
  epochs: 20
  learning_rate: 0.0002
  batch_size: 2048
  support_query_ratio: 0.5
  residual_scale: 0.1
  grad_clip_norm: 1.0
  early_stopping_patience: 8
  min_delta: 0.001
  tabpfn_max_classes: 2
  many_class_redundancy: 4
  validation_episodes: 8

dataset:
  raw_dir: /global/cfs/projectdirs/atlas/joshua/gnn_data/stats_100K
  split: {train: 0.5, val: 0.25, test: 0.25}
  labels:
    - {label: 0, files: [SingleT_schan.root]}
    - {label: 1, files: [VBF_NLO_inc.root]}
    - {label: 2, files: [WH_NLO_inc.root]}
    - {label: 3, files: [ZH_NLO_inc.root]}
    - {label: 4, files: [ggF_NLO_inc.root]}
    - {label: 5, files: [tHjb_NLO_inc.root]}
    - {label: 6, files: [ttH_NLO_inc.root]}
    - {label: 7, files: [ttW.root]}
    - {label: 8, files: [ttbar.root]}
    - {label: 9, files: [ttt.root]}
    - {label: 10, files: [tttt.root]}
    - {label: 11, files: [ttyy.root]}
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
  output_dir: /global/cfs/projectdirs/atlas/joshua/tabpfn/runs/source_residual_mlp/open_data_generalization
  tree_name: mini
  context_min_per_class: 100
  context_scan_points: 16
  context_repeats: 5
  query_chunk_size: 1024
  split: {train: 0.5, val: 0.25, test: 0.25}
  labels:
    - {label: 0, name: ttH, files: [mc_341081.ttH125_gamgam.GamGam.root]}
    - {label: 1, name: ggF, files: [mc_343981.ggH125_gamgam.GamGam.root]}
    - {label: 2, name: VBF, files: [mc_345041.VBFH125_gamgam.GamGam.root]}
    - {label: 3, name: WH, files: [mc_345318.WpH125J_Wincl_gamgam.GamGam.root]}
    - {label: 4, name: ZH, files: [mc_345319.ZH125J_Zincl_gamgam.GamGam.root]}
```

Every row from every configured source file is used. The default source task uses
the 12 non-CP-held-out ROOT files in `stats_100K`. `ttH_NLO.root` and
`ttH_CPodd.root` are intentionally excluded from source training and used for the
CP even/odd generalization test.

```text
classes: 12
split: 50/25/25 stratified train/validation/test
```

With `batch_size: 2048`, each source optimizer step samples a balanced
support/query episode of roughly that size. Source validation uses
`encoder.validation_episodes` rotating validation support/query episodes per
epoch. Downstream TabPFN inference is separate from source training: it scans
context sizes sampled from the downstream validation split, then predicts the
held-out test split in chunks of `transfer.query_chunk_size`.

```text
context: 100 events/class -> full validation split
scan points: 16
subsets per point: 5
query chunk: 1024
```

With the default `type: residual_mlp`, `output_dim: 72` is the flat feature count
sent to TabPFN. The residual branch is zero-initialized, so epoch-zero behavior is
the raw standardized TabPFN baseline.

With `type: gnn`, `output_dim: 72` is the event embedding size sent to TabPFN.
The GNN embedding is intentionally hybrid: it concatenates a learned graph
representation with fixed graph summary features, including global features,
event particle count, per-type particle counts, pooled particle statistics, and
per-particle-type pooled statistics. Start from:

```bash
bash scripts/run_source_encoder.sh configs/source_gnn.yaml
```

With `type: transformer`, the same `output_dim` and event-summary behavior apply,
but the learned representation comes from particle self-attention instead of GNN
message passing. Start from:

```bash
bash scripts/run_source_encoder.sh configs/source_transformer.yaml
```

With `type: residual_mlp`, `type: feature_mixer`, or `type: feature_gate`,
`output_dim` must match the number of flat input features, currently 72.

Source validation uses rotating validation support/query prompts and drives
early stopping. Downstream CP even/odd and open-data TabPFN tests use their own
validation split as the TabPFN context pool and their test split as the query
set. Set `transfer.context_size` only when you want to cap the largest scanned
context size; by default the scan reaches the full validation split.

## Data Cache

The first run reads ROOT files and saves processed `DatasetBundle` caches under:

```text
/pscratch/sd/j/joshuaho/tabpfn/source_multiclass/
/pscratch/sd/j/joshuaho/tabpfn/gamgam_production_modes/
```

Later runs load that cache and skip ROOT reading. The cache fingerprint includes
the input files, branches, split fractions, padding mode, seed, and whether graph
features were built. If those change, a new cache file is created.

When a cache does not exist yet, ROOT-to-feature creation is parallelized across
the configured input files. The worker count is
`min(os.cpu_count() or 1, number_of_files)`, so source training can use up to 12
workers, the held-out CP generalization task uses two workers, and the GamGam
transfer task uses up to five workers. During graph construction, each worker
prints a progress line every 50K events processed, plus one final line when that
file is complete.

## Training

Full workflow for the nominal benchmark:

```bash
bash scripts/run_full_workflow.sh
```

The dispatcher spelling also works:

```bash
bash scripts/run full workflow
```

This runs the residual MLP, GNN, and transformer configs. For each config, the
launcher trains on the 12-class source task, then runs the held-out CP even/odd
transfer and ATLAS open-data GamGam transfer evaluations.

On a multi-GPU node, the full workflow runs configs in parallel by default, one
config per visible GPU. Logs stream to the terminal with per-config prefixes and
are also written under `runs/workflow_logs/<timestamp>/`. For a 4-GPU node, the
default three encoder configs all start together.

To keep the terminal quiet and write only log files:

```bash
TABPFN_WORKFLOW_STREAM_LOGS=0 bash scripts/run_full_workflow.sh
```

To choose GPUs explicitly:

```bash
TABPFN_WORKFLOW_GPUS=0,1,2,3 bash scripts/run_full_workflow.sh
```

To force sequential execution:

```bash
TABPFN_WORKFLOW_PARALLEL=0 bash scripts/run_full_workflow.sh
```

To run only selected encoder configs:

```bash
bash scripts/run_full_workflow.sh \
  configs/source_residual_mlp.yaml \
  configs/source_transformer.yaml
```

Recommended launcher:

```bash
bash scripts/run_source_encoder.sh
```

The one-config launcher:

- Uses the `tabpfn` conda env automatically if the command is not already on `PATH`.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Uses an existing TabPFN checkpoint from `~/.cache/tabpfn` when available.
- Otherwise lets TabPFN download into `$SCRATCH/tabpfn_model_cache` by default.

To force a specific TabPFN checkpoint:

```bash
export TABPFN_MODEL_PATH=/path/to/model.ckpt
bash scripts/run_source_encoder.sh
```

To use a different config:

```bash
bash scripts/run_source_encoder.sh configs/other.yaml
```

Equivalent direct CLI:

```bash
tabpfn-encoder-train train --config configs/source_residual_mlp.yaml
```

## Source Training

The default `train` command trains only encoder weights, but the loss comes from
frozen TabPFN predictions. Each optimizer step samples support/query events,
passes both through the encoder, fits the TabPFN prompt on encoded support
features and support labels, predicts encoded query features, and backpropagates
the query loss through TabPFN's input path into the encoder. TabPFN model
parameters are not optimized.

For source tasks above TabPFN's class limit, the trainer builds an
error-correcting output-code (ECOC) codebook. Each ECOC column is a balanced
small-class TabPFN task; losses are averaged over these support/query subtasks
across training steps, and validation probabilities are decoded back to the
original source classes from rotating validation support/query episodes.

After source training, the run freezes the encoder and evaluates source,
CP even/odd, and open-data generalization with TabPFN:

1. `baseline_tabpfn`: TabPFN on the downstream flat features.
2. `frozen_encoder_tabpfn`: frozen source encoder output sent into TabPFN.
3. `delta`: frozen encoder metrics minus baseline metrics.

## Transfer Evaluation

The default `train` command runs three TabPFN context scans after source encoder
training: the 12-class source task itself, held-out CP even/odd, and the 5-class
ATLAS open-data Higgs production-mode task. The separate `transfer-source`,
`transfer-cp`, and `transfer` commands rerun those comparisons from a saved
checkpoint.

```text
ttH vs ggF vs VBF vs WH vs ZH
```

It compares two TabPFN evaluations on the same GamGam validation/test split:

1. `frozen_encoder_tabpfn`: source-trained encoder is frozen, validation events are encoded as the TabPFN context, and test events are encoded as the TabPFN query.
2. `baseline_tabpfn`: TabPFN is fit directly on CP-compatible flat validation context features and evaluated on test features.

The context scan starts at `transfer.context_min_per_class` events per class,
uses `transfer.context_scan_points` log-spaced context sizes, repeats each size
`transfer.context_repeats` times with different stratified validation-context
subsets, and ends at the full validation split unless `transfer.context_size`
caps it. Every scan point evaluates the full downstream test split, chunked by
`transfer.query_chunk_size`.

Run the standalone CP transfer command after training when you want to rerun only
the CP even/odd comparison:

```bash
bash scripts/run_cp_transfer.sh
```

or directly:

```bash
tabpfn-encoder-train transfer-cp --config configs/source_residual_mlp.yaml
```

CP transfer outputs are written to `output_dir/cp_generalization`:

```text
cp_even_odd_generalization_metrics.json
cp_even_odd_generalization_context_scan_metrics.json
cp_even_odd_generalization_context_scan_metrics.csv
cp_even_odd_generalization_context_scan_roc_auc.png
cp_even_odd_generalization_context_scan_accuracy.png
cp_even_odd_generalization_context_scan_log_loss.png
cp_even_odd_generalization_frozen_encoder_proba.npy
cp_even_odd_generalization_baseline_proba.npy
```

Run the standalone source transfer command when you want to rerun only the
12-class source-task comparison:

```bash
bash scripts/run_source_transfer.sh
```

or directly:

```bash
tabpfn-encoder-train transfer-source --config configs/source_residual_mlp.yaml
```

Source transfer outputs are written to `output_dir/source_generalization`:

```text
source_12_class_generalization_metrics.json
source_12_class_generalization_context_scan_metrics.json
source_12_class_generalization_context_scan_metrics.csv
source_12_class_generalization_context_scan_roc_auc.png
source_12_class_generalization_context_scan_accuracy.png
source_12_class_generalization_context_scan_log_loss.png
source_12_class_generalization_frozen_encoder_proba.npy
source_12_class_generalization_baseline_proba.npy
```

Run the standalone open-data transfer command when you want to rerun only the
GamGam comparison:

```bash
bash scripts/run_gamgam_transfer.sh
```

or directly:

```bash
tabpfn-encoder-train transfer --config configs/source_residual_mlp.yaml
```

To evaluate a specific saved encoder:

```bash
tabpfn-encoder-train transfer-cp \
  --config configs/source_residual_mlp.yaml \
  --model /path/to/encoder_classifier.pkl

tabpfn-encoder-train transfer-source \
  --config configs/source_residual_mlp.yaml \
  --model /path/to/encoder_classifier.pkl

tabpfn-encoder-train transfer \
  --config configs/source_residual_mlp.yaml \
  --model /path/to/encoder_classifier.pkl
```

Open-data transfer outputs are written to `transfer.output_dir`:

```text
open_data_generalization_metrics.json
open_data_generalization_context_scan_metrics.json
open_data_generalization_context_scan_metrics.csv
open_data_generalization_context_scan_roc_auc.png
open_data_generalization_context_scan_accuracy.png
open_data_generalization_context_scan_log_loss.png
open_data_generalization_frozen_encoder_proba.npy
open_data_generalization_baseline_proba.npy
```

After all three default encoders have finished, the full workflow writes
comparison PDFs to:

```text
/global/cfs/projectdirs/atlas/joshua/tabpfn/runs/context_scan_comparison/
```

The standalone plotting command is:

```bash
bash scripts/plot_context_comparison.sh
```

It makes AUC and accuracy PDFs for each task, with one baseline TabPFN curve and
one curve each for the MLP, GNN, and transformer encoders. Error bars are the
standard deviation over the repeated validation-context subsets.

## Expected Logs

A run should look like:

```text
Using TabPFN model: ...
Loading cached dataset: ...
Encoder+TabPFN settings: type=residual_mlp, device=cuda, layers=4, hidden_dim=64, output_dim=72, trainable_encoder_params=17672, batch_size=2048, support_query_ratio=0.5, learning_rate=0.0002, grad_clip_norm=1.0, validation_episodes=8, ecoc_tasks=16, alphabet_size=2
encoder_tabpfn epoch 1/20: train_loss=..., train_accuracy=..., train_roc_auc=..., grad_norm_mean=..., grad_norm_max=..., batches=49/49, val_loss=..., val_accuracy=..., val_roc_auc=...
source_12_class val: accuracy=..., log_loss=..., roc_auc=...
source_12_class test: accuracy=..., log_loss=..., roc_auc=...
source_12_class_generalization context=1200 repeat=1/5: baseline_auc=..., encoder_auc=..., delta_auc=...
source_12_class_generalization context=... repeat=.../5: baseline_auc=..., encoder_auc=..., delta_auc=...
source_12_class_generalization context scan: split=val, query=test, completed=.../..., summary_context=...
source_12_class_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
source_12_class_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
source_12_class_generalization delta: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization context=200 repeat=1/5: baseline_auc=..., encoder_auc=..., delta_auc=...
cp_even_odd_generalization context=... repeat=.../5: baseline_auc=..., encoder_auc=..., delta_auc=...
cp_even_odd_generalization context scan: split=val, query=test, completed=.../..., summary_context=...
cp_even_odd_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
cp_even_odd_generalization delta: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization context=... repeat=.../5: baseline_auc=..., encoder_auc=..., delta_auc=...
open_data_generalization context scan: split=val, query=test, completed=.../..., summary_context=...
open_data_generalization baseline_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization frozen_encoder_tabpfn: accuracy=..., log_loss=..., roc_auc=...
open_data_generalization delta: accuracy=..., log_loss=..., roc_auc=...
```

Metrics are printed to three decimal places in the terminal. Saved CSV/JSON files
keep full precision.

Use `configs/source_gnn.yaml` or `configs/source_transformer.yaml` for graph experiments.
For transformer runs, the settings line also prints `attention_heads`.

## Outputs

Artifacts are written to `output_dir`:

```text
run_metadata.json
metrics.json
training_summary.json
epoch_metrics.csv
encoder_classifier.pkl
best_checkpoint.json
source_generalization/source_12_class_generalization_metrics.json
source_generalization/source_12_class_generalization_context_scan_metrics.json
source_generalization/source_12_class_generalization_context_scan_metrics.csv
source_generalization/source_12_class_generalization_context_scan_roc_auc.png
source_generalization/source_12_class_generalization_context_scan_accuracy.png
source_generalization/source_12_class_generalization_context_scan_log_loss.png
source_generalization/source_12_class_generalization_baseline_proba.npy
source_generalization/source_12_class_generalization_frozen_encoder_proba.npy
cp_generalization/cp_even_odd_generalization_metrics.json
cp_generalization/cp_even_odd_generalization_context_scan_metrics.json
cp_generalization/cp_even_odd_generalization_context_scan_metrics.csv
cp_generalization/cp_even_odd_generalization_context_scan_roc_auc.png
cp_generalization/cp_even_odd_generalization_context_scan_accuracy.png
cp_generalization/cp_even_odd_generalization_context_scan_log_loss.png
cp_generalization/cp_even_odd_generalization_baseline_proba.npy
cp_generalization/cp_even_odd_generalization_frozen_encoder_proba.npy
```

Open-data generalization artifacts are written to `transfer.output_dir`:

```text
open_data_generalization_metrics.json
open_data_generalization_context_scan_metrics.json
open_data_generalization_context_scan_metrics.csv
open_data_generalization_context_scan_roc_auc.png
open_data_generalization_context_scan_accuracy.png
open_data_generalization_context_scan_log_loss.png
open_data_generalization_baseline_proba.npy
open_data_generalization_frozen_encoder_proba.npy
```

Cross-encoder comparison PDFs are written to the shared runs directory:

```text
context_scan_comparison/source_12_class_generalization_roc_auc_comparison.pdf
context_scan_comparison/source_12_class_generalization_accuracy_comparison.pdf
context_scan_comparison/cp_even_odd_generalization_roc_auc_comparison.pdf
context_scan_comparison/cp_even_odd_generalization_accuracy_comparison.pdf
context_scan_comparison/open_data_generalization_roc_auc_comparison.pdf
context_scan_comparison/open_data_generalization_accuracy_comparison.pdf
```

`epoch_metrics.csv` is the easiest file to inspect during development. It contains
one row per epoch with train loss/accuracy/AUC and validation loss/accuracy/AUC.
`metrics.json` contains source validation/test metrics plus source, CP even/odd,
and open-data generalization summaries. `encoder_classifier.pkl` is the checkpoint
to load for standalone transfer reruns. It keeps the trained encoder, ECOC
metadata, label scheme, and preprocessing state on CPU so it can be reused
without a GPU session. If
`device: cuda` is set on a machine without CUDA, the trainer automatically falls
back to CPU.

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

Avoid very large training batches like `8192` unless you have confirmed enough
GPU memory. For transfer, the scan starts small and grows toward the full
validation split. If a large context size OOMs, the scan records that failed
point, keeps the smaller completed points, and stops the larger part of the
scan. To cap the scan manually, set `transfer.context_size`; reduce
`transfer.query_chunk_size` only if test prediction chunks are the bottleneck.

## Tests

```bash
bash scripts/run_tests.sh
```

To run a subset, pass pytest paths or flags through:

```bash
bash scripts/run_tests.sh tests/test_config.py
```
