# Environment Setup

From the repo root:

```bash
conda env create -f setup/environment.yml
conda activate tabpfn
python -m pip install -e ".[train,atlas,plots]"
```

Check the important imports:

```bash
python - <<'PY'
import torch
import tabpfn
import uproot

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("tabpfn", getattr(tabpfn, "__version__", "installed"))
print("uproot", uproot.__version__)
PY
```

Launch training:

```bash
./scripts/run_cp_encoder.sh
```

The runner uses an existing checkpoint from `~/.cache/tabpfn` when available. Set
`TABPFN_MODEL_PATH=/path/to/model.ckpt` to override it.
