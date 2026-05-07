#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TABPFN_ENV_NAME:-tabpfn}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/runs"
OUTPUT_DIR="${1:-${RUN_ROOT}/context_scan_comparison}"

cd "${REPO_ROOT}"

args=(
    plot-context-comparison
    --output-dir "${OUTPUT_DIR}"
    --run "MLP encoder" "${RUN_ROOT}/source_residual_mlp"
    --run "GNN encoder" "${RUN_ROOT}/source_gnn"
    --run "Transformer encoder" "${RUN_ROOT}/source_transformer"
)

if command -v tabpfn-encoder-train >/dev/null 2>&1; then
    tabpfn-encoder-train "${args[@]}"
elif command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "${ENV_NAME}" tabpfn-encoder-train "${args[@]}"
else
    echo "Could not find tabpfn-encoder-train or conda." >&2
    echo "Activate the env with: conda activate ${ENV_NAME}" >&2
    return 127 2>/dev/null || exit 127
fi
