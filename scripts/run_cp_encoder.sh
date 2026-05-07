#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TABPFN_ENV_NAME:-tabpfn}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/cp_encoder.yaml}"

export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/runs}/tabpfn_model_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${TABPFN_MODEL_PATH:-}" ]]; then
    for candidate in \
        "${HOME}/.cache/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt" \
        "${HOME}/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt" \
        "${HOME}/.cache/tabpfn/tabpfn-v2-classifier.ckpt"
    do
        if [[ -s "${candidate}" ]]; then
            export TABPFN_MODEL_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -n "${TABPFN_MODEL_PATH:-}" ]]; then
    echo "Using TabPFN model: ${TABPFN_MODEL_PATH}"
else
    echo "Using TabPFN model cache: ${TABPFN_MODEL_CACHE_DIR}"
fi

if command -v tabpfn-encoder-train >/dev/null 2>&1; then
    tabpfn-encoder-train train --config "${CONFIG_PATH}"
elif command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "${ENV_NAME}" \
        tabpfn-encoder-train train --config "${CONFIG_PATH}"
else
    echo "Could not find tabpfn-encoder-train or conda." >&2
    echo "Activate the env with: conda activate ${ENV_NAME}" >&2
    return 127 2>/dev/null || exit 127
fi
