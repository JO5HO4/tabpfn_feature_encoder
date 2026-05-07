#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
    CONFIGS=("$@")
else
    CONFIGS=(
        "${REPO_ROOT}/configs/source_residual_mlp.yaml"
        "${REPO_ROOT}/configs/source_gnn.yaml"
        "${REPO_ROOT}/configs/source_transformer.yaml"
    )
fi

echo "Running full TabPFN feature-encoder workflow for ${#CONFIGS[@]} config(s)."
echo "Each config trains the source encoder, then runs CP even/odd and GamGam transfer evaluations."

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi

    echo
    echo "================================================================"
    echo "Running workflow config: ${config}"
    echo "================================================================"
    "${SCRIPT_DIR}/run_source_encoder.sh" "${config}"
done

echo
echo "Full workflow complete."
