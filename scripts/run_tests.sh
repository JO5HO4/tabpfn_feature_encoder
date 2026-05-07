#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TABPFN_ENV_NAME:-tabpfn}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

run_in_env() {
    if command -v conda >/dev/null 2>&1; then
        conda run --no-capture-output -n "${ENV_NAME}" "$@"
    else
        "$@"
    fi
}

run_in_env python -m compileall -q src tests
run_in_env python -m pytest -q "$@"
