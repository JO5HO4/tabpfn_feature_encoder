#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
    configs=("$@")
else
    configs=(
        "${REPO_ROOT}/configs/source_residual_mlp.yaml"
        "${REPO_ROOT}/configs/source_gnn.yaml"
        "${REPO_ROOT}/configs/source_transformer.yaml"
    )
fi

gpu_ids=()
if [[ -n "${TABPFN_WORKFLOW_GPUS:-}" ]]; then
    IFS="," read -r -a gpu_ids <<< "${TABPFN_WORKFLOW_GPUS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS="," read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t gpu_ids < <(
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d " "
    )
fi

clean_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ -n "${gpu_id}" ]]; then
        clean_gpu_ids+=("${gpu_id}")
    fi
done
gpu_ids=("${clean_gpu_ids[@]}")

parallel_enabled="${TABPFN_WORKFLOW_PARALLEL:-1}"
max_jobs=1
if [[ "${parallel_enabled}" != "0" && "${#gpu_ids[@]}" -gt 1 ]]; then
    max_jobs="${#gpu_ids[@]}"
fi
if [[ "${max_jobs}" -gt "${#configs[@]}" ]]; then
    max_jobs="${#configs[@]}"
fi

echo "Running full TabPFN feature-encoder workflow for ${#configs[@]} config(s)."
echo "Each config trains the source encoder, then runs CP even/odd and GamGam transfer evaluations."

for config in "${configs[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi
done

if [[ "${max_jobs}" -le 1 ]]; then
    if [[ "${#gpu_ids[@]}" -gt 0 ]]; then
        echo "Running sequentially on GPU ${gpu_ids[0]}."
    else
        echo "Running sequentially. No GPU list was detected."
    fi

    for config in "${configs[@]}"; do
        echo
        echo "================================================================"
        echo "Running workflow config: ${config}"
        echo "================================================================"
        if [[ "${#gpu_ids[@]}" -gt 0 ]]; then
            CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${SCRIPT_DIR}/run_source_encoder.sh" "${config}"
        else
            "${SCRIPT_DIR}/run_source_encoder.sh" "${config}"
        fi
    done

    echo
    echo "Full workflow complete."
    exit 0
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${TABPFN_WORKFLOW_LOG_DIR:-${REPO_ROOT}/runs/workflow_logs/${timestamp}}"
mkdir -p "${log_dir}"

echo "Running up to ${max_jobs} config(s) in parallel on GPUs: ${gpu_ids[*]}"
echo "Workflow logs: ${log_dir}"

pids=()
names=()
logs=()
failures=0

wait_for_wave() {
    local idx
    for idx in "${!pids[@]}"; do
        if wait "${pids[$idx]}"; then
            echo "Finished ${names[$idx]} successfully. Log: ${logs[$idx]}"
        else
            echo "FAILED ${names[$idx]}. Log: ${logs[$idx]}" >&2
            echo "Last 80 log lines for ${names[$idx]}:" >&2
            tail -n 80 "${logs[$idx]}" >&2 || true
            failures=$((failures + 1))
        fi
    done
    pids=()
    names=()
    logs=()
}

for idx in "${!configs[@]}"; do
    config="${configs[$idx]}"
    slot=$((idx % max_jobs))
    gpu_id="${gpu_ids[$slot]}"
    name="$(basename "${config}")"
    name="${name%.*}"
    log_path="${log_dir}/${name}.log"

    echo
    echo "================================================================"
    echo "Starting workflow config: ${config}"
    echo "GPU: ${gpu_id}"
    echo "Log: ${log_path}"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${SCRIPT_DIR}/run_source_encoder.sh" "${config}" \
        >"${log_path}" 2>&1 &
    pids+=("$!")
    names+=("${name}")
    logs+=("${log_path}")

    if [[ "${#pids[@]}" -ge "${max_jobs}" ]]; then
        wait_for_wave
    fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_for_wave
fi

if [[ "${failures}" -gt 0 ]]; then
    echo
    echo "Full workflow finished with ${failures} failed config(s)." >&2
    exit 1
fi

echo
echo "Full workflow complete."
