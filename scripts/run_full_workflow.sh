#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    echo "[$(timestamp_utc)] $*"
}

prefix_stream() {
    local name="$1"
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
        echo "[$(timestamp_utc)] [${name}] ${line}"
    done
}

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

log "Running full TabPFN feature-encoder workflow for ${#configs[@]} config(s)."
log "Each config trains the source encoder, then runs source, CP even/odd, and GamGam transfer evaluations."

plot_comparison() {
    if [[ "${TABPFN_WORKFLOW_PLOT:-1}" == "0" ]]; then
        return
    fi
    echo
    log "================================================================"
    log "Plotting context-scan comparisons"
    log "================================================================"
    "${SCRIPT_DIR}/plot_context_comparison.sh"
}

for config in "${configs[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi
done

if [[ "${max_jobs}" -le 1 ]]; then
    if [[ "${#gpu_ids[@]}" -gt 0 ]]; then
        log "Running sequentially on GPU ${gpu_ids[0]}."
    else
        log "Running sequentially. No GPU list was detected."
    fi

    for idx in "${!configs[@]}"; do
        config="${configs[$idx]}"
        name="$(basename "${config}")"
        name="${name%.*}"
        echo
        log "================================================================"
        log "Starting config $((idx + 1))/${#configs[@]}: ${name}"
        log "Config path: ${config}"
        log "================================================================"
        if [[ "${#gpu_ids[@]}" -gt 0 ]]; then
            CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${SCRIPT_DIR}/run_source_encoder.sh" "${config}"
        else
            "${SCRIPT_DIR}/run_source_encoder.sh" "${config}"
        fi
        log "Finished config $((idx + 1))/${#configs[@]}: ${name}"
    done

    echo
    plot_comparison
    echo
    log "Full workflow complete."
    exit 0
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${TABPFN_WORKFLOW_LOG_DIR:-${REPO_ROOT}/runs/workflow_logs/${timestamp}}"
mkdir -p "${log_dir}"

stream_logs="${TABPFN_WORKFLOW_STREAM_LOGS:-1}"

log "Running up to ${max_jobs} config(s) in parallel on GPUs: ${gpu_ids[*]}"
log "Workflow logs: ${log_dir}"
if [[ "${stream_logs}" == "0" ]]; then
    log "Live log streaming is disabled by TABPFN_WORKFLOW_STREAM_LOGS=0."
else
    log "Streaming live logs with per-config prefixes. Full logs are still saved."
fi

pids=()
names=()
logs=()
failures=0

wait_for_wave() {
    local idx
    for idx in "${!pids[@]}"; do
        if wait "${pids[$idx]}"; then
            log "Finished ${names[$idx]} successfully. Log: ${logs[$idx]}"
        else
            log "FAILED ${names[$idx]}. Log: ${logs[$idx]}" >&2
            log "Last 80 log lines for ${names[$idx]}:" >&2
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
    log "================================================================"
    log "Starting config $((idx + 1))/${#configs[@]}: ${name}"
    log "Config path: ${config}"
    log "GPU: ${gpu_id}"
    log "Log: ${log_path}"
    log "================================================================"

    if [[ "${stream_logs}" == "0" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${SCRIPT_DIR}/run_source_encoder.sh" "${config}" \
            >"${log_path}" 2>&1 &
    else
        (
            set -o pipefail
            CUDA_VISIBLE_DEVICES="${gpu_id}" "${SCRIPT_DIR}/run_source_encoder.sh" "${config}" 2>&1 \
                | prefix_stream "${name}" \
                | tee "${log_path}"
        ) &
    fi
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
    log "Full workflow finished with ${failures} failed config(s)." >&2
    exit 1
fi

echo
plot_comparison
echo
log "Full workflow complete."
