#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CONFIGS=(
    "${REPO_ROOT}/configs/source_residual_mlp.yaml"
    "${REPO_ROOT}/configs/source_gnn.yaml"
    "${REPO_ROOT}/configs/source_transformer.yaml"
)

usage() {
    cat <<'EOF'
Submit TabPFN feature-encoder jobs to Perlmutter Slurm.

Usage:
  scripts/submit_perlmutter.sh full-workflow [config ...]
  scripts/submit_perlmutter.sh source-encoders [config ...]
  scripts/submit_perlmutter.sh transfer-suite [config] [model]

Modes:
  full-workflow    One multi-GPU job that runs scripts/run_full_workflow.sh.
  source-encoders  Separate 1-GPU jobs, one job per encoder config.
  transfer-suite   One 3-GPU job that runs source, CP, and GamGam transfer scans.

Common environment overrides:
  NERSC_ACCOUNT          Slurm account, default: atlas
  NERSC_QOS              Slurm QOS, default: shared
  NERSC_TIME             Walltime, default: 12:00:00
  NERSC_CONSTRAINT       Node constraint, default: gpu
  NERSC_LOG_DIR          Slurm log dir, default: runs/slurm_logs
  NERSC_JOB_DIR          Generated sbatch script dir, default: runs/slurm_jobs
  NERSC_DRY_RUN=1        Write the sbatch script but do not submit it

Mode-specific overrides:
  NERSC_GPUS             GPUs for full-workflow/transfer-suite, default: 4 or 3
  NERSC_CPUS             CPUs for full-workflow job, default: 32 * NERSC_GPUS
  NERSC_CPUS_PER_GPU     CPUs per requested GPU, default: 32
  NERSC_CPUS_PER_TASK    Override CPUs for 1-GPU encoder jobs

Examples:
  scripts/submit_perlmutter.sh source-encoders
  scripts/submit_perlmutter.sh source-encoders configs/source_gnn.yaml
  scripts/submit_perlmutter.sh transfer-suite configs/source_gnn.yaml
EOF
}

timestamp_utc() {
    date -u +"%Y%m%dT%H%M%SZ"
}

shell_join() {
    local quoted=()
    local item
    local piece
    for item in "$@"; do
        printf -v piece "%q" "${item}"
        quoted+=("${piece}")
    done
    printf "%s" "${quoted[*]}"
}

require_sbatch() {
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "Could not find sbatch. Run this from a Perlmutter login node." >&2
        exit 127
    fi
}

validate_paths() {
    local path
    for path in "$@"; do
        if [[ ! -f "${path}" ]]; then
            echo "Missing file: ${path}" >&2
            exit 1
        fi
    done
}

gpu_list() {
    local count="$1"
    local ids=()
    local idx
    for ((idx = 0; idx < count; idx++)); do
        ids+=("${idx}")
    done
    local IFS=,
    echo "${ids[*]}"
}

write_or_submit() {
    local job_file="$1"
    echo "Wrote sbatch script: ${job_file}"
    if [[ "${NERSC_DRY_RUN:-0}" == "1" ]]; then
        echo "NERSC_DRY_RUN=1, not submitting."
        return
    fi
    require_sbatch
    sbatch "${job_file}"
}

common_setup() {
    ACCOUNT="${NERSC_ACCOUNT:-${SBATCH_ACCOUNT:-${SLURM_ACCOUNT:-atlas}}}"
    QOS="${NERSC_QOS:-shared}"
    TIME_LIMIT="${NERSC_TIME:-12:00:00}"
    CONSTRAINT="${NERSC_CONSTRAINT:-gpu}"
    LOG_DIR="${NERSC_LOG_DIR:-${REPO_ROOT}/runs/slurm_logs}"
    JOB_DIR="${NERSC_JOB_DIR:-${REPO_ROOT}/runs/slurm_jobs}"
    mkdir -p "${LOG_DIR}" "${JOB_DIR}"
}

submit_full_workflow() {
    common_setup
    local configs=("$@")
    if [[ "${#configs[@]}" -eq 0 ]]; then
        configs=("${DEFAULT_CONFIGS[@]}")
    fi
    validate_paths "${configs[@]}"

    local gpus="${NERSC_GPUS:-4}"
    local cpus_per_gpu="${NERSC_CPUS_PER_GPU:-32}"
    local cpus="${NERSC_CPUS:-$((gpus * cpus_per_gpu))}"
    local stamp
    stamp="$(timestamp_utc)"
    local job_name="tabpfn-full-${stamp}"
    local job_file="${JOB_DIR}/${job_name}.slurm"
    local config_args
    config_args="$(shell_join "${configs[@]}")"
    local visible_gpus
    visible_gpus="$(gpu_list "${gpus}")"

    cat >"${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail

cd ${REPO_ROOT@Q}
export PYTHONUNBUFFERED="\${PYTHONUNBUFFERED:-1}"
export TABPFN_WORKFLOW_GPUS="\${TABPFN_WORKFLOW_GPUS:-${visible_gpus}}"
export TABPFN_WORKFLOW_LOG_DIR="\${TABPFN_WORKFLOW_LOG_DIR:-${LOG_DIR}/${job_name}_workflow_logs}"

echo "Job: \${SLURM_JOB_ID:-unknown}"
echo "Node list: \${SLURM_JOB_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "TABPFN_WORKFLOW_GPUS: \${TABPFN_WORKFLOW_GPUS}"

bash scripts/run_full_workflow.sh ${config_args}
EOF
    write_or_submit "${job_file}"
}

submit_source_encoders() {
    common_setup
    local configs=("$@")
    if [[ "${#configs[@]}" -eq 0 ]]; then
        configs=("${DEFAULT_CONFIGS[@]}")
    fi
    validate_paths "${configs[@]}"

    local cpus="${NERSC_CPUS_PER_TASK:-${NERSC_CPUS_PER_GPU:-32}}"
    local stamp
    stamp="$(timestamp_utc)"
    local config
    local config_name
    local job_name
    local job_file
    for config in "${configs[@]}"; do
        config_name="$(basename "${config}")"
        config_name="${config_name%.*}"
        job_name="tabpfn-${config_name}-${stamp}"
        job_file="${JOB_DIR}/${job_name}.slurm"

        cat >"${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail

cd ${REPO_ROOT@Q}
config=${config@Q}

echo "Job: \${SLURM_JOB_ID:-unknown}"
echo "Node list: \${SLURM_JOB_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: \${config}"

bash scripts/run_source_encoder.sh "\${config}"
EOF
        write_or_submit "${job_file}"
    done
}

submit_transfer_suite() {
    common_setup
    local config="${1:-${REPO_ROOT}/configs/source_residual_mlp.yaml}"
    local model="${2:-}"
    validate_paths "${config}"
    if [[ -n "${model}" ]]; then
        validate_paths "${model}"
    fi

    local gpus="${NERSC_GPUS:-3}"
    if [[ "${gpus}" -lt 3 ]]; then
        echo "transfer-suite needs at least 3 GPUs. Set NERSC_GPUS=3 or higher." >&2
        exit 1
    fi
    local cpus_per_task="${NERSC_CPUS_PER_TASK:-${NERSC_CPUS_PER_GPU:-32}}"
    local cpus="${NERSC_CPUS:-$((gpus * cpus_per_task))}"
    local stamp
    stamp="$(timestamp_utc)"
    local job_name="tabpfn-transfer-${stamp}"
    local job_file="${JOB_DIR}/${job_name}.slurm"
    local model_arg=""
    if [[ -n "${model}" ]]; then
        printf -v model_arg "%q" "${model}"
    fi

    cat >"${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --constraint=${CONSTRAINT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${gpus}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail

cd ${REPO_ROOT@Q}
config=${config@Q}
model=${model@Q}
run_log_dir="${LOG_DIR}/${job_name}_transfer_logs"
mkdir -p "\${run_log_dir}"

echo "Job: \${SLURM_JOB_ID:-unknown}"
echo "Node list: \${SLURM_JOB_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: \${config}"
echo "Model: \${model:-default output_dir/encoder_classifier.pkl}"
echo "Transfer logs: \${run_log_dir}"

CUDA_VISIBLE_DEVICES=0 bash scripts/run_source_transfer.sh "\${config}" ${model_arg} >"\${run_log_dir}/source.log" 2>&1 &
pid_source=\$!
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cp_transfer.sh "\${config}" ${model_arg} >"\${run_log_dir}/cp.log" 2>&1 &
pid_cp=\$!
CUDA_VISIBLE_DEVICES=2 bash scripts/run_gamgam_transfer.sh "\${config}" ${model_arg} >"\${run_log_dir}/gamgam.log" 2>&1 &
pid_gamgam=\$!

failures=0
for item in source:\${pid_source} cp:\${pid_cp} gamgam:\${pid_gamgam}; do
    name="\${item%%:*}"
    pid="\${item##*:}"
    if wait "\${pid}"; then
        echo "Finished \${name}. Log: \${run_log_dir}/\${name}.log"
    else
        echo "FAILED \${name}. Log: \${run_log_dir}/\${name}.log" >&2
        tail -n 80 "\${run_log_dir}/\${name}.log" >&2 || true
        failures=\$((failures + 1))
    fi
done

if [[ "\${failures}" -gt 0 ]]; then
    exit 1
fi
EOF
    write_or_submit "${job_file}"
}

mode="${1:-}"
if [[ -z "${mode}" || "${mode}" == "-h" || "${mode}" == "--help" ]]; then
    usage
    exit 0
fi
shift

case "${mode}" in
    full|full-workflow)
        submit_full_workflow "$@"
        ;;
    encoders|source-encoders)
        submit_source_encoders "$@"
        ;;
    transfers|transfer-suite)
        submit_transfer_suite "$@"
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac
