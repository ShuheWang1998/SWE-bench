#!/usr/bin/env bash
#
# Launch 2 independent vLLM servers, each sharding Qwen3.5-9B across 4 GPUs
# with tp=4. The two processes do not share any state (no DP coordinator,
# no zmq wave); they are treated as two external backends and the client
# does HTTP-layer load balancing between them.
#
# Why not `--data-parallel-size 2` inside a single server?
#   vLLM's internal DP load balancer in 0.19 is a "least loaded" picker
#   that uses coordinator-reported waiting/running counts refreshed every
#   ~100ms. For Qwen3.5-9B (non-MoE) + tp=4 on fast H100s, most requests
#   finish inside that window, so the picker sees all replicas looking
#   "empty" and repeatedly selects rank 0. Result: 4 GPUs pinned at 97%
#   and 4 GPUs idle (nvidia-smi-confirmed). The upstream DP docs
#   explicitly recommend external load balancing for non-MoE models:
#   https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html
#
# GPU assignment: rank 0 -> GPUs 0..3, rank 1 -> GPUs 4..7.
# Ports: rank 0 -> 8000, rank 1 -> 8001.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HOME}/.cache/swebench-pycache-${HOSTNAME:-$(hostname)}"
find "$(dirname "$0")/../distributed" -type d -name __pycache__ -prune -print0 2>/dev/null \
    | xargs -0 -r rm -rf 2>/dev/null || true

MODEL_PATH="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"
HOST="0.0.0.0"

# Qwen3.5-9B has 4 KV heads, so vLLM constrains tp to {1, 2, 4}. With 8
# H100s we launch two tp=4 replicas, each owning 4 GPUs. Each replica
# fits the model's native 262144-token context (KV ~32 GiB/seq / 4 tp =
# ~8 GiB per GPU per full seq, alongside ~4.5 GiB sharded weights).
TENSOR_PARALLEL_SIZE="4"
MAX_MODEL_LEN="262144"
GPU_MEMORY_UTILIZATION="0.95"
DTYPE="bfloat16"

# Port layout.
PORT_RANK0="8000"
PORT_RANK1="8001"

# Log files; tmux capture can be annoying when two processes interleave,
# so mirror their stdout/stderr to individual logs that are easy to tail.
LOG_DIR="${HOME}/.cache/swebench-vllm-logs"
mkdir -p "${LOG_DIR}"
LOG_RANK0="${LOG_DIR}/vllm-rank0.log"
LOG_RANK1="${LOG_DIR}/vllm-rank1.log"

# Launch helper.
launch_rank() {
    local rank="$1"; shift
    local gpus="$1"; shift
    local port="$1"; shift
    local logfile="$1"; shift
    echo "[serve_model.sh] Launching rank ${rank} on GPUs ${gpus}, port ${port}, log ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    python -m distributed.serve_model \
        --model_path "${MODEL_PATH}" \
        --served_model_name "${SERVED_MODEL_NAME}" \
        --host "${HOST}" --port "${port}" \
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
        --data_parallel_size 1 \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --dtype "${DTYPE}" \
        >"${logfile}" 2>&1 &
    echo $!
}

# Make sure all 8 GPUs are actually visible to this shell (the Python
# pre-flight check in distributed/serve_model.py trusts CUDA_VISIBLE_DEVICES).
if ! command -v nvidia-smi >/dev/null; then
    echo "[serve_model.sh] ERROR: nvidia-smi not available, aborting." >&2
    exit 2
fi
VISIBLE=$(nvidia-smi -L | wc -l)
if [ "${VISIBLE}" -lt 8 ]; then
    echo "[serve_model.sh] ERROR: expected 8 GPUs, only found ${VISIBLE}." >&2
    exit 3
fi

PID0=$(launch_rank 0 "0,1,2,3" "${PORT_RANK0}" "${LOG_RANK0}")
PID1=$(launch_rank 1 "4,5,6,7" "${PORT_RANK1}" "${LOG_RANK1}")
echo "[serve_model.sh] Launched rank 0 (pid=${PID0}) and rank 1 (pid=${PID1})."
echo "[serve_model.sh] Tail the logs with:"
echo "    tail -F ${LOG_RANK0} ${LOG_RANK1}"

# Propagate SIGINT/SIGTERM to both children so ^C from tmux actually
# kills both vLLM instances.
term_children() {
    echo "[serve_model.sh] Caught signal, terminating children ${PID0} ${PID1}."
    kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true
    wait "${PID0}" "${PID1}" 2>/dev/null || true
    exit 0
}
trap term_children INT TERM

# Wait for either child to exit; if one does, kill the other so we never
# silently run with just one server. The client expects both.
set +e
wait -n "${PID0}" "${PID1}"
rc=$?
echo "[serve_model.sh] A child exited with status ${rc}; bringing down the other." >&2
kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true
wait "${PID0}" "${PID1}" 2>/dev/null
exit "${rc}"
