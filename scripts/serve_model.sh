#!/usr/bin/env bash
set -euo pipefail

# Node-local bytecode cache; see scripts/run_inference.sh for rationale.
# Same repo lives on Lustre and is shared with the CPU box, so neither
# side should write .pyc files into distributed/__pycache__.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HOME}/.cache/swebench-pycache-${HOSTNAME:-$(hostname)}"
find "$(dirname "$0")/../distributed" -type d -name __pycache__ -prune -print0 2>/dev/null \
    | xargs -0 -r rm -rf 2>/dev/null || true

MODEL_PATH="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"
HOST="0.0.0.0"
PORT="8000"
# Qwen3.5-9B has 4 KV heads, so vLLM constrains tp to {1, 2, 4}. With 8
# H100s the best "max-tp" topology is therefore tp=4 * dp=2: two replicas
# each sharded across 4 GPUs, which covers all 8 devices and splits
# weights+KV cache by 4 per GPU. That in turn makes the model's native
# 262144-token context window fit without sacrificing throughput.
TENSOR_PARALLEL_SIZE="4"
DATA_PARALLEL_SIZE="2"
# Native max for Qwen3.5-9B (config.json max_position_embeddings).
# KV cache ~32 GiB/seq at this ctx, but /4 after tp=4 sharding -> ~8 GiB
# per GPU per full seq, which fits alongside the ~4.5 GiB sharded weights.
MAX_MODEL_LEN="262144"
# tp=4 leaves generous headroom per GPU; push utilization up so we have
# the KV-cache budget to keep both replicas busy even when prompts are
# deep into the 262k window.
GPU_MEMORY_UTILIZATION="0.95"
DTYPE="bfloat16"



python -m distributed.serve_model \
    --model_path ${MODEL_PATH} \
    --served_model_name ${SERVED_MODEL_NAME} \
    --host ${HOST} --port ${PORT} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --data_parallel_size ${DATA_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --max_model_len ${MAX_MODEL_LEN} \
    --dtype ${DTYPE}