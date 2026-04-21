#!/usr/bin/env bash
set -euo pipefail

# --- Lustre safety ---------------------------------------------------------
# The repo lives on a shared filesystem, so the GPU box and the CPU box
# both see the same distributed/*.py. If one node writes .pyc bytecode
# under distributed/__pycache__/ and the clock skew / mtime granularity
# on Lustre lets an older .pyc look "up to date" on the other node,
# Python will silently execute stale code. We avoid that class of bug
# entirely by (a) not writing .pyc here and (b) storing any .pyc we do
# produce in a node-local directory keyed by hostname.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HOME}/.cache/swebench-pycache-${HOSTNAME:-$(hostname)}"
# If a previous run left a shared __pycache__ in the repo, nuke it so we
# can't be tricked into loading ancient bytecode.
find "$(dirname "$0")/../distributed" -type d -name __pycache__ -prune -print0 2>/dev/null \
    | xargs -0 -r rm -rf 2>/dev/null || true
# ---------------------------------------------------------------------------

DATASET_NAME_OR_PATH="princeton-nlp/SWE-bench_oracle"
SPLIT="test"
MODEL_NAME_OR_PATH="Qwen3.5-9B"
BASE_URL="http://localhost:40220/v1"
OUTPUT_DIR="/home/wangshuhe/results/swe_qwen35_9b_outputs"
TOKENIZER="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
MAX_NEW_TOKENS="0"


python -m distributed.run_api_remote \
    --dataset_name_or_path ${DATASET_NAME_OR_PATH} --split ${SPLIT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --base_url ${BASE_URL} --output_dir ${OUTPUT_DIR} \
    --tokenizer ${TOKENIZER} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --chat