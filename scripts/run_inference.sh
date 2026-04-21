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
# Two independent vLLM replicas (tp=4 each) launched by scripts/serve_model.sh.
# Pass both URLs as a comma-separated list; the client picks whichever backend
# has the fewest in-flight requests and breaks ties with round-robin so it
# alternates send/receive between them instead of serializing on one replica.
# Replace localhost:40220/40221 with the tunnel ports you use to reach the
# GPU box (the example below assumes two SSH tunnels on those ports; if you
# run the client on the GPU box directly, use http://localhost:8000/v1 and
# http://localhost:8001/v1 to match scripts/serve_model.sh).
BASE_URL="http://localhost:40220/v1,http://localhost:40221/v1"
OUTPUT_DIR="/home/wangshuhe/results/swe_qwen35_9b_outputs"
TOKENIZER="/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B"
# Upstream swebench.inference.run_llama.py uses 200 as the default output
# budget. This is plenty for SWE-bench target patches (typically 50-500
# tokens) and prevents pathological runaway generations from pinning a
# replica for minutes. Pass ``0`` here to let the model run until it hits
# EOS or the hard context limit (slower, but allowed).
MAX_NEW_TOKENS="0"
# vLLM's ``--data_parallel_size`` (on the GPU box) is the ceiling on how
# many requests this client should have in flight at once: more doesn't
# help, less leaves replicas idle. The server is currently tp=4 dp=2, so
# 2 is the natural replica-matched value. vLLM's continuous batching
# inside each replica can absorb several more in-flight requests though,
# so we set the client slightly higher (4x replicas) to keep both boxes
# fully pipelined without starving KV cache.
CONCURRENCY="2"

# --- Tokenizer fallback ----------------------------------------------------
# Some CPU hosts cannot see the full GPU-side model directory (e.g. only
# part of the Lustre tree is mounted). If the primary TOKENIZER path is
# not a readable directory, transparently fall back to a small ~22 MB
# tokenizer-only snapshot that we ship inside the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ ! -d "${TOKENIZER}" ] || [ ! -f "${TOKENIZER}/tokenizer.json" ]; then
    REPO_ASSET_TAR="${REPO_ROOT}/distributed/assets/qwen35_9b_tokenizer.tar.gz"
    LOCAL_TOK_DIR="${HOME}/.cache/swebench-tokenizer-Qwen3.5-9B"
    if [ -f "${REPO_ASSET_TAR}" ]; then
        if [ ! -f "${LOCAL_TOK_DIR}/tokenizer.json" ]; then
            echo "[run_inference.sh] Primary tokenizer path not readable:" >&2
            echo "[run_inference.sh]   '${TOKENIZER}'" >&2
            echo "[run_inference.sh] Extracting repo-shipped tokenizer to" >&2
            echo "[run_inference.sh]   '${LOCAL_TOK_DIR}'" >&2
            mkdir -p "${LOCAL_TOK_DIR}"
            tar -xzf "${REPO_ASSET_TAR}" -C "${LOCAL_TOK_DIR}" --strip-components=1
        fi
        TOKENIZER="${LOCAL_TOK_DIR}"
    else
        echo "[run_inference.sh] WARNING: tokenizer path '${TOKENIZER}' is" >&2
        echo "[run_inference.sh] not a directory and no repo fallback was" >&2
        echo "[run_inference.sh] found at ${REPO_ASSET_TAR}." >&2
        echo "[run_inference.sh] Run the GPU-side helper to generate one:" >&2
        echo "[run_inference.sh]   bash distributed/package_tokenizer.sh \\" >&2
        echo "[run_inference.sh]        <gpu_path_to_model_dir> \\" >&2
        echo "[run_inference.sh]        distributed/assets/qwen35_9b_tokenizer.tar.gz" >&2
        echo "[run_inference.sh] Or pass --allow_heuristic_tokenizer to this script." >&2
    fi
fi
# ---------------------------------------------------------------------------


python -m distributed.run_api_remote \
    --dataset_name_or_path ${DATASET_NAME_OR_PATH} --split ${SPLIT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --base_url ${BASE_URL} --output_dir ${OUTPUT_DIR} \
    --tokenizer ${TOKENIZER} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --concurrency ${CONCURRENCY} \
    --chat