#!/usr/bin/env bash
set -euo pipefail

# scripts/run_inference_anthropic.sh
# --------------------------------------------------------------------------
# Claude API counterpart to scripts/run_inference.sh.
#
# We reuse the exact same Python client (distributed/run_api_remote.py)
# that talks to our vLLM server, because Anthropic officially exposes an
# OpenAI-compatible endpoint at https://api.anthropic.com/v1/ and our
# client is already an OpenAI-compatible chat client. That gives us, for
# free:
#   - async concurrency + least-in-flight backend selection
#   - prompt-token pre-counting + dynamic max_tokens
#   - resume-on-restart based on output file contents
#   - repeat-tail truncation post-filter
#   - the same predictions.jsonl schema run_evaluation.sh already eats
#
# Expected usage (on any CPU box with network access to api.anthropic.com;
# Docker is NOT required for inference, only for evaluation):
#
#     export ANTHROPIC_API_KEY=sk-ant-...
#     bash scripts/run_inference_anthropic.sh                # 10-row smoke
#     LIMIT=0 bash scripts/run_inference_anthropic.sh        # full dataset
#     MODEL_API_ID=claude-opus-4-7 LIMIT=50 bash ...         # swap models
#
# IMPORTANT -- cost. SWE-bench_oracle prompts are large (~15-30k input
# tokens each). At Sonnet 4.6 prices ($3/M input, $15/M output) a full
# 2.2k-row run can cost on the order of $150. This script therefore
# *defaults* to LIMIT=10 (roughly $0.50) and will refuse to run an
# unbounded job unless you set LIMIT=0 *and* ACK_COST=1. If you are on
# Opus (5x the price) budget accordingly.
# --------------------------------------------------------------------------

# --- Lustre safety -------------------------------------------------------
# Same reasoning as the GPU-path script: shared filesystem + two
# machines means stale .pyc in a shared __pycache__ can produce
# confusing action-at-a-distance bugs.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HOME}/.cache/swebench-pycache-${HOSTNAME:-$(hostname)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
find "${REPO_ROOT}/distributed" -type d -name __pycache__ -prune -print0 2>/dev/null \
    | xargs -0 -r rm -rf 2>/dev/null || true
# --------------------------------------------------------------------------

# --- Dataset --------------------------------------------------------------
DATASET_NAME_OR_PATH="${DATASET_NAME_OR_PATH:-princeton-nlp/SWE-bench_oracle}"
SPLIT="${SPLIT:-test}"

# --- Model ---------------------------------------------------------------
# MODEL_API_ID is what goes on the wire as the `model` field. Claude
# API IDs are published at
#   https://docs.claude.com/en/docs/about-claude/models
# Current (April 2026) stable IDs worth using from this script:
#
#   claude-sonnet-4-6   (default; $3 / $15 per M tok, 200k ctx, 64k out)
#   claude-opus-4-7     ($15 / $75 per M tok, 200k ctx, 128k out)
#   claude-haiku-4-5-20251001  (cheapest, $1 / $5 per M tok)
#
# MODEL_NAME_OR_PATH is the string we record in predictions.jsonl and
# that forms the filename stem. We keep it equal to MODEL_API_ID so the
# artifacts are self-describing when you have several runs side-by-side.
MODEL_API_ID="${MODEL_API_ID:-claude-sonnet-4-6}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_API_ID}}"

# --- Endpoint / auth ------------------------------------------------------
# Single backend: Anthropic's OpenAI-compat layer. No load balancing
# needed (Anthropic handles it server-side).
BASE_URL="${BASE_URL:-https://api.anthropic.com/v1/}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[run_inference_anthropic.sh] ERROR: ANTHROPIC_API_KEY is unset." >&2
    echo "[run_inference_anthropic.sh] Get one at https://console.anthropic.com/" >&2
    echo "[run_inference_anthropic.sh] and do: export ANTHROPIC_API_KEY=sk-ant-..." >&2
    exit 1
fi
API_KEY="${ANTHROPIC_API_KEY}"

# --- Output --------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs_anthropic}"

# --- Tokenizer ----------------------------------------------------------
# Claude uses its own BPE and does not ship tokenizer files publicly.
# We don't have an accurate counter for Claude, but all we use the
# counter for here is to pre-filter "prompt is already too long" and
# to back off max_tokens; a moderately wrong counter is fine as long
# as it over-counts (we just lose a few free output tokens in those
# cases, no correctness issue).
#
# Using the Qwen3.5 tokenizer we already have on disk tends to
# over-count Claude by 5-10 % on English+code, which is the safe
# direction. Fall back to the small repo-shipped snapshot if the
# full model dir is not readable on this host (same logic as
# scripts/run_inference.sh).
TOKENIZER="${TOKENIZER:-/mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B}"
if [ ! -d "${TOKENIZER}" ] || [ ! -f "${TOKENIZER}/tokenizer.json" ]; then
    REPO_ASSET_TAR="${REPO_ROOT}/distributed/assets/qwen35_9b_tokenizer.tar.gz"
    LOCAL_TOK_DIR="${HOME}/.cache/swebench-tokenizer-Qwen3.5-9B"
    if [ -f "${REPO_ASSET_TAR}" ]; then
        if [ ! -f "${LOCAL_TOK_DIR}/tokenizer.json" ]; then
            echo "[run_inference_anthropic.sh] Primary tokenizer not readable," \
                 "extracting repo snapshot to ${LOCAL_TOK_DIR}" >&2
            mkdir -p "${LOCAL_TOK_DIR}"
            tar -xzf "${REPO_ASSET_TAR}" -C "${LOCAL_TOK_DIR}" --strip-components=1
        fi
        TOKENIZER="${LOCAL_TOK_DIR}"
    else
        echo "[run_inference_anthropic.sh] WARNING: no tokenizer available." >&2
        echo "[run_inference_anthropic.sh] Pass --allow_heuristic_tokenizer by" >&2
        echo "[run_inference_anthropic.sh] setting ALLOW_HEURISTIC_TOKENIZER=1." >&2
    fi
fi

ALLOW_HEURISTIC_TOKENIZER="${ALLOW_HEURISTIC_TOKENIZER:-0}"

# --- Context window ------------------------------------------------------
# Anthropic does not expose max_model_len via /v1/models, so we tell
# the client explicitly. 200k is the standard Claude 4 context; Sonnet
# 4.6 has a 1M-token beta that we deliberately do NOT default to (it
# requires an extra beta header and has different pricing). If you
# want to use it, set SERVER_MAX_MODEL_LEN=1000000 and make sure your
# account has access.
SERVER_MAX_MODEL_LEN="${SERVER_MAX_MODEL_LEN:-200000}"

# Safety margin for the dynamic max_tokens computation. For vLLM+Qwen
# we used 32 because client and server tokenize identically. For
# Claude the client tokenizer is not the server tokenizer, so a
# per-request delta of a few hundred tokens is possible on large
# prompts. Bumping the margin to 512 keeps us safely below the
# 200k ceiling at a cost of 512 "lost" output tokens per request --
# negligible compared to the ~4k typical generation we'd want here.
CONTEXT_SAFETY_MARGIN="${CONTEXT_SAFETY_MARGIN:-512}"

# --- Generation budget --------------------------------------------------
# Unlike the vLLM path we do NOT want to default to "unbounded" here:
# every output token costs real money, and Claude 4 models can easily
# produce tens of thousands of tokens of "thinking" before giving an
# answer if you let them. 4096 is a generous budget for SWE-bench
# patch generation (actual patches are typically 50-500 tokens) while
# still capping worst-case per-request cost at a predictable number.
#
# Rough cost-per-request bound at max_tokens=4096:
#   Sonnet 4.6: 20k input * $3/M + 4k output * $15/M ~= $0.12
#   Opus 4.7:   20k input * $15/M + 4k output * $75/M ~= $0.60
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

# --- Concurrency --------------------------------------------------------
# Anthropic rate limits by tier. Tier 1 (default for new keys) is
# ~50 RPM for Sonnet and ~50 RPM for Opus; higher tiers scale up.
# Concurrency of 4 stays well under Tier 1 for typical SWE-bench
# response latencies (5-30 s per request) and can comfortably sustain
# ~200-400 requests/minute on Tier 3+. If you hit 429s, lower this.
CONCURRENCY="${CONCURRENCY:-4}"

# --- Request timeout -----------------------------------------------------
# Claude responses can take a while on long-context inputs (20-60 s
# is common). Give each request 10 min before we give up.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

# --- Cost guard ----------------------------------------------------------
# LIMIT=0 means "process the entire dataset". For SWE-bench that is
# ~2200 rows and can easily be $150+ per run. Require a deliberate
# acknowledgement before we let that happen.
LIMIT="${LIMIT:-10}"
ACK_COST="${ACK_COST:-0}"
if [ "${LIMIT}" = "0" ] && [ "${ACK_COST}" != "1" ]; then
    echo "[run_inference_anthropic.sh] ERROR: LIMIT=0 runs the full dataset." >&2
    echo "[run_inference_anthropic.sh] On SWE-bench_oracle that is ~2200 rows" >&2
    echo "[run_inference_anthropic.sh] and can cost \$100-\$300 depending on model." >&2
    echo "[run_inference_anthropic.sh] If you meant it, re-run with:" >&2
    echo "[run_inference_anthropic.sh]   LIMIT=0 ACK_COST=1 bash \$0" >&2
    exit 1
fi

# --- Summary printout ---------------------------------------------------
echo "[run_inference_anthropic.sh] dataset       : ${DATASET_NAME_OR_PATH} (${SPLIT})"
echo "[run_inference_anthropic.sh] model_api_id  : ${MODEL_API_ID}"
echo "[run_inference_anthropic.sh] base_url      : ${BASE_URL}"
echo "[run_inference_anthropic.sh] output_dir    : ${OUTPUT_DIR}"
echo "[run_inference_anthropic.sh] tokenizer     : ${TOKENIZER}"
echo "[run_inference_anthropic.sh] max_model_len : ${SERVER_MAX_MODEL_LEN}"
echo "[run_inference_anthropic.sh] max_new_tokens: ${MAX_NEW_TOKENS}"
echo "[run_inference_anthropic.sh] concurrency   : ${CONCURRENCY}"
echo "[run_inference_anthropic.sh] safety_margin : ${CONTEXT_SAFETY_MARGIN}"
if [ "${LIMIT}" = "0" ]; then
    echo "[run_inference_anthropic.sh] limit         : (unbounded, full dataset)"
else
    echo "[run_inference_anthropic.sh] limit         : ${LIMIT} row(s)"
fi
echo

# --- Build the command ---------------------------------------------------
cd "${REPO_ROOT}"

cmd=(
    python -m distributed.run_api_remote
        --dataset_name_or_path "${DATASET_NAME_OR_PATH}"
        --split "${SPLIT}"
        --model_name_or_path "${MODEL_NAME_OR_PATH}"
        --base_url "${BASE_URL}"
        --api_key "${API_KEY}"
        --output_dir "${OUTPUT_DIR}"
        --tokenizer "${TOKENIZER}"
        --server_max_model_len "${SERVER_MAX_MODEL_LEN}"
        --context_safety_margin "${CONTEXT_SAFETY_MARGIN}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --concurrency "${CONCURRENCY}"
        --request_timeout "${REQUEST_TIMEOUT}"
        --chat
)

if [ "${LIMIT}" != "0" ]; then
    cmd+=(--limit "${LIMIT}")
fi

if [ "${ALLOW_HEURISTIC_TOKENIZER}" = "1" ]; then
    cmd+=(--allow_heuristic_tokenizer)
fi

"${cmd[@]}"

# --- Hand-off to evaluation ---------------------------------------------
model_nick="${MODEL_NAME_OR_PATH//\//__}"
ds_nick="${DATASET_NAME_OR_PATH//\//__}"
predictions_path="${OUTPUT_DIR}/${model_nick}__${ds_nick}__${SPLIT}.jsonl"
echo
echo "[run_inference_anthropic.sh] Predictions written to:"
echo "[run_inference_anthropic.sh]   ${predictions_path}"
echo "[run_inference_anthropic.sh] To evaluate (on the CPU/Docker box):"
echo "[run_inference_anthropic.sh]   OUTPUT_DIR=${OUTPUT_DIR} \\"
echo "[run_inference_anthropic.sh]   MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} \\"
echo "[run_inference_anthropic.sh]   INFERENCE_DATASET=${DATASET_NAME_OR_PATH} \\"
echo "[run_inference_anthropic.sh]   bash scripts/run_evaluation.sh"
