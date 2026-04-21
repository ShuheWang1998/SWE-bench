#!/usr/bin/env bash
set -euo pipefail

# scripts/make_verified_oracle.sh
# --------------------------------------------------------------------------
# Turn a base SWE-bench dataset (e.g. princeton-nlp/SWE-bench_Verified) into
# an _oracle-flavored on-disk dataset that our inference clients can consume.
#
# WHY WE NEED THIS
#
# princeton-nlp/SWE-bench_oracle and _bm25_13K etc. ship with a pre-built
# ``text`` column containing the full prompt (problem statement + oracle
# or retrieved files). The "base" datasets (SWE-bench, SWE-bench_Lite,
# SWE-bench_Verified) do NOT have that column — they're the ground-truth
# inputs for the harness, not inference-ready prompts.
#
# Our ``distributed/run_api_remote.py`` and the upstream ``run_api.py``
# both require ``row["text"]`` to exist. So to run a model on Verified
# (or any other base variant) you first have to transform it with
# ``swebench.inference.make_datasets.create_text_dataset`` — that's the
# official helper and it's what this wrapper drives.
#
# USAGE
#
#     bash scripts/make_verified_oracle.sh
#
# or with overrides:
#
#     SOURCE_DATASET=princeton-nlp/SWE-bench_Lite \
#     SPLITS="test" \
#     OUTPUT_DIR=./datasets \
#     bash scripts/make_verified_oracle.sh
#
# The output is a HuggingFace ``save_to_disk`` directory (binary Arrow
# format, not JSONL). Feed its path directly to run_api_remote.py / the
# run_inference*.sh scripts via --dataset_name_or_path.
# --------------------------------------------------------------------------

# --- Lustre safety --------------------------------------------------------
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${HOME}/.cache/swebench-pycache-${HOSTNAME:-$(hostname)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# --------------------------------------------------------------------------

# --- Configuration --------------------------------------------------------
# Which base dataset to transform. The defaults target SWE-bench_Verified
# (the curated 500-instance subset of SWE-bench with extra human review);
# for a smoke run switch to SWE-bench_Lite (300 instances) to save time.
SOURCE_DATASET="${SOURCE_DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLITS="${SPLITS:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/datasets}"

# Prompt template used inside the ``text`` column. ``style-3`` is the
# one upstream run_api.py uses for OpenAI/Anthropic models — it includes
# the problem statement, hints, and full-file contents of every file
# touched by the gold patch ("oracle" retrieval). The alternatives
# (style-1, style-2) predate run_api and aren't worth using here.
PROMPT_STYLE="${PROMPT_STYLE:-style-3}"

# "oracle" uses the set of files actually touched by the gold patch as
# in-context retrieval -- same setup as princeton-nlp/SWE-bench_oracle.
# Alternatives: "bm25" (requires a retrieval file) or "all" (every file
# in the repo, which blows past any context window). Stick with oracle
# unless you're experimenting with retrieval.
FILE_SOURCE="${FILE_SOURCE:-oracle}"

# Optional truncation. If you're running a 32k-context model you can
# cap prompt length to 32000 tokens here (with tokenizer_name=cl100k
# for OpenAI, llama for llama-family, etc.). Leave unset for full-size
# prompts.
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
# --------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

echo "[make_verified_oracle.sh] source       : ${SOURCE_DATASET}"
echo "[make_verified_oracle.sh] splits       : ${SPLITS}"
echo "[make_verified_oracle.sh] output_dir   : ${OUTPUT_DIR}"
echo "[make_verified_oracle.sh] prompt_style : ${PROMPT_STYLE}"
echo "[make_verified_oracle.sh] file_source  : ${FILE_SOURCE}"
if [ -n "${MAX_CONTEXT_LEN}" ]; then
    echo "[make_verified_oracle.sh] truncation   : ${MAX_CONTEXT_LEN} tok (${TOKENIZER_NAME:-<unset>})"
fi
echo

cd "${REPO_ROOT}"

cmd=(
    python -m swebench.inference.make_datasets.create_text_dataset
        --dataset_name_or_path "${SOURCE_DATASET}"
        --splits ${SPLITS}
        --output_dir "${OUTPUT_DIR}"
        --prompt_style "${PROMPT_STYLE}"
        --file_source "${FILE_SOURCE}"
        --validation_ratio 0
)

if [ -n "${MAX_CONTEXT_LEN}" ]; then
    cmd+=(--max_context_len "${MAX_CONTEXT_LEN}")
    if [ -n "${TOKENIZER_NAME}" ]; then
        cmd+=(--tokenizer_name "${TOKENIZER_NAME}")
    else
        echo "[make_verified_oracle.sh] WARNING: MAX_CONTEXT_LEN is set but" \
             "TOKENIZER_NAME is not; the helper requires one of" \
             "{llama, cl100k} to measure prompt length." >&2
    fi
fi

"${cmd[@]}"

# --- Report the output path -----------------------------------------------
# create_text_dataset.construct_output_filename builds a deterministic
# stem from the arguments, so we can reconstruct it here and tell the
# user exactly what to feed into run_inference.sh.
source_basename="$(basename "${SOURCE_DATASET}")"
stem="${source_basename}__${PROMPT_STYLE}__fs-${FILE_SOURCE}"
if [ -n "${MAX_CONTEXT_LEN}" ]; then
    stem="${stem}__mcc-${MAX_CONTEXT_LEN}-${TOKENIZER_NAME:-unknown}"
fi
output_path="${OUTPUT_DIR}/${stem}"

echo
if [ -d "${output_path}" ]; then
    echo "[make_verified_oracle.sh] Done. On-disk dataset saved at:"
    echo "[make_verified_oracle.sh]   ${output_path}"
    echo
    echo "[make_verified_oracle.sh] To run inference, set in run_inference.sh"
    echo "[make_verified_oracle.sh] (or run_inference_anthropic.sh):"
    echo "[make_verified_oracle.sh]   DATASET_NAME_OR_PATH=${output_path}"
    echo
    echo "[make_verified_oracle.sh] For evaluation, remember to use the"
    echo "[make_verified_oracle.sh] BASE dataset (not this _oracle form):"
    echo "[make_verified_oracle.sh]   DATASET_NAME=${SOURCE_DATASET} \\"
    echo "[make_verified_oracle.sh]   INFERENCE_DATASET=${output_path} \\"
    echo "[make_verified_oracle.sh]   bash scripts/run_evaluation.sh"
else
    # Not found at the deterministic path — fall back to listing OUTPUT_DIR
    # so the user can still see what landed.
    echo "[make_verified_oracle.sh] Done. Contents of ${OUTPUT_DIR}:"
    ls -1 "${OUTPUT_DIR}"
fi
