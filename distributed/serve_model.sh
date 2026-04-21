#!/usr/bin/env bash
# Thin wrapper around ``python -m distributed.serve_model``.
#
# Usage:
#   bash distributed/serve_model.sh <model_path> [host] [port] [--extra=vllm args...]
#
# Example:
#   bash distributed/serve_model.sh \
#       /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
#       0.0.0.0 8000

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_path> [host] [port] [extra vllm args...]" >&2
    exit 64
fi

MODEL_PATH="$1"; shift
HOST="${1:-0.0.0.0}"; shift || true
PORT="${1:-8000}"; shift || true

SERVED_NAME="$(basename "${MODEL_PATH%/}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

exec python -m distributed.serve_model \
    --model_path "${MODEL_PATH}" \
    --served_model_name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --extra_vllm_args "$*"
