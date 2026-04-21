#!/usr/bin/env bash
# Package just the tokenizer files from a HuggingFace model checkpoint.
#
# Why: the full Qwen3.5-9B snapshot is ~36 GB (four 5 GB safetensors
# shards). On distributed setups where the CPU-side inference client
# only needs to *count* prompt tokens (never load weights) it is
# wasteful — and often impossible, on Lustre/NFS mount boundaries —
# to copy the whole model across. The tokenizer-only bundle is ~22 MB
# uncompressed / ~7 MB gzipped.
#
# Usage:
#   bash distributed/package_tokenizer.sh <model_dir> [output_tarball]
#
# Example:
#   bash distributed/package_tokenizer.sh \
#        /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
#        distributed/assets/qwen35_9b_tokenizer.tar.gz
set -euo pipefail

SRC="${1:-}"
DEST_TAR="${2:-distributed/assets/qwen35_9b_tokenizer.tar.gz}"

if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "usage: $0 <model_dir> [output.tar.gz]" >&2
    echo "  <model_dir> must be a readable HF model directory containing" >&2
    echo "  at least tokenizer.json and tokenizer_config.json." >&2
    exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PACK="${STAGE}/qwen35_9b_tokenizer"
mkdir -p "$PACK"

NEEDED_ANY_COPIED=0
for f in tokenizer.json tokenizer_config.json chat_template.jinja \
         config.json vocab.json merges.txt special_tokens_map.json \
         added_tokens.json; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$PACK/"
        NEEDED_ANY_COPIED=1
    fi
done

if [ "$NEEDED_ANY_COPIED" -eq 0 ] || [ ! -f "$PACK/tokenizer.json" ]; then
    echo "ERROR: $SRC is missing tokenizer.json; is this really a HF model?" >&2
    exit 3
fi

mkdir -p "$(dirname "$DEST_TAR")"
tar -czf "$DEST_TAR" -C "$STAGE" "$(basename "$PACK")"

SIZE=$(ls -lh "$DEST_TAR" | awk '{print $5}')
echo "Wrote $SIZE -> $DEST_TAR"
echo "Contents:"
tar -tzf "$DEST_TAR" | sed 's/^/  /'
