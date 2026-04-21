#!/usr/bin/env python3
"""
Launch an OpenAI-compatible inference server for a SWE-bench tested model
on the GPU machine.

The server defaults mirror ``swebench.inference.run_llama.generate``:
    * greedy by default (``temperature=0``)
    * stop-on-repeating-tokens
    * ``max_new_tokens = 200``

These values are applied on the *client* side (see
``distributed.run_api_remote``); the server just hosts the model. We use vLLM
because it speaks the OpenAI HTTP protocol out of the box and supports
``Qwen3_5ForConditionalGeneration`` (the architecture used by
``Qwen3.5-9B``).

Example (single GPU)
--------------------
    python -m distributed.serve_model \\
        --model_path /mgfs/shared/.../Qwen3.5-9B \\
        --served_model_name Qwen3.5-9B \\
        --host 0.0.0.0 --port 8000 \\
        --tensor_parallel_size 1 \\
        --gpu_memory_utilization 0.9 \\
        --max_model_len 32768

Example (use all 8 GPUs on a box; best throughput for a 9B model)
-----------------------------------------------------------------
    python -m distributed.serve_model \\
        --model_path /mgfs/shared/.../Qwen3.5-9B \\
        --served_model_name Qwen3.5-9B \\
        --host 0.0.0.0 --port 8000 \\
        --data_parallel_size 8 \\
        --gpu_memory_utilization 0.9 \\
        --max_model_len 32768
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local directory or HF repo id of the model to serve.",
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default=None,
        help=(
            "Name that clients must pass as ``model`` in OpenAI requests. "
            "Defaults to the last component of --model_path."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Interface to bind on (use 0.0.0.0 so the CPU machine can reach it).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help=(
            "Shard one model copy across this many GPUs. vLLM requires "
            "num_key_value_heads %% tp == 0, so for Qwen3.5-9B (kv_heads=4) "
            "only tp in {1, 2, 4} is legal; tp=8 will be rejected. Use tp "
            "only when one replica does not fit on a single GPU or when "
            "per-request latency matters more than throughput. For SWE-bench "
            "(many independent prompts) tp=1 + data_parallel_size=8 "
            "produces higher throughput than any tp>1 configuration."
        ),
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help=(
            "Run this many full replicas of the model within a single server "
            "process (vLLM load-balances across them). Total GPUs used = "
            "tensor_parallel_size * data_parallel_size, and must not exceed "
            "the number of visible GPUs. For Qwen3.5-9B on 8xH100, "
            "--data_parallel_size 8 is the right choice."
        ),
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help=(
            "Split model layers across this many GPUs (pipeline parallelism). "
            "Rarely useful at the 9B scale; leave at 1 unless you know you "
            "need it. Total GPUs used = tp * pp * dp."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU vLLM may use (0 < x <= 1).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help=(
            "Maximum context length served (total = prompt + output tokens). "
            "Leave empty to use the model's native maximum. SWE-bench oracle "
            "prompts regularly run 20k-35k tokens; 32768 is *not* enough once "
            "you add generation budget. 65536 is a safe default for "
            "Qwen3.5-9B (native 262144). Lower this only if you need to "
            "shrink the KV cache."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Weight dtype; run_llama.py uses bfloat16.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Forwarded to vLLM / HF loader (needed for some custom model files).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "Optional bearer token to protect the endpoint. "
            "If set, clients must send 'Authorization: Bearer <key>'."
        ),
    )
    parser.add_argument(
        "--extra_vllm_args",
        type=str,
        default="",
        help=(
            "Raw string appended verbatim to the ``vllm serve`` command. Use "
            "for flags this wrapper does not expose, e.g. "
            "'--quantization awq --enable-prefix-caching'."
        ),
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Print the resolved command and exit without starting the server.",
    )
    return parser.parse_args(argv)


def _visible_gpu_count() -> int | None:
    """Number of GPUs CUDA will expose, honoring ``CUDA_VISIBLE_DEVICES``."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _read_model_config(model_path: str) -> dict | None:
    """Best-effort read of ``config.json`` to enable friendly pre-flight checks.

    We don't want to hard-depend on transformers here, so we just parse the
    JSON directly. Returns the (possibly nested) text-model config dict, or
    ``None`` if the file is missing / unreadable.
    """
    try:
        path = Path(model_path) / "config.json"
        if not path.is_file():
            return None
        import json as _json

        data = _json.loads(path.read_text())
        return data.get("text_config") or data
    except Exception:
        return None


def _kv_gib_per_token(text_cfg: dict) -> float | None:
    """KV-cache bytes per token for a GQA/MHA transformer in bf16."""
    try:
        layers = int(text_cfg["num_hidden_layers"])
        kv = int(text_cfg.get("num_key_value_heads") or text_cfg["num_attention_heads"])
        head_dim = int(
            text_cfg.get("head_dim")
            or text_cfg["hidden_size"] // text_cfg["num_attention_heads"]
        )
    except (KeyError, ZeroDivisionError, ValueError, TypeError):
        return None
    # 2x for K and V, 2 bytes per bf16 element.
    per_layer_per_token = 2 * kv * head_dim * 2
    return (layers * per_layer_per_token) / (1024**3)


def _preflight_checks(args: argparse.Namespace, tp: int, dp: int, pp: int) -> None:
    """Catch configs that vLLM would otherwise reject with a noisy traceback."""
    cfg = _read_model_config(str(Path(args.model_path).expanduser()))
    if cfg is None:
        return

    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    if isinstance(kv_heads, int) and tp > 1 and kv_heads % tp != 0:
        legal = sorted(d for d in range(1, kv_heads + 1) if kv_heads % d == 0)
        raise SystemExit(
            f"--tensor_parallel_size={tp} is not valid for this model: "
            f"num_key_value_heads={kv_heads} is not divisible by {tp}. "
            f"Legal values of tp for this model are {legal}. "
            f"Hint: for 8 GPUs on a {kv_heads}-kv-head model, the typical "
            f"layouts are tp=1*dp=8 (throughput-optimal for short contexts), "
            f"tp=2*dp=4 (balanced), or tp={max(legal)}*dp={8 // max(legal)} "
            f"(required for very large max_model_len)."
        )

    model_max = cfg.get("max_position_embeddings")
    if (
        isinstance(model_max, int)
        and args.max_model_len is not None
        and args.max_model_len > model_max
    ):
        raise SystemExit(
            f"--max_model_len={args.max_model_len} exceeds the model's "
            f"max_position_embeddings={model_max}."
        )

    gib_per_tok = _kv_gib_per_token(cfg)
    if gib_per_tok is not None and args.max_model_len is not None:
        per_seq_gib = gib_per_tok * args.max_model_len
        # tp shards the KV cache across the tp group, so per-GPU KV is
        # (full KV / tp). Warn based on what each GPU actually pays, not
        # the aggregate across the replica -- otherwise e.g. tp=4 at 256k
        # looks scary (32 GiB "per replica") while each GPU only carries
        # 8 GiB, which is fine on an 80 GB H100.
        per_gpu_gib = per_seq_gib / max(1, tp)
        if per_gpu_gib > 25:
            print(
                f"[serve_model] warning: max_model_len={args.max_model_len} "
                f"implies a KV cache of ~{per_gpu_gib:.1f} GiB per sequence "
                f"per GPU (bf16, tp={tp}; full-replica KV = {per_seq_gib:.1f} "
                f"GiB). On an 80 GB GPU you will only fit a handful of "
                f"concurrent sequences per replica, which hurts throughput. "
                f"Consider raising tp, lowering --max_model_len, or "
                f"switching to a tighter retrieval variant.",
                file=sys.stderr,
            )


def build_command(args: argparse.Namespace) -> list[str]:
    model_path = str(Path(args.model_path).expanduser())
    served_name = args.served_model_name or Path(model_path).name

    tp = max(1, args.tensor_parallel_size)
    dp = max(1, args.data_parallel_size)
    pp = max(1, args.pipeline_parallel_size)
    total_gpus = tp * dp * pp
    visible = _visible_gpu_count()
    if visible is not None and total_gpus > visible:
        raise SystemExit(
            f"Requested tp={tp} * dp={dp} * pp={pp} = {total_gpus} GPUs but "
            f"only {visible} are visible. Lower the parallel sizes or set "
            f"CUDA_VISIBLE_DEVICES to expose more GPUs."
        )

    _preflight_checks(args, tp, dp, pp)

    cmd: list[str] = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        served_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(tp),
        "--data-parallel-size",
        str(dp),
        "--pipeline-parallel-size",
        str(pp),
        "--gpu-memory-utilization",
        f"{args.gpu_memory_utilization}",
        "--dtype",
        args.dtype,
    ]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.extra_vllm_args.strip():
        cmd += shlex.split(args.extra_vllm_args)
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cmd = build_command(args)

    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[serve_model] launching: {printable}", flush=True)

    if args.print_only:
        return 0

    env = os.environ.copy()
    # run_llama.py keeps the KV cache off (use_cache=False); vLLM manages its
    # own cache, so we don't need to mirror that. We do mirror the bfloat16
    # default when --dtype auto is requested via --dtype bfloat16 above.
    try:
        completed = subprocess.run(cmd, env=env, check=False)
        return completed.returncode
    except FileNotFoundError:
        print(
            "[serve_model] error: the 'vllm' CLI was not found on PATH. "
            "Install the GPU-side requirements with "
            "'pip install -r distributed/requirements-gpu.txt'.",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    sys.exit(main())
