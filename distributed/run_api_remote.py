#!/usr/bin/env python3
"""
Remote (distributed) inference client for SWE-bench.

This script is the "CPU-side" counterpart of ``distributed.serve_model``.
Instead of loading a model locally (as ``swebench.inference.run_llama`` does)
or calling OpenAI/Anthropic (as ``swebench.inference.run_api`` does), it hits
any **OpenAI-compatible** HTTP endpoint (vLLM, TGI, llama.cpp server, ...)
running on the GPU machine, and writes a predictions JSONL file that
``swebench.harness.run_evaluation`` can consume directly.

Feature parity with the upstream inference scripts
--------------------------------------------------
* **Resume & sharding** — mirrors ``run_api.py``: already-processed
  ``instance_id``s are skipped, and ``--shard_id / --num_shards`` partition
  the dataset.
* **Length filter** — instances whose prompt exceeds the server's
  ``max_model_len`` (queried from ``/v1/models``) are dropped, analogous
  to the ``MODEL_LIMITS`` filter in ``run_api.py``. You can further
  restrict this with ``--max_context_tokens``.
* **Dynamic output budget** — each request's ``max_tokens`` is set to
  ``min(--max_new_tokens, server_max - prompt_tokens - --context_safety_margin)``.
  This means the model can generate up to that point and is only cut off
  by its own EOS / stop-string or by hitting the hard context limit —
  no more "prompt + output > max_model_len" 400s from vLLM. Pass
  ``--max_new_tokens 0`` to "let the model decide" and use every leftover
  token in the window.
* **Generation defaults** — ``temperature=0.0``, ``top_p=1.0``, and
  ``max_new_tokens=200`` match ``run_llama.py``.
* **Stop-on-repeat** — ``run_llama.py`` stops generation when the last
  ``min_length`` tokens contain fewer than ``min_tokens`` unique tokens. We
  approximate this on the client by:
    - passing extra stop sequences to the server when we can, and
    - post-processing the returned text and truncating at the first
      detected repetition window of length ``--repeat_stop_window`` with
      fewer than ``--repeat_stop_unique`` unique characters.
* **Output format** — each written line is
  ``{"instance_id", "model_name_or_path", "model_patch", "full_output"}``,
  exactly what ``run_evaluation`` expects.

Minimal example
---------------
    python -m distributed.run_api_remote \\
        --dataset_name_or_path princeton-nlp/SWE-bench_Lite_oracle \\
        --split test \\
        --model_name_or_path Qwen3.5-9B \\
        --base_url http://<gpu-host>:8000/v1 \\
        --api_key EMPTY \\
        --output_dir ./outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset, load_from_disk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm

# We inline ``extract_diff`` instead of importing it from
# ``swebench.inference.make_datasets.utils`` to avoid pulling the full
# ``swebench/__init__.py`` (which imports docker / ghapi / etc.) just to parse
# a string. The implementation below is byte-for-byte identical to
# https://github.com/swe-bench/SWE-bench/blob/main/swebench/inference/make_datasets/utils.py
def extract_diff(response: str | None) -> str | None:
    """Extract the diff from a response formatted in different ways."""
    if response is None:
        return None
    diff_matches: list[str] = []
    other_matches: list[str] = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_api_remote")

try:
    import openai
except ImportError as e:  # pragma: no cover - guidance for operator
    raise SystemExit(
        "The 'openai' python package is required. Install it with "
        "'pip install -r distributed/requirements-cpu.txt'."
    ) from e


# --------------------------------------------------------------------------- #
# Dataclass + CLI                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class GenerationConfig:
    """Knobs that match ``run_llama.py`` / ``run_api.py`` defaults."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 200
    stop: tuple[str, ...] = ()
    repeat_stop_window: int = 100
    repeat_stop_unique: int = 10
    request_timeout: float = 600.0
    chat: bool = False
    context_safety_margin: int = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help=(
            "HuggingFace dataset name or local ``save_to_disk`` path. The "
            "dataset must contain a 'text' column (e.g. *_oracle or "
            "*_bm25_13K variants)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to run on.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "Identifier sent to the server as ``model``. Must match the "
            "``--served-model-name`` used when starting the GPU server. "
            "This string is also written to predictions.jsonl."
        ),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help=(
            "Base URL of the OpenAI-compatible server, e.g. "
            "'http://gpu-host:8000/v1'. Falls back to the env var "
            "``SWEBENCH_REMOTE_BASE_URL`` if not set."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "Bearer token. vLLM accepts any non-empty string; defaults to "
            "env var ``OPENAI_API_KEY`` or 'EMPTY'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the predictions JSONL will be written.",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help=(
            "Upper bound on tokens generated per instance (matches "
            "run_llama.py default of 200). Whatever you pass is further "
            "clipped to the remaining window "
            "(max_model_len - prompt_tokens - context_safety_margin) so "
            "vLLM never rejects a request. Pass 0 to disable the upper "
            "bound and let the model run until EOS or until the context "
            "window is exhausted."
        ),
    )
    parser.add_argument(
        "--context_safety_margin",
        type=int,
        default=8,
        help=(
            "Tokens reserved at the tail of the context window when "
            "computing the dynamic max_tokens (covers tokenizer "
            "off-by-one, chat-template overhead, etc.)."
        ),
    )
    parser.add_argument(
        "--server_max_model_len",
        type=int,
        default=None,
        help=(
            "Override the server's max context length. Normally the "
            "client queries /v1/models and uses whatever vLLM reports. "
            "Use this only if auto-detection fails."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Path or HF id of the tokenizer used to count prompt tokens. "
            "Defaults to --model_name_or_path. If neither is a real "
            "tokenizer repo/path, a 'chars/4' heuristic is used instead."
        ),
    )
    parser.add_argument(
        "--stop",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of stop strings forwarded to the server.",
    )
    parser.add_argument(
        "--repeat_stop_window",
        type=int,
        default=100,
        help="Window size used for the stop-on-repeat post-filter.",
    )
    parser.add_argument(
        "--repeat_stop_unique",
        type=int,
        default=10,
        help=(
            "Minimum number of unique characters a window must contain; "
            "below this the tail is truncated (mirrors run_llama.py)."
        ),
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout, in seconds.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Use /v1/chat/completions instead of /v1/completions. Recommended "
            "for Instruct models so the server applies its chat template."
        ),
    )

    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help=(
            "Upper bound on the total context window the client will use. "
            "Defaults to whatever the server reports; set this to a smaller "
            "value to artificially tighten the budget (e.g. test at 16k "
            "even though vLLM serves 64k). Prompts that don't fit are "
            "skipped instead of triggering a server-side 400."
        ),
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard index to process (0-based). Used with --num_shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many instances (debugging).",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="*",
        default=None,
        help="If set, only run these instance IDs.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def resolve_client(base_url: str | None, api_key: str | None, timeout: float) -> openai.OpenAI:
    base_url = base_url or os.environ.get("SWEBENCH_REMOTE_BASE_URL")
    if not base_url:
        raise SystemExit(
            "No --base_url provided and SWEBENCH_REMOTE_BASE_URL is unset. "
            "Set one to the address of the GPU server (e.g. "
            "'http://gpu-host:8000/v1')."
        )
    key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    logger.info("Connecting to %s", base_url)
    return openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout)


def query_server_max_model_len(client: openai.OpenAI, model: str) -> int | None:
    """Ask the server for ``max_model_len`` of the requested model.

    vLLM's OpenAI-compatible ``/v1/models`` response includes ``max_model_len``
    on each entry (this is exactly what vLLM uses internally for the
    ``prompt + max_tokens > max_model_len`` validator). Returns None if the
    field is absent (e.g. talking to a non-vLLM server).
    """
    try:
        resp = client.models.list()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not list models on the server: %s", exc)
        return None
    for entry in getattr(resp, "data", []) or []:
        if getattr(entry, "id", None) != model:
            continue
        # Pydantic models expose unknown fields through model_extra in v2.
        extra = getattr(entry, "model_extra", None) or {}
        for key in ("max_model_len", "max_context_length", "context_length"):
            if key in extra and isinstance(extra[key], int):
                return extra[key]
        # Older openai-python shapes: entry may expose attrs directly.
        for key in ("max_model_len", "max_context_length", "context_length"):
            val = getattr(entry, key, None)
            if isinstance(val, int):
                return val
    return None


class PromptCounter:
    """Counts tokens for a prompt, with a chars/4 fallback.

    We try, in order:
      1. ``transformers.AutoTokenizer`` loaded from ``preferred`` then any
         fallback candidate (model path, HF id, ...).
      2. ``tiktoken`` (only really useful if the user points at a
         GPT-shaped tokenizer).
      3. ``len(text) // 4`` as a last resort (logs a warning once).
    """

    def __init__(self, candidates: Iterable[str]) -> None:
        self._tokenizer = None
        self._source = None
        tried: list[str] = []
        for cand in candidates:
            if not cand:
                continue
            tried.append(cand)
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._tokenizer = AutoTokenizer.from_pretrained(
                    cand, trust_remote_code=True
                )
                self._source = cand
                logger.info("Counting prompt tokens with tokenizer '%s'.", cand)
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("Tokenizer '%s' not usable: %s", cand, exc)
        logger.warning(
            "No usable tokenizer found (tried %s); falling back to "
            "len(text)//4 for prompt-token accounting. This is only a "
            "rough estimate; consider passing --tokenizer.",
            ", ".join(tried) or "<none>",
        )

    def __call__(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None


def output_file_for(args: argparse.Namespace) -> Path:
    ds = args.dataset_name_or_path
    ds_nick = Path(ds).name if Path(ds).exists() else ds.replace("/", "__")
    model_nick = args.model_name_or_path.replace("/", "__")
    stem = f"{model_nick}__{ds_nick}__{args.split}"
    if args.shard_id is not None and args.num_shards is not None:
        stem += f"__shard-{args.shard_id}-of-{args.num_shards}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.jsonl"


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            try:
                ids.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info("Found %d previously completed instance(s) in %s", len(ids), path)
    return ids


def load_dataset_generic(name_or_path: str, split: str):
    if Path(name_or_path).exists():
        ds = load_from_disk(name_or_path)
    else:
        ds = load_dataset(name_or_path)
    if split not in ds:
        raise ValueError(
            f"Split '{split}' not found in dataset {name_or_path} "
            f"(available: {list(ds.keys())})."
        )
    return ds[split]


# --------------------------------------------------------------------------- #
# Stop-on-repeat approximation (mirrors RepeatingTokensCriteria)              #
# --------------------------------------------------------------------------- #


def truncate_on_repeat(text: str, window: int, min_unique: int) -> str:
    """Approximate ``RepeatingTokensCriteria`` from ``run_llama.py``.

    run_llama.py stops generation the moment the last ``window`` *tokens*
    contain fewer than ``min_unique`` unique tokens. We don't know the
    tokenization server-side, so we apply the same heuristic on characters
    instead; it's noticeably conservative (windows of 100 chars ≈ 25-30
    tokens) but it still reliably catches the pathological "aaaaaa…" /
    "</s></s>…" tails that the upstream criterion targets.
    """
    if not text or window <= 0 or min_unique <= 0:
        return text
    if len(text) <= window:
        return text
    for end in range(window, len(text) + 1):
        chunk = text[end - window : end]
        if len(set(chunk)) < min_unique:
            # keep everything before the degenerate window
            return text[: end - window]
    return text


# --------------------------------------------------------------------------- #
# Remote call                                                                 #
# --------------------------------------------------------------------------- #


_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def resolve_max_tokens(
    cfg: GenerationConfig,
    prompt_tokens: int,
    server_max_len: int | None,
) -> int | None:
    """Decide how many tokens we can still safely request.

    Returns the number of output tokens to allow, or ``None`` if the
    prompt is already too big to leave any room (caller should skip).

    When ``server_max_len`` is unknown we just honour ``--max_new_tokens``
    (or 200 if the user asked for "unbounded" generation).
    """
    if server_max_len is None:
        if cfg.max_new_tokens > 0:
            return cfg.max_new_tokens
        # No server budget info and user asked for "unbounded" -> pick a
        # sensible default so we don't send ``max_tokens=None`` which
        # would also disable server-side limits implicitly.
        return 200

    remaining = server_max_len - prompt_tokens - cfg.context_safety_margin
    if remaining <= 0:
        return None
    if cfg.max_new_tokens > 0:
        return min(cfg.max_new_tokens, remaining)
    return remaining


@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _remote_generate(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    cfg: GenerationConfig,
    max_tokens: int,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": max_tokens,
    }
    if cfg.stop:
        kwargs["stop"] = list(cfg.stop)

    if cfg.chat:
        # The "system" vs "user" split mimics what run_api.py does for OpenAI
        # when a prompt starts with a system line. Fall back to a single user
        # turn when the prompt does not have a clear split.
        if "\n" in prompt:
            system_msg, user_msg = prompt.split("\n", 1)
        else:
            system_msg, user_msg = "", prompt
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        response = client.chat.completions.create(messages=messages, **kwargs)
        return response.choices[0].message.content or ""
    else:
        response = client.completions.create(prompt=prompt, **kwargs)
        return response.choices[0].text or ""


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #


def iter_filtered_dataset(
    dataset: Iterable[dict[str, Any]],
    args: argparse.Namespace,
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    items = list(dataset)

    if args.instance_ids:
        wanted = set(args.instance_ids)
        items = [row for row in items if row["instance_id"] in wanted]

    items = [row for row in items if row["instance_id"] not in existing_ids]

    if "text" not in (items[0] if items else {}):
        raise ValueError(
            "The dataset must contain a 'text' column with the pre-built prompt. "
            "Either use a pre-built dataset (e.g. princeton-nlp/SWE-bench_Lite_oracle) "
            "or run swebench.inference.make_datasets.create_text_dataset first."
        )

    # Sort short-to-long so that if we die halfway through we still cover the
    # easy cases — this is what run_api.py does.
    lens = np.asarray([len(row["text"]) for row in items])
    order = np.argsort(lens)
    items = [items[int(i)] for i in order]

    if args.shard_id is not None and args.num_shards is not None:
        # deterministic contiguous sharding, same semantics as datasets.shard
        shard = args.shard_id
        total = args.num_shards
        items = [row for idx, row in enumerate(items) if idx % total == shard]

    if args.limit is not None:
        items = items[: args.limit]

    return items


def run(args: argparse.Namespace) -> int:
    cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stop=tuple(args.stop) if args.stop else (),
        repeat_stop_window=args.repeat_stop_window,
        repeat_stop_unique=args.repeat_stop_unique,
        request_timeout=args.request_timeout,
        chat=args.chat,
        context_safety_margin=args.context_safety_margin,
    )
    client = resolve_client(args.base_url, args.api_key, cfg.request_timeout)

    # Ask the server (or the user) for the hard context ceiling.
    server_max_len = args.server_max_model_len
    if server_max_len is None:
        server_max_len = query_server_max_model_len(client, args.model_name_or_path)
    if server_max_len is not None:
        logger.info("Server reports max_model_len=%d for %s.",
                    server_max_len, args.model_name_or_path)
    else:
        logger.warning(
            "Server did not report max_model_len for %s; dynamic budget "
            "disabled. Pass --server_max_model_len to re-enable it.",
            args.model_name_or_path,
        )

    # Allow the user to artificially tighten the cap (useful for testing
    # against a server configured with a bigger window than you want to use).
    effective_max_len = server_max_len
    if args.max_context_tokens is not None:
        effective_max_len = (
            args.max_context_tokens
            if effective_max_len is None
            else min(effective_max_len, args.max_context_tokens)
        )
        logger.info(
            "Using --max_context_tokens=%d as the effective budget cap.",
            args.max_context_tokens,
        )

    counter = PromptCounter(
        candidates=[args.tokenizer, args.model_name_or_path],
    )

    output_path = output_file_for(args)
    existing_ids = load_existing_ids(output_path)
    logger.info("Writing predictions to %s", output_path)

    dataset = load_dataset_generic(args.dataset_name_or_path, args.split)
    items = iter_filtered_dataset(dataset, args, existing_ids)
    logger.info("Will process %d instance(s).", len(items))

    if not items:
        logger.info("Nothing to do.")
        return 0

    consecutive_failures = 0
    skipped_too_long = 0
    with output_path.open("a", encoding="utf-8") as fout:
        for row in tqdm(items, desc=f"remote inference ({args.model_name_or_path})"):
            instance_id = row["instance_id"]
            prompt = row["text"]

            prompt_tokens = counter(prompt)
            max_tokens = resolve_max_tokens(cfg, prompt_tokens, effective_max_len)
            if max_tokens is None:
                skipped_too_long += 1
                logger.warning(
                    "[%s] prompt uses %d tokens which leaves no room in the "
                    "%d-token window after a %d-token safety margin; skipping.",
                    instance_id,
                    prompt_tokens,
                    effective_max_len or 0,
                    cfg.context_safety_margin,
                )
                continue

            start = time.time()
            try:
                raw = _remote_generate(
                    client,
                    args.model_name_or_path,
                    prompt,
                    cfg,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                consecutive_failures += 1
                logger.error("Generation failed for %s: %s", instance_id, exc)
                if consecutive_failures >= 5:
                    logger.error("Aborting after 5 consecutive failures.")
                    return 1
                continue
            consecutive_failures = 0
            trimmed = truncate_on_repeat(
                raw,
                window=cfg.repeat_stop_window,
                min_unique=cfg.repeat_stop_unique,
            )
            patch = extract_diff(trimmed)
            payload = {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name_or_path,
                "full_output": trimmed,
                "model_patch": patch,
            }
            fout.write(json.dumps(payload) + "\n")
            fout.flush()
            logger.info(
                "[%s] prompt=%d tok, budget=%d tok, gen=%d chars in %.2fs; patch_bytes=%d",
                instance_id,
                prompt_tokens,
                max_tokens,
                len(trimmed),
                time.time() - start,
                len(patch or ""),
            )

    if skipped_too_long:
        logger.info(
            "Skipped %d instance(s) whose prompt exceeded the context window.",
            skipped_too_long,
        )
    logger.info("Done. Output: %s", output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
