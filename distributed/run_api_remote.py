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
import asyncio
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
    AsyncRetrying,
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
    context_safety_margin: int = 32


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
            "Base URL(s) of the OpenAI-compatible server(s). A single "
            "value targets one backend, e.g. 'http://gpu-host:8000/v1'. "
            "Pass a comma-separated list (e.g. "
            "'http://gpu-host:8000/v1,http://gpu-host:8001/v1') to fan "
            "out across multiple independent replicas; requests are "
            "distributed with least-in-flight routing so both replicas "
            "stay busy without waiting for a stats-update tick. Falls "
            "back to the env var ``SWEBENCH_REMOTE_BASE_URL`` if not set."
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
        default=32,
        help=(
            "Tokens reserved at the tail of the context window when "
            "computing the dynamic max_tokens. Chat-completion requests "
            "already count chat-template tokens on the client side, so "
            "this margin only needs to cover minor tokenizer / server "
            "off-by-one quirks; 32 is conservative."
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
            "Defaults to --model_name_or_path. If this tokenizer can't be "
            "loaded the client aborts — we've seen silent 'chars/4' "
            "fallbacks lead to a 100% 400 rate against vLLM because the "
            "crude estimate is just *below* the real token count, and the "
            "sign error pushes every request one token past max_model_len. "
            "Use --allow_heuristic_tokenizer to opt back into the fallback."
        ),
    )
    parser.add_argument(
        "--allow_heuristic_tokenizer",
        action="store_true",
        help=(
            "If no tokenizer can be loaded, keep running with a "
            "chars/4 (safety-inflated by 25%%) estimator instead of "
            "aborting. Off by default because an under-counting "
            "tokenizer produces a 100%% 400 rate against vLLM."
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
        "--concurrency",
        type=int,
        default=8,
        help=(
            "Maximum number of in-flight generation requests. Sized to "
            "the server's data-parallel replica count (vLLM's "
            "--data_parallel_size); 1 forces sequential behavior. "
            "Increase if you also have room in pipelined scheduling "
            "(vLLM's continuous batching can often absorb 2-4x more) "
            "but beware: each in-flight request reserves its own KV "
            "cache budget, so too high a number starves throughput."
        ),
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
        "--verbose_skips",
        action="store_true",
        help=(
            "By default the client logs one line per *skipped* instance at "
            "DEBUG level (i.e. hidden) and only prints a final total. Pass "
            "this to restore per-instance WARNINGs when diagnosing which "
            "prompts are too long."
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
        help=(
            "Process at most this many instances per invocation. NOTE: "
            "applied AFTER the resume filter (existing instance_ids in the "
            "output file are removed first). So re-running the same command "
            "with --limit N will process the *next* N undone instances, "
            "not re-process the first N. This matches upstream run_api.py."
        ),
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


def parse_base_urls(base_url_arg: str | None) -> list[str]:
    """Parse ``--base_url`` into a list of backend endpoints.

    Accepts either a single URL (``http://h:8000/v1``) or a
    comma-separated list (``http://h:8000/v1,http://h:8001/v1``) and
    also falls back to the ``SWEBENCH_REMOTE_BASE_URL`` env var.

    The returned list preserves input order, is de-duplicated (handy when
    the same endpoint is passed twice accidentally), and strips
    surrounding whitespace on each entry. It never contains an empty
    string; if no URL could be determined we raise a ``SystemExit`` with
    an actionable message rather than silently default to ``None``.
    """
    raw = base_url_arg or os.environ.get("SWEBENCH_REMOTE_BASE_URL") or ""
    items: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        v = chunk.strip().rstrip("/")
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        items.append(v)
    if not items:
        raise SystemExit(
            "No --base_url provided and SWEBENCH_REMOTE_BASE_URL is unset. "
            "Set one (or a comma-separated list) to the address of the GPU "
            "server(s), e.g. 'http://gpu-host:8000/v1' or "
            "'http://gpu-host:8000/v1,http://gpu-host:8001/v1'."
        )
    return items


def resolve_client(
    base_url: str | None,
    api_key: str | None,
    timeout: float,
) -> openai.OpenAI:
    """Build a *synchronous* OpenAI client for the first backend.

    We only use this client for startup probes (``/v1/models`` and the
    tokenizer calibration round-trip). At runtime every worker uses an
    ``AsyncOpenAI`` instance built per-backend in ``_run_workers``.
    """
    urls = parse_base_urls(base_url)
    key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    logger.info(
        "Connecting to %d backend(s): %s",
        len(urls),
        ", ".join(urls),
    )
    return openai.OpenAI(base_url=urls[0], api_key=key, timeout=timeout)


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
    """Counts tokens **the same way the server will count them**.

    The match-the-server detail matters: vLLM rejects any request where
    ``server_prompt_tokens + max_tokens > max_model_len``, and for a
    chat-completions request ``server_prompt_tokens`` is measured *after*
    applying the model's chat template (which adds ``<|im_start|>`` /
    ``<|im_end|>`` markers, role names, and a trailing generation prompt).

    - ``chat=True``  -> count ``apply_chat_template(..., add_generation_prompt=True)``
    - ``chat=False`` -> count raw text the way the completions endpoint sees
      it (``add_special_tokens=True`` matches vLLM's default).

    If no tokenizer is loadable and ``allow_heuristic=True``, a
    *pessimistic* chars/3 heuristic is used. We deliberately over-count
    instead of under-count: an over-count loses a few free output tokens,
    while the chars/4 under-count we used previously produced a 100%
    400-rate against vLLM (prompt_tokens + max_tokens would land exactly
    one token past max_model_len).
    """

    def __init__(
        self,
        candidates: Iterable[str],
        chat: bool,
        allow_heuristic: bool = False,
    ) -> None:
        self._tokenizer = None
        self._source = None
        self._chat = chat
        self._allow_heuristic = allow_heuristic
        tried: list[tuple[str, str]] = []
        for cand in candidates:
            if not cand:
                continue
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as exc:
                tried.append((cand, f"cannot import transformers: {exc}"))
                break
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    cand, trust_remote_code=True
                )
                self._source = cand
                logger.info(
                    "Counting prompt tokens with tokenizer '%s' (chat=%s).",
                    cand,
                    chat,
                )
                return
            except Exception as exc:  # noqa: BLE001
                # Most useful debugging info is the exception *class*; the
                # message is often generic ("Can't load tokenizer ...").
                tried.append((cand, f"{type(exc).__name__}: {exc}"))

        detail = "; ".join(f"{p!r} -> {err}" for p, err in tried) or "<no candidates>"
        msg = (
            "No usable tokenizer could be loaded; tried: " + detail + ". "
            "An accurate tokenizer is required because vLLM rejects any "
            "request where prompt_tokens + max_tokens > max_model_len, and "
            "the chars/4 fallback under-counts by 5-10% which lands every "
            "request one token past the limit. Fixes: (1) ensure "
            "--tokenizer points at a directory containing tokenizer.json "
            "(e.g. the same HF snapshot the GPU server is serving), or "
            "(2) pip install 'transformers>=4.44' 'sentencepiece' on the "
            "CPU node, or (3) pass --allow_heuristic_tokenizer to opt "
            "into the (pessimistic) chars/3 estimator."
        )
        if allow_heuristic:
            logger.warning(msg)
        else:
            raise SystemExit(msg)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for a chat request, after chat-template expansion."""
        if self._tokenizer is None:
            return self._pessimistic_chat_count(messages)
        try:
            ids = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            # Not every tokenizer has a chat template. Add a conservative
            # estimate of template overhead (~16 tokens) on top of the raw
            # content count so we still don't under-estimate.
            logger.debug(
                "apply_chat_template failed (%s); estimating overhead.", exc
            )
            raw = sum(
                len(self._tokenizer.encode(m.get("content") or "",
                                           add_special_tokens=False))
                for m in messages
            )
            return raw + 16

    def count_text(self, text: str) -> int:
        """Count tokens for a completions request."""
        if self._tokenizer is None:
            return self._pessimistic_text_count(text)
        # Match vLLM's default for /v1/completions, which does add special
        # tokens unless the client explicitly disables them.
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    @staticmethod
    def _pessimistic_text_count(text: str) -> int:
        # Subword tokenizers for English code hover around 3.3–3.8 chars
        # per token; using /3 deliberately over-counts. Add a fixed
        # headroom to cover chat-template markers even in completions
        # mode (some providers still inject system/user wrappers).
        return max(1, len(text) // 3 + 32)

    @classmethod
    def _pessimistic_chat_count(cls, messages: list[dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content") or "") for m in messages)
        # Per-message overhead covers <|im_start|>role\n and <|im_end|>\n
        # tokens that every chat-template emits (~4 tokens per role).
        return max(1, total_chars // 3 + 16 + 4 * len(messages))

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


def _server_tokenize(
    client: openai.OpenAI,
    model: str,
    chat: bool,
    text_or_messages: Any,
) -> int | None:
    """Ask the vLLM server to tokenize a payload. Returns the exact token
    count or ``None`` if the server doesn't expose ``/tokenize``.

    Uses the OpenAI client's transport so it inherits timeouts, retries,
    and auth headers without a second HTTP client dependency.
    """
    body: dict[str, Any] = {"model": model}
    if chat:
        body["messages"] = text_or_messages
        body["add_generation_prompt"] = True
    else:
        body["prompt"] = text_or_messages
    try:
        # The openai python client exposes a raw POST via
        # ``client.with_raw_response.post`` only in newer versions; for
        # portability we just use the underlying httpx client.
        base = str(client.base_url).rstrip("/")
        # /tokenize lives at the server root, not under /v1 — vLLM's
        # OpenAI bridge registers it outside the versioned namespace.
        base_root = base.rsplit("/v1", 1)[0] if base.endswith("/v1") else base
        url = base_root + "/tokenize"
        resp = client._client.post(url, json=body, timeout=30.0)  # type: ignore[attr-defined]
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "count" in data and isinstance(data["count"], int):
            return data["count"]
        if "tokens" in data and isinstance(data["tokens"], list):
            return len(data["tokens"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("server /tokenize probe failed: %s", exc)
    return None


def _calibrate_counter_or_die(
    counter: PromptCounter,
    client: openai.OpenAI,
    model: str,
    chat: bool,
    safety_margin: int,
) -> None:
    """Compare local token count to the server's, abort on mismatch.

    Sends a tiny, realistic probe through both paths and verifies the
    client's count is *at least as large as* the server's. The client
    counting ``>=`` the server is fine (we'll just waste a few output
    tokens); the client counting ``<`` the server is fatal (request will
    be rejected). We only tolerate a ``safety_margin / 2`` shortfall to
    leave a tiny cushion for tokenizer revision skew.
    """
    probe_msgs = [
        {"role": "system", "content": "You are a careful code assistant."},
        {"role": "user",
         "content": "def add(a: int, b: int) -> int:\n    return a + b\n\n"
                    "Write a docstring for the function above."},
    ]
    probe_text = probe_msgs[-1]["content"]

    server = _server_tokenize(
        client, model, chat, probe_msgs if chat else probe_text,
    )
    if server is None:
        # Server doesn't expose /tokenize (non-vLLM endpoint); we can't
        # calibrate. Accept the risk silently — it's an exact-tokenizer
        # deployment assumption breaking, not our bug.
        logger.info(
            "Server has no /tokenize endpoint; skipping counter calibration."
        )
        return

    local = counter.count_messages(probe_msgs) if chat else counter.count_text(probe_text)
    delta = server - local  # positive means we undercounted
    logger.info(
        "Tokenizer calibration: local=%d, server=%d, delta=%+d (margin=%d).",
        local, server, delta, safety_margin,
    )
    if delta > safety_margin // 2:
        raise SystemExit(
            f"Client tokenizer under-counts the server by {delta} tokens on "
            f"a short probe (local={local}, server={server}). This will "
            f"cause vLLM to reject most requests with 400 "
            f"(prompt_tokens + max_tokens > max_model_len). "
            f"Fix: pass --tokenizer pointing at the exact HF snapshot the "
            f"GPU server is serving (it must contain tokenizer.json), or "
            f"increase --context_safety_margin to at least {(delta + 16) * 2} "
            f"so the budget absorbs the mismatch."
        )


def build_messages(prompt: str, row: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Build the OpenAI ``messages`` payload for a chat-completions request.

    We deliberately do **not** try to split ``prompt`` into system/user
    halves on the first newline (the old behaviour of ``run_api.py`` for
    OpenAI); that split is brittle and can corrupt multi-line system
    prompts. Upstream ``run_llama.py`` doesn't split either — it just
    feeds the whole formatted prompt to the model. We do the same, with
    one exception: if the dataset row explicitly carries a
    ``system_prompt`` column we honour it.
    """
    messages: list[dict[str, str]] = []
    if row is not None:
        sys_prompt = row.get("system_prompt")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


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
    messages: list[dict[str, str]] | None = None,
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
        msgs = messages if messages is not None else build_messages(prompt)
        response = client.chat.completions.create(messages=msgs, **kwargs)
        return response.choices[0].message.content or ""
    else:
        response = client.completions.create(prompt=prompt, **kwargs)
        return response.choices[0].text or ""


async def _remote_generate_async(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    cfg: GenerationConfig,
    max_tokens: int,
    messages: list[dict[str, str]] | None = None,
) -> str:
    """Async counterpart of :func:`_remote_generate`.

    Tenacity's async retry loop requires ``AsyncRetrying`` rather than the
    decorator; we can't put the retry decorator on an ``async def`` and
    keep compatibility with the rest of the stack.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": max_tokens,
    }
    if cfg.stop:
        kwargs["stop"] = list(cfg.stop)
    msgs = messages if messages is not None else build_messages(prompt) if cfg.chat else None

    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    ):
        with attempt:
            if cfg.chat:
                response = await client.chat.completions.create(messages=msgs, **kwargs)
                return response.choices[0].message.content or ""
            response = await client.completions.create(prompt=prompt, **kwargs)
            return response.choices[0].text or ""
    # ``AsyncRetrying(..., reraise=True)`` guarantees we never fall through
    # without either returning or raising, but type-checkers don't know
    # that. The sentinel below is unreachable at runtime.
    raise RuntimeError("unreachable: AsyncRetrying exited without result")


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

    # Only validate the schema when there is work left to do — otherwise
    # an empty resume run (every instance already present) would falsely
    # complain about a missing 'text' column.
    if items and "text" not in items[0]:
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
    base_urls = parse_base_urls(args.base_url)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    logger.info(
        "Connecting to %d backend(s): %s",
        len(base_urls),
        ", ".join(base_urls),
    )

    # Probe each backend independently for max_model_len. We refuse to
    # continue if different backends disagree, because the dynamic
    # budget logic assumes a single ceiling applies to every request:
    # if one replica is configured at 64k and another at 256k, a prompt
    # sized to the larger would 400 on the smaller.
    server_max_len = args.server_max_model_len
    reported: dict[str, int | None] = {}
    if server_max_len is None:
        for url in base_urls:
            probe = openai.OpenAI(
                base_url=url, api_key=api_key, timeout=cfg.request_timeout
            )
            try:
                reported[url] = query_server_max_model_len(
                    probe, args.model_name_or_path
                )
            finally:
                probe.close()
        distinct = {v for v in reported.values() if v is not None}
        if len(distinct) > 1:
            raise SystemExit(
                "Backends disagree on max_model_len: "
                + ", ".join(f"{k}={v}" for k, v in reported.items())
                + ". Reconfigure so every backend serves the same "
                "max_model_len, or pass --server_max_model_len to "
                "override (in which case the client will use that value "
                "for *every* backend)."
            )
        server_max_len = next(iter(distinct), None)
    if server_max_len is not None:
        logger.info(
            "Server(s) report max_model_len=%d for %s.",
            server_max_len,
            args.model_name_or_path,
        )
    else:
        logger.warning(
            "Server(s) did not report max_model_len for %s; dynamic "
            "budget disabled. Pass --server_max_model_len to re-enable it.",
            args.model_name_or_path,
        )

    # One synchronous client, only used for the tokenizer calibration
    # below. Workers will construct their own async clients later.
    client = openai.OpenAI(
        base_url=base_urls[0], api_key=api_key, timeout=cfg.request_timeout
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
        chat=cfg.chat,
        allow_heuristic=args.allow_heuristic_tokenizer,
    )

    # Calibrate the local counter against the server's own tokenizer on
    # a synthetic short message. If they disagree by more than
    # ``context_safety_margin / 2`` we refuse to continue, rather than
    # emit hundreds of doomed requests: previous mis-loaded tokenizers
    # underestimated by ~5-10% and the sign error caused a 100% 400 rate.
    _calibrate_counter_or_die(
        counter=counter,
        client=client,
        model=args.model_name_or_path,
        chat=cfg.chat,
        safety_margin=cfg.context_safety_margin,
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

    concurrency = max(1, int(args.concurrency))
    logger.info(
        "Running with concurrency=%d across %d backend(s): %s "
        "(request_timeout=%.0fs).",
        concurrency,
        len(base_urls),
        ", ".join(base_urls),
        cfg.request_timeout,
    )

    skipped_too_long, sent = asyncio.run(
        _run_workers(
            items=items,
            args=args,
            cfg=cfg,
            counter=counter,
            effective_max_len=effective_max_len,
            output_path=output_path,
            concurrency=concurrency,
            base_urls=base_urls,
            api_key=api_key,
        )
    )

    if skipped_too_long:
        total = len(items)
        pct = 100.0 * skipped_too_long / total if total else 0.0
        logger.info(
            "Skipped %d/%d instance(s) (%.1f%%) because their prompt > the "
            "%d-token context window. This is a dataset/model-size mismatch: "
            "raise --max_model_len on the server (at the cost of KV-cache "
            "memory) or switch to a tighter retrieval variant (e.g. "
            "SWE-bench_bm25_13K instead of SWE-bench_oracle) to recover "
            "them. Pass --verbose_skips to see which instance_ids were "
            "skipped.",
            skipped_too_long,
            total,
            pct,
            effective_max_len or 0,
        )
    logger.info("Done. %d prediction(s) written to %s", sent, output_path)
    return 0


async def _run_workers(
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    cfg: GenerationConfig,
    counter: PromptCounter,
    effective_max_len: int | None,
    output_path: Path,
    concurrency: int,
    base_urls: list[str],
    api_key: str,
) -> tuple[int, int]:
    """Process ``items`` concurrently and append predictions to disk.

    Returns ``(skipped_too_long, predictions_written)``.

    Dispatch is done at the HTTP layer: we keep one ``AsyncOpenAI`` per
    backend URL and route each request to the backend with the fewest
    in-flight requests right now. That is "external load balancing" in
    the vLLM sense and is what the upstream DP docs explicitly recommend
    for non-MoE models. It avoids the single-replica stickiness we hit
    when using vLLM's internal DP LB with dp=2 on fast H100 hardware
    (see issue vllm-project/vllm#39384).
    """
    import httpx  # imported locally — httpx is an openai dependency

    # Concurrency budget per backend: split the total concurrency
    # evenly across backends (rounded up so we never starve any single
    # one). Each HTTP pool is sized to that budget * 2 so we have
    # headroom for slightly-racy opens/closes without queueing on
    # sockets.
    n_backends = len(base_urls)
    per_backend_conc = max(1, (concurrency + n_backends - 1) // n_backends)

    class Backend:
        __slots__ = ("url", "client", "http", "inflight")

        def __init__(self, url: str) -> None:
            self.url = url
            self.http = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max(per_backend_conc * 2, 16),
                    max_keepalive_connections=max(per_backend_conc, 8),
                ),
                timeout=cfg.request_timeout,
                transport=httpx.AsyncHTTPTransport(retries=0),
            )
            self.client = openai.AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                http_client=self.http,
            )
            # Number of requests currently in flight against this
            # backend. Incremented when a request is dispatched,
            # decremented once the HTTP reply returns (success or fail).
            self.inflight = 0

    backends: list[Backend] = [Backend(u) for u in base_urls]
    # Round-robin cursor, used only to break ties in in-flight count.
    # Critical on cold start: if we just did ``min(backends, key=...)``
    # when both were at inflight=0, Python's stable min would always
    # return backends[0] and re-create the "stuck on one replica"
    # problem we're trying to fix. Instead, when the smallest-inflight
    # set has more than one backend we advance the cursor and pick
    # whichever of them is next in rotation.
    #
    # Start the cursor at ``len(backends) - 1`` so the first pick sees
    # ``rr_cursor = 0`` after the pre-increment, meaning the very
    # first request lands on backends[0], the second on backends[1],
    # etc. — the smooth A→B→A→B pattern the operator will expect.
    rr_cursor = len(backends) - 1

    def _pick_backend() -> Backend:
        """Return a backend using least-in-flight with round-robin ties.

        Behaviour:
        * If exactly one backend has the smallest ``inflight`` count,
          pick it (least-loaded wins).
        * If several tie, pick the next one after ``rr_cursor``, so
          consecutive "both idle" requests alternate instead of all
          landing on backend 0.

        This gives you the smooth "send to A, then B, then A, ..." the
        user observed is missing, with zero idle time between
        requests and no reliance on a sleepy stats coordinator.
        """
        nonlocal rr_cursor
        min_if = min(b.inflight for b in backends)
        tied = [i for i, b in enumerate(backends) if b.inflight == min_if]
        if len(tied) == 1:
            return backends[tied[0]]
        # Advance the cursor until we find a tied backend.
        n = len(backends)
        for _ in range(n):
            rr_cursor = (rr_cursor + 1) % n
            if rr_cursor in tied:
                return backends[rr_cursor]
        return backends[tied[0]]

    # The writer is a single thread-safe coroutine; workers push payloads
    # onto a queue and the writer drains them in arrival order. This keeps
    # the file format identical (one JSON per line, flushed after each
    # write) regardless of how many workers are running.
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    skipped_too_long = 0
    stats = {
        "sent": 0,
        "failed": 0,
        "consecutive_failures": 0,
        "abort": False,
        "inflight": 0,
    }
    stats_lock = asyncio.Lock()

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(
        total=len(items),
        desc=f"remote inference ({args.model_name_or_path})",
    )

    # --- writer coroutine ---------------------------------------------------
    async def writer() -> None:
        # Open the file inside the coroutine so any exceptions during file
        # open propagate to the main task (asyncio.run will show them).
        def _fopen():
            return output_path.open("a", encoding="utf-8", buffering=1)

        loop = asyncio.get_running_loop()
        fout = await loop.run_in_executor(None, _fopen)
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    return
                line = json.dumps(payload) + "\n"
                await loop.run_in_executor(None, fout.write, line)
                await loop.run_in_executor(None, fout.flush)
        finally:
            await loop.run_in_executor(None, fout.close)

    writer_task = asyncio.create_task(writer(), name="predictions-writer")

    # --- per-instance worker ------------------------------------------------
    async def handle(row: dict[str, Any]) -> None:
        nonlocal skipped_too_long
        if stats["abort"]:
            return
        instance_id = row["instance_id"]
        prompt = row["text"]

        if cfg.chat:
            messages = build_messages(prompt, row)
            prompt_tokens = counter.count_messages(messages)
        else:
            messages = None
            prompt_tokens = counter.count_text(prompt)

        max_tokens = resolve_max_tokens(cfg, prompt_tokens, effective_max_len)
        if max_tokens is None:
            async with stats_lock:
                skipped_too_long += 1
                skipped_now = skipped_too_long
                inflight_now = stats["inflight"]
            # These skips are a fixed property of the dataset vs. the
            # server's max_model_len (~7% for SWE-bench_oracle at 65k); a
            # per-instance WARNING drowns the log without telling the
            # operator anything actionable. Emit at DEBUG by default,
            # restore WARNING via ``--verbose_skips``, and expose the
            # running total on the tqdm postfix so it's still visible.
            logger.log(
                logging.WARNING if args.verbose_skips else logging.DEBUG,
                "[%s] prompt uses %d tokens which leaves no room in the "
                "%d-token window after a %d-token safety margin; skipping.",
                instance_id,
                prompt_tokens,
                effective_max_len or 0,
                cfg.context_safety_margin,
            )
            pbar.update(1)
            pbar.set_postfix_str(
                f"inflight={inflight_now}/{concurrency} skipped={skipped_now}",
                refresh=False,
            )
            return

        if (
            effective_max_len is not None
            and prompt_tokens + max_tokens > effective_max_len
        ):
            logger.error(
                "[%s] BUG: prompt_tokens(%d) + max_tokens(%d) = %d > "
                "effective_max_len(%d). This should never happen; "
                "refusing to send the request.",
                instance_id,
                prompt_tokens,
                max_tokens,
                prompt_tokens + max_tokens,
                effective_max_len,
            )
            pbar.update(1)
            return

        start = time.time()
        async with sem:
            # Pick the target backend *inside* the semaphore so we see
            # up-to-date inflight counts. Updating ``b.inflight`` has
            # to happen atomically with the pick itself; since all of
            # these counters are only ever read/written from the main
            # asyncio event loop (no threads), regular integer arith
            # is safe without a lock.
            async with stats_lock:
                backend = _pick_backend()
                backend.inflight += 1
                stats["inflight"] += 1
            try:
                raw = await _remote_generate_async(
                    backend.client,
                    args.model_name_or_path,
                    prompt,
                    cfg,
                    max_tokens=max_tokens,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001
                async with stats_lock:
                    stats["failed"] += 1
                    stats["consecutive_failures"] += 1
                    stats["inflight"] -= 1
                    backend.inflight -= 1
                    if stats["consecutive_failures"] >= 5:
                        stats["abort"] = True
                logger.error(
                    "Generation failed for %s on %s: %s",
                    instance_id,
                    backend.url,
                    exc,
                )
                pbar.update(1)
                return

        async with stats_lock:
            stats["consecutive_failures"] = 0
            stats["sent"] += 1
            stats["inflight"] -= 1
            backend.inflight -= 1
            inflight_now = stats["inflight"]
            skipped_now = skipped_too_long

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
        await queue.put(payload)
        elapsed = time.time() - start
        pbar.update(1)
        # Per-backend inflight breakdown is the easiest way to see
        # whether the dispatch is actually alternating. With 2
        # backends you should usually see something like "1/1" when
        # busy and "0/0" briefly between waves.
        per_backend = ",".join(str(b.inflight) for b in backends)
        pbar.set_postfix_str(
            f"inflight={inflight_now}/{concurrency} "
            f"per-backend=[{per_backend}] skipped={skipped_now}",
            refresh=False,
        )
        logger.info(
            "[%s] backend=%s prompt=%d tok, budget=%d tok, gen=%d chars "
            "in %.2fs; patch_bytes=%d",
            instance_id,
            backend.url,
            prompt_tokens,
            max_tokens,
            len(trimmed),
            elapsed,
            len(patch or ""),
        )

    tasks = [asyncio.create_task(handle(row), name=f"row-{i}") for i, row in enumerate(items)]
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    finally:
        pbar.close()
        await queue.put(None)
        await writer_task
        for b in backends:
            try:
                await b.client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            try:
                await b.http.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    if stats["abort"]:
        logger.error(
            "Aborted after 5 consecutive failures (sent=%d, failed=%d).",
            stats["sent"],
            stats["failed"],
        )

    return skipped_too_long, stats["sent"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
