# SWE-bench Distributed Setup (GPU ⇄ CPU)

This folder contains everything you need to run SWE-bench across **two
machines** when the GPU host cannot run Docker:

| Machine | Role |
| --- | --- |
| **GPU machine** | Serves the tested model (e.g. `Qwen3.5-9B`) over an OpenAI-compatible HTTP API. No Docker required. |
| **CPU machine** | Runs `swebench.harness.run_evaluation` (Docker). Also runs the inference client that calls the GPU machine to produce `predictions.jsonl`. |

`swebench.harness.run_evaluation` itself **never** calls the model — it only
reads a `predictions.jsonl` file. So the distributed workflow is simply:

```
                    (HTTP, OpenAI-compatible)
┌──────────────┐   ─────────────────────────►   ┌──────────────┐
│ CPU machine  │                                │ GPU machine  │
│              │   ◄─────────────────────────   │              │
│              │          model patches         │              │
│ run_api_     │                                │ vLLM serving │
│ remote.py    │                                │ Qwen3.5-9B   │
│ +            │                                │              │
│ run_evalua-  │                                │              │
│ tion.py      │                                │              │
└──────────────┘                                └──────────────┘
```

---

## 1. GPU machine — start the inference server

Install the GPU-side dependencies (Python 3.10+, CUDA-enabled PyTorch + vLLM):

```bash
pip install -r distributed/requirements-gpu.txt
```

Launch the server (listens on port `8000` by default; bind to
`0.0.0.0` so the CPU machine can connect):

```bash
bash distributed/serve_model.sh \
    /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    0.0.0.0 8000
```

Or call the Python entry-point directly for more control:

```bash
python -m distributed.serve_model \
    --model_path /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    --served_model_name Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 65536 \
    --data_parallel_size 8
```

> **Pick `--max_model_len` large enough for the prompts you'll send.** It
> is the *total* window — prompt + output — that vLLM enforces, and it
> rejects any request where `prompt_tokens + max_tokens > max_model_len`.
> SWE-bench_Lite_oracle prompts regularly run 20k–35k tokens, so 32768 is
> usually too small once you add generation budget. **Use 65536 for
> Qwen3.5-9B on H100s**; the model natively supports up to 262144 and the
> KV-cache hit is modest at DP=8.

### Choosing the parallelism strategy

`distributed/serve_model.py` forwards three knobs to vLLM:

| Flag | What it does | When to use it |
| --- | --- | --- |
| `--tensor_parallel_size N` (`-tp`) | Shards a *single* model across `N` GPUs. Requires `num_key_value_heads % N == 0`. | Only when one model copy does not fit on one GPU. Lowers latency but adds cross-GPU comms per token. |
| `--data_parallel_size N` (`-dp`) | Runs `N` full replicas of the model inside one server, load-balances requests. | **Default choice for SWE-bench.** Batch-of-independent-prompts workload, no inter-GPU sync per token, linear throughput scaling. |
| `--pipeline_parallel_size N` (`-pp`) | Splits layers across `N` GPUs. | Very large models only. Leave at 1 at the 9B scale. |

Total GPUs consumed = `tp × dp × pp`. The wrapper refuses to launch if that
exceeds `CUDA_VISIBLE_DEVICES` (or the system's GPU count).

For `Qwen3.5-9B` (≈18 GB bf16 — fits comfortably on any H100 with room for
KV cache) on an 8×H100 node, **`--data_parallel_size 8`** is the right
setting. `--tensor_parallel_size 8` would additionally fail validation
because the model's `num_key_value_heads=4` is not divisible by 8.

#### Combining tensor parallel with data parallel

vLLM requires `num_key_value_heads % tensor_parallel_size == 0`. Qwen3.5-9B
has `num_key_value_heads=4`, so only **tp ∈ {1, 2, 4}** is legal. On your
8-GPU box that leaves three combinations:

| `tp` | `dp` | Replicas × shards | What it's good for |
| --- | --- | --- | --- |
| `1` | `8` | 8 × 1 | **Recommended for SWE-bench.** Max throughput, no per-token cross-GPU comms. |
| `2` | `4` | 4 × 2 | Lower per-request latency, ~half the throughput of tp=1. Pick if you want to shave wall-clock on individual long generations. |
| `4` | `2` | 2 × 4 | Even lower latency, ~¼ throughput. Rarely the right trade on SWE-bench. |
| `8` | `1` | rejected | vLLM refuses: 4 is not divisible by 8. `serve_model.py` catches this up front. |

`distributed/serve_model.py` pre-reads the model's `config.json` and will
exit with a friendly error (not a vLLM traceback) if you request an
illegal `tp`, if `--max_model_len` exceeds the model's native window, or
if you set `tp * dp * pp` higher than `CUDA_VISIBLE_DEVICES`. It also
prints a warning if the chosen `--max_model_len` implies a KV cache so
large per replica (>25 GiB/sequence) that concurrency will collapse — in
practice that means you should use 65536 for SWE-bench_Lite_oracle on
Qwen3.5-9B, not the native 262144.

Sanity-check the server from the GPU host:

```bash
curl http://localhost:8000/v1/models
```

Make sure the `host:port` is reachable from the CPU host (`telnet <gpu-ip> 8000`).
If you use an SSH tunnel the examples below will still work — just point the
client at `localhost:<tunnelled-port>` on the CPU machine.

For a one-shot end-to-end check from the CPU host once the server is up:

```bash
python -m distributed.check_connection \
    --base_url http://<gpu-host-or-tunnel>:8000/v1 \
    --model Qwen3.5-9B
```

> **Why vLLM?** The project's upstream local-inference script
> (`swebench/inference/run_llama.py`) uses `model.generate(...)` with greedy
> decoding, `max_new_tokens=200`, and early-stops on repeating tokens. vLLM
> exposes an OpenAI-compatible endpoint (`/v1/completions` and
> `/v1/chat/completions`), handles batching automatically, and supports
> `Qwen3_5ForConditionalGeneration`, so it is the simplest drop-in server.

---

## 2. CPU machine — run inference against the GPU machine

Install the CPU-side dependencies (no torch/cuda needed):

```bash
pip install -e .                       # base swebench
pip install -e ".[datasets]"           # for dataset creation (tokenizers, etc.)
pip install -r distributed/requirements-cpu.txt
```

### 2.1 (Optional) Build the prompt dataset once

`run_api_remote.py` — like the upstream `run_api.py` — expects each row of
the input dataset to already contain a pre-built `text` column. If you use
one of the HuggingFace datasets that end in `_oracle` or `_bm25_13K` you
can skip this step. Otherwise:

```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite \
    --splits test \
    --output_dir ./datasets \
    --prompt_style style-3 \
    --file_source oracle
```

### 2.2 Produce `predictions.jsonl` via the remote server

The client mirrors the retry / resume logic of `swebench/inference/run_api.py`
but hits any OpenAI-compatible endpoint (vLLM, llama.cpp server, TGI with
OpenAI adapter, etc.) and matches the generation settings used by
`run_llama.py` (greedy by default, stop-on-repeat, capped output length).

```bash
python -m distributed.run_api_remote \
    --dataset_name_or_path princeton-nlp/SWE-bench_Lite_oracle \
    --split test \
    --model_name_or_path Qwen3.5-9B \
    --base_url http://<gpu-host-or-tunnel>:8000/v1 \
    --api_key EMPTY \
    --output_dir ./outputs \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_new_tokens 200 \
    --concurrency 8 \
    --tokenizer /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B
```

Arguments:

- `--dataset_name_or_path` – same as upstream. Must be a dataset whose rows
  have a pre-computed `"text"` prompt column (e.g. `*_oracle`, `*_bm25_13K`,
  or a local dataset you built with
  `python -m swebench.inference.make_datasets.create_text_dataset`).
- `--model_name_or_path` – the string you want to appear in the predictions
  file and logs. Should match the `--served-model-name` used on the GPU host
  so the request is routed correctly.
- `--base_url` – OpenAI v1 base URL of the GPU server.
- `--api_key` – any non-empty string (vLLM accepts `EMPTY`).
- `--concurrency` – maximum number of in-flight generation requests
  (default `8`). **Set this equal to the GPU server's
  `--data_parallel_size`.** `1` reproduces the old sequential
  behaviour; a value larger than the number of replicas just spends
  client memory buffering requests that vLLM will queue internally.
- `--max_new_tokens` – **upper bound** for completion length (default 200,
  matches `run_llama.py`). The client further clips this to whatever the
  context window can still spare, so the request never exceeds
  `max_model_len` and the model stops naturally on EOS / stop strings.
  Pass **`--max_new_tokens 0`** to say "let the model use everything that's
  left in the context window." Only recommended for debugging — a single
  pathological generation can otherwise pin a replica for minutes.
- `--context_safety_margin` – tokens held back from the hard ceiling when
  computing the dynamic budget (default `32`, covers chat-template
  overhead and tokenizer off-by-one).
- `--tokenizer` – path (or HF id) of the tokenizer used to count prompt
  tokens locally. **Required** for stable runs: the client auto-aborts
  at startup if it can't load this and `--allow_heuristic_tokenizer`
  wasn't passed. The repo ships a 6.8 MB tarball at
  `distributed/assets/qwen35_9b_tokenizer.tar.gz` — `scripts/run_inference.sh`
  extracts it to `~/.cache/swebench-tokenizer-Qwen3.5-9B/` automatically
  when the primary path isn't readable.
- `--allow_heuristic_tokenizer` – opt into a pessimistic `chars/3` fallback
  if no tokenizer can be loaded. Off by default: the fallback over-counts
  by ~20% (wastes output budget) but never under-counts (never causes
  400s), which is the opposite of the silent `chars/4` path it replaces.
- `--max_context_tokens` – optional *user-imposed* cap on the window. Use
  this to test at a tighter budget than the server actually serves;
  prompts that don't fit are skipped instead of triggering a server-side
  `VLLMValidationError`.
- `--server_max_model_len` – escape hatch if auto-detection of the
  server's `max_model_len` fails (e.g. non-vLLM backend that doesn't
  expose it on `/v1/models`).
- `--shard_id`, `--num_shards` – same meaning as upstream; lets you split
  the run across several CPU machines / processes.
- `--chat` – send `/v1/chat/completions` instead of `/v1/completions`.
  Recommended for Qwen3.5 Instruct so that the model's chat template is
  applied; omit for base models or when your dataset already contains the
  full formatted prompt.

### 2.2.1 Scaling throughput

The client is async and maintains up to `--concurrency` in-flight
requests. If your GPU server runs with `--data_parallel_size 8`, the
benchmark below shows the speedup you should see out of the box:

| Client `--concurrency` | Wall-clock for 16 instances (Qwen3.5-9B, `--max_new_tokens 200`) | Throughput |
| ---: | ---: | ---: |
| 1 | 23.5 s (1.47 s/it) | 0.68 req/s |
| 8 | 3.7 s (4.33 it/s) | 4.33 req/s (**6.4×**) |

Scaling past the replica count yields diminishing returns (vLLM's own
continuous batching inside a replica is already very efficient) and
eventually *hurts* throughput once KV-cache pressure forces evictions.
Start at `--concurrency == --data_parallel_size` and only go higher if
the progress bar's `inflight=N/K` indicator is consistently pinned at
`K` and the GPUs aren't saturated.

The output is a standard SWE-bench predictions JSONL with three keys per line:

```json
{"instance_id": "...", "model_name_or_path": "...", "model_patch": "<diff>"}
```

### 2.3 Evaluate with the existing harness

Nothing special here — the patches come from the GPU server but from the
harness's point of view they are just a JSONL file:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ./outputs/Qwen3.5-9B__<dataset>__test.jsonl \
    --max_workers 8 \
    --run_id qwen35-9b-lite
```

---

## 3. Troubleshooting

- **`404` or wrong model name** – vLLM's OpenAI endpoint matches the
  `--served-model-name` flag exactly. Pass the same string to both
  `distributed.serve_model` (GPU) and `distributed.run_api_remote` (CPU).
- **`VLLMValidationError: This model's maximum context length is N tokens.
  However, you requested M output tokens and your prompt contains at
  least P input tokens …`** – the server's `--max_model_len` is too small
  for the prompt you're sending. Either raise it on the GPU host (e.g.
  `--max_model_len 65536` for SWE-bench_Lite_oracle on Qwen3.5-9B) or
  accept losing those instances — the client already skips prompts it
  can tell won't fit, so you only see this if the *server* is configured
  too tightly. The dynamic `max_tokens` logic guarantees the client itself
  never over-requests once `--tokenizer` is set correctly.
- **Timeouts on large contexts** – raise `--request_timeout` on the client
  and `--max-model-len` on the server. The vLLM default may not cover
  the full SWE-bench-oracle prompts (often 20k+ tokens).
- **The model keeps spitting out junk after the patch** – the client
  applies the same stop-on-repeating-token criterion as `run_llama.py`
  (via `--repeat_stop_window` / `--repeat_stop_unique`). Tune those
  values or tighten `--max_new_tokens`.
- **OOM on the GPU** – reduce `--gpu_memory_utilization`, lower
  `--max_model_len`, or shard across GPUs with `--tensor_parallel_size N`.

