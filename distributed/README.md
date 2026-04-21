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

Launch the server(s). The shipped wrapper launches **two independent
vLLM processes** — one tp=4 replica on GPUs 0–3 listening on `:8000`
and another tp=4 replica on GPUs 4–7 listening on `:8001`:

```bash
bash scripts/serve_model.sh
```

Or call the Python entry-point directly per replica (for example, to
pick a different port layout):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m distributed.serve_model \
    --model_path /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    --served_model_name Qwen3.5-9B \
    --host 0.0.0.0 --port 8000 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 262144 \
    --tensor_parallel_size 4 \
    --data_parallel_size 1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m distributed.serve_model \
    --model_path /mgfs/shared/Group_GY/wenchao/shhh/models/Qwen3.5-9B \
    --served_model_name Qwen3.5-9B \
    --host 0.0.0.0 --port 8001 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 262144 \
    --tensor_parallel_size 4 \
    --data_parallel_size 1 &
wait
```

> **Pick `--max_model_len` large enough for the prompts you'll send.** It
> is the *total* window — prompt + output — that vLLM enforces, and it
> rejects any request where `prompt_tokens + max_tokens > max_model_len`.
> SWE-bench_oracle prompts regularly run 20k–35k tokens with a long tail
> that goes well past 100k. **The wrapper script ships with
> two tp=4 replicas at `max_model_len=262144`** so Qwen3.5-9B on
> 8×H100 fits the model's native 262 144-token window (recovers ~7 % of
> instances that would otherwise be skipped at 65k). If you prefer lower
> per-request latency and don't need the huge context, drop to **eight
> tp=1 replicas** at `max_model_len=65536` (one replica per GPU,
> throughput-max but ~7% of SWE-bench_oracle skipped).

### Choosing the parallelism strategy

`distributed/serve_model.py` exposes the following vLLM knobs, and each
**replica** is launched as a separate process:

| Flag | What it does | When to use it |
| --- | --- | --- |
| `--tensor_parallel_size N` (`-tp`) | Shards a *single* model across `N` GPUs. Requires `num_key_value_heads % N == 0`. | Only when one model copy does not fit on one GPU, or when you want the larger KV-cache budget that per-GPU sharding buys. Lowers per-request latency but adds cross-GPU comms per token. |
| `--data_parallel_size N` (`-dp`) | Runs `N` replicas of the model inside a single server, sharing one API port. | **Avoid for non-MoE models like Qwen3.5-9B.** See the warning below — use per-replica processes and client-side load-balancing instead. |
| `--pipeline_parallel_size N` (`-pp`) | Splits layers across `N` GPUs. | Very large models only. Leave at 1 at the 9B scale. |

**Important — why we launch two processes instead of one server with
`-dp 2`.** vLLM's internal DP load-balancer in the 0.19 series reports
per-replica load to the dispatcher via a coordinator that only refreshes
every ~100 ms. For Qwen3.5-9B (non-MoE) on H100s, most requests complete
well inside that window, so the dispatcher sees every replica as "empty"
and repeatedly picks rank 0 — the result is **one replica pinned at
~97 % and the other idle at ~0 %** (nvidia-smi-confirmed, matches
vllm-project/vllm#39384). The upstream docs recommend *external*
load-balancing for non-MoE deployments, which is exactly what we do:
`scripts/serve_model.sh` launches one fully independent `vllm serve`
per replica (no DP flags) and the Python client alternates between them
with a least-in-flight scheduler.

Total GPUs consumed = `tp × (# replicas)`. The wrapper refuses to launch
if that exceeds the box's GPU count.

For `Qwen3.5-9B` (≈18 GB bf16 — fits comfortably on any H100 with room for
KV cache) on an 8×H100 node, the two interesting points in the design
space are **8 × tp=1 replicas** (throughput-max, but the 65 GB-per-GPU
budget caps `max_model_len` at ~65k) and **2 × tp=4 replicas** (shards
each replica over 4 GPUs, which quarters the weight + KV-per-seq cost
and lets `max_model_len=262144` fit alongside 2 replicas). The wrapper
script defaults to the second because it's the only configuration that
covers Qwen3.5-9B's **native** 262 144-token context on 8 H100s.

#### Layouts you might deploy

vLLM requires `num_key_value_heads % tensor_parallel_size == 0`. Qwen3.5-9B
has `num_key_value_heads=4`, so only **tp ∈ {1, 2, 4}** is legal. On your
8-GPU box that gives three practical layouts:

| `tp` | # replicas | Ports | Max practical `max_model_len` | What it's good for |
| --- | --- | --- | --- | --- |
| `1` | `8` | 8000–8007 | ~65k | Throughput-max. 8 replicas run in parallel but each one must hold the full weight + KV in a single 80 GB GPU, so context is capped where KV cache + weights first exceed ~65 GB (bf16). Skips ~7 % of SWE-bench_oracle that need bigger prompts. |
| `2` | `4` | 8000–8003 | ~131k | Balanced middle. Weights are halved per GPU; KV per full seq is halved too. Good if you want 4 replicas at double the context of tp=1. |
| `4` | `2` | 8000, 8001 | **262k (native)** | **Shipped default.** Weights + KV are /4 per GPU, which is exactly what lets Qwen3.5-9B's native 262 144-token context fit. Skips ~0.8 % of SWE-bench_oracle (essentially "impossible" instances that exceed even the native window). |
| `8` | `1` | 8000 | — | vLLM refuses: `num_key_value_heads=4` is not divisible by 8. `serve_model.py` catches this up front with a helpful error listing the legal tps. |

Whichever layout you pick, pass every replica's `/v1` base URL to the
client as a comma-separated list (see `--base_url` below) and it will
balance requests across all of them.

`distributed/serve_model.py` pre-reads the model's `config.json` and will
exit with a friendly error (not a vLLM traceback) if you request an
illegal `tp`, if `--max_model_len` exceeds the model's native window, or
if the chosen layout exceeds `CUDA_VISIBLE_DEVICES`. It also prints a
warning if the chosen `--max_model_len` implies a KV cache so large *per
GPU* (>25 GiB/sequence after tp-sharding) that concurrency will
collapse. With the shipped `tp=4` setting that figure is ~8 GiB/GPU/seq
at 262k, well below the warning threshold.

Sanity-check both servers from the GPU host:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

Make sure the `host:port` is reachable from the CPU host (`telnet <gpu-ip> 8000`).
If you use an SSH tunnel the examples below will still work — just point the
client at `localhost:<tunnelled-port>` on the CPU machine.

For a one-shot end-to-end check from the CPU host once the server is up:

```bash
python -m distributed.check_connection \
    --base_url http://<gpu-host-or-tunnel>:8000/v1 \
    --model Qwen3.5-9B
# and the second replica:
python -m distributed.check_connection \
    --base_url http://<gpu-host-or-tunnel>:8001/v1 \
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

### 2.1 (Sometimes required) Build the prompt dataset once

`run_api_remote.py` — like the upstream `run_api.py` — expects each row of
the input dataset to already contain a pre-built `text` column. The
HuggingFace datasets that end in `_oracle` / `_bm25_13K` / `_bm25_27K` /
etc. already have that column, so if you point the client at e.g.
`princeton-nlp/SWE-bench_Lite_oracle` it just works and you can skip
this step.

The **base** datasets (the ones the evaluation harness consumes) do
**not** have a `text` column — they only carry the ground truth
(`patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, …). This
includes the popular curated subset
[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
for which there is no `_oracle` variant published on the Hub. For
those you have to generate a prompt dataset locally with the official
helper:

```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
    --splits test \
    --output_dir ./datasets \
    --prompt_style style-3 \
    --file_source oracle \
    --validation_ratio 0
```

That writes a HuggingFace `save_to_disk` directory at

```
./datasets/SWE-bench_Verified__style-3__fs-oracle/
```

which is the path you hand to `run_api_remote.py` via
`--dataset_name_or_path`. There's a wrapper script at
[`scripts/make_verified_oracle.sh`](../scripts/make_verified_oracle.sh)
that does the above for you with sensible defaults; point
`SOURCE_DATASET` at any base SWE-bench variant:

```bash
bash scripts/make_verified_oracle.sh                             # Verified
SOURCE_DATASET=princeton-nlp/SWE-bench_Lite \
    bash scripts/make_verified_oracle.sh                         # Lite
SOURCE_DATASET=princeton-nlp/SWE-bench \
    bash scripts/make_verified_oracle.sh                         # full 2.2k
```

Two things to keep straight for Verified specifically:

1. **Inference** uses the *generated* on-disk path
   (`./datasets/SWE-bench_Verified__style-3__fs-oracle`) because that's
   where the `text` column lives.
2. **Evaluation** uses the *original* Verified dataset name
   (`princeton-nlp/SWE-bench_Verified`) because the harness reads repo,
   base commit, and `FAIL_TO_PASS` / `PASS_TO_PASS` from there.
   `scripts/run_evaluation.sh` accepts the two as separate environment
   variables (`INFERENCE_DATASET` and `DATASET_NAME`) precisely because
   they must differ on Verified — see §2.3.

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
    --base_url http://<gpu-host-or-tunnel>:8000/v1,http://<gpu-host-or-tunnel>:8001/v1 \
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
- `--base_url` – OpenAI v1 base URL of the GPU server. **Accepts a
  comma-separated list** when you run more than one replica (recommended
  for non-MoE models, see §1). The client opens one connection pool per
  URL and dispatches each request to the backend with the lowest
  in-flight count, breaking ties with round-robin so it alternates
  `A→B→A→B…` instead of serializing on one replica. On startup we
  probe `max_model_len` on *every* backend and refuse to run if they
  disagree.
- `--api_key` – any non-empty string (vLLM accepts `EMPTY`). The same
  key is sent to every backend.
- `--concurrency` – maximum number of *total* in-flight generation
  requests across all backends (default `8`). The client splits this
  budget evenly: with `8` concurrency and `2` backends, each replica
  sees up to 4 concurrent requests, which is enough to keep vLLM's
  continuous batching fed without waiting for KV cache churn. Rule of
  thumb: `replicas × 4` is a safe ceiling on H100s. `1` reproduces the
  old sequential behaviour; going much higher than `replicas × 8` just
  spends client memory buffering requests vLLM will queue internally.
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
- `--verbose_skips` – restore per-instance `WARNING` messages for prompts
  that don't fit the context window. Off by default (those skips are a
  fixed property of the dataset × server `max_model_len` and would
  otherwise flood the log with ~150 lines). The running count is always
  visible on the tqdm postfix as `skipped=N`, and a single summary line
  with the total + percentage is printed at the end of the run.

### 2.2.1 Scaling throughput

The client is async and maintains up to `--concurrency` in-flight
requests, split evenly across every URL you pass to `--base_url`. If
your GPU server runs with 2 tp=4 replicas (the shipped default), the
benchmark below shows the speedup you should see out of the box:

| Client `--concurrency` | Wall-clock for 16 instances (Qwen3.5-9B, `--max_new_tokens 200`) | Throughput |
| ---: | ---: | ---: |
| 1 | 23.5 s (1.47 s/it) | 0.68 req/s |
| 8 | 3.7 s (4.33 it/s) | 4.33 req/s (**6.4×**) |

Scaling past `replicas × 4` yields diminishing returns (vLLM's own
continuous batching inside a replica is already very efficient) and
eventually *hurts* throughput once KV-cache pressure forces evictions.
The tqdm progress bar shows `per-backend=[a,b,…]` live: if those
numbers track each other closely (e.g. `[3,3]` or `[4,4]`) the
dispatch is healthy; if one number stays at `0` while another stays
pinned at the per-backend budget, a backend is down or your tunnel is
only pointed at one replica.

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
- **"prompt uses N tokens which leaves no room in the M-token window …
  skipping."** – this is *not* an error, just the client filtering out
  instances whose oracle-retrieved prompt is larger than the server's
  `max_model_len`. Expect roughly **7 %** of SWE-bench_oracle at 65k
  and **3 %** at 131k (measured empirically). To recover them either
  (a) bump `--max_model_len` on the GPU (doubles the per-token KV cache,
  so monitor `nvidia-smi`), (b) switch to a tighter retrieval variant
  such as `SWE-bench_bm25_13K`, or (c) accept the loss — the harness
  still scores the run, it just can't credit the skipped instances.
  Use `--verbose_skips` to see which `instance_id`s were filtered.
- **The model keeps spitting out junk after the patch** – the client
  applies the same stop-on-repeating-token criterion as `run_llama.py`
  (via `--repeat_stop_window` / `--repeat_stop_unique`). Tune those
  values or tighten `--max_new_tokens`.
- **OOM on the GPU** – reduce `--gpu_memory_utilization`, lower
  `--max_model_len`, or shard across GPUs with `--tensor_parallel_size N`.
- **One replica pinned at ~97 %, other(s) idle at ~0 %** – this is the
  symptom of vLLM's internal DP load balancer getting stuck on rank 0
  for non-MoE models on fast GPUs (vllm-project/vllm#39384). The fix is
  already the shipped default: run each replica as its own `vllm serve`
  process (no `--data-parallel-size` flag) and let the client do
  HTTP-layer least-in-flight routing by passing both URLs to
  `--base_url`. If you're seeing this, double-check that
  `scripts/serve_model.sh` actually launched *two* processes (look for
  two listening ports: `ss -tlnp | grep -E '800[01]'`) and that the
  client's `--base_url` has both URLs comma-separated, not just one.
- **Client hits "Backends disagree on max_model_len"** – one replica
  was launched with a different `--max_model_len`. Re-check
  `scripts/serve_model.sh` is the same on all processes, or pass
  `--server_max_model_len N` to the client to override (N will be
  applied to every backend; requests larger than that are skipped).

