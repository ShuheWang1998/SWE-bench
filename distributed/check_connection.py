#!/usr/bin/env python3
"""
Smoke-test the GPU inference server from the CPU machine.

Usage
-----
    python -m distributed.check_connection \\
        --base_url http://<gpu-host>:8000/v1 \\
        --model Qwen3.5-9B

Exits with code 0 on success, 1 on any failure. Intended to be run once on
the CPU machine before launching the full evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import openai
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install the CPU-side deps first: "
        "'pip install -r distributed/requirements-cpu.txt'."
    ) from e


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base_url",
        default=os.environ.get("SWEBENCH_REMOTE_BASE_URL"),
        required=False,
    )
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True, help="Served model name.")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--prompt",
        default="Write 'OK' and nothing else.",
        help="Test prompt sent to /v1/completions.",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="Hit /v1/chat/completions instead of /v1/completions.",
    )
    args = p.parse_args(argv)

    if not args.base_url:
        print(
            "error: --base_url (or SWEBENCH_REMOTE_BASE_URL) is required",
            file=sys.stderr,
        )
        return 2

    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    print(f"[check] contacting {args.base_url} ...")
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        print(f"[check] server advertises models: {names}")
        if args.model not in names:
            print(
                f"[check] WARNING: '{args.model}' is not in the server's model list; "
                "requests will likely 404. Match --served-model-name on the GPU host."
            )
    except Exception as exc:
        print(f"[check] could not list models: {exc}", file=sys.stderr)
        return 1

    print(f"[check] sending a 16-token probe to model='{args.model}' ...")
    start = time.time()
    try:
        if args.chat:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
        else:
            resp = client.completions.create(
                model=args.model,
                prompt=args.prompt,
                max_tokens=16,
                temperature=0.0,
            )
            text = resp.choices[0].text or ""
    except Exception as exc:
        print(f"[check] generation failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    print(f"[check] OK in {elapsed:.2f}s; model replied with: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
