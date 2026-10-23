#!/usr/bin/env python3
"""Drive vLLM torch-profiler windows for representative serving workloads.

Requires the server started with:
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/prof"}'
POSTs /start_profile and /stop_profile around each workload and archives the
per-rank traces into /tmp/prof/<label>/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import statistics
import time
import urllib.request
import uuid

FILLER = ("Background reading: engineers build reliable systems by checking assumptions, "
          "measuring results, and testing changes under different workloads. ")


def make_prompt(target_tokens: int, tail: str = "") -> str:
    nonce = uuid.uuid4().hex
    body = (FILLER * ((target_tokens * 4 // len(FILLER)) + 1))[: target_tokens * 4]
    return f"{nonce} {body} {tail}\nQuestion: Summarise the engineering log above.\nAnswer:"


def post_stream(base: str, model: str, prompt: str, max_tokens: int, timeout: float = 1800.0) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            for choice in obj.get("choices") or []:
                if choice.get("text"):
                    if first is None:
                        first = time.perf_counter()
                    tokens += 1
            if obj.get("usage"):
                tokens = obj["usage"].get("completion_tokens", tokens)
    done = time.perf_counter()
    return {"total_s": done - t0, "ttft_s": (first - t0) if first else None,
            "tokens": max(tokens, 1)}


def post(base: str, path: str, timeout: float = 900.0) -> int:
    req = urllib.request.Request(f"{base}{path}", data=b"{}",
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except Exception as exc:  # noqa: BLE001
        print(f"  POST {path} failed: {exc}", flush=True)
        return -1


def run_batch(base: str, model: str, n_tokens: int, concurrency: int, requests: int, max_tokens: int) -> dict:
    prompts = [make_prompt(n_tokens, f"req{i}") for i in range(requests)]
    results = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(post_stream, base, model, p, max_tokens) for p in prompts]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t0
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"]]
    out_tokens = sum(r["tokens"] for r in results)
    return {"wall_s": wall, "requests": requests, "concurrency": concurrency,
            "ttft_ms_median": statistics.median(ttfts) * 1000 if ttfts else None,
            "output_tokens": out_tokens, "output_tok_s": out_tokens / wall if wall else None}


WORKLOADS = [
    # label, input tokens, concurrency, requests, max_tokens
    ("prefill_32k_c1", 32768, 1, 1, 1),
    ("prefill_8k_c8", 8192, 8, 8, 1),
    ("decode_c1", 512, 1, 1, 256),
    ("decode_c8", 512, 8, 8, 128),
    ("decode_c32", 512, 32, 32, 64),
    ("mixed_16k_c4", 16384, 4, 4, 64),
]


def archive(prof_dir: str, label: str) -> int:
    dst = os.path.join(prof_dir, label)
    os.makedirs(dst, exist_ok=True)
    deadline = time.time() + 600
    moved = 0
    while time.time() < deadline:
        files = [f for f in glob.glob(os.path.join(prof_dir, "*.json*")) if os.path.isfile(f)]
        if files:
            break
        time.sleep(1.0)
    for f in glob.glob(os.path.join(prof_dir, "*.json*")):
        if os.path.isfile(f):
            shutil.move(f, os.path.join(dst, os.path.basename(f)))
            moved += 1
    return moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30100")
    ap.add_argument("--model", default="deepseek-v41-flash")
    ap.add_argument("--prof-dir", default="/tmp/prof")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/mnt/nas/alex/models/deepseek-v41-flash-sm120/results/sm120-kernel-ab/profile_windows.json")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    summary = []
    for label, n_tokens, conc, reqs, max_tokens in WORKLOADS:
        if only and label not in only:
            continue
        print(f"\n=== {label}: tokens={n_tokens} C={conc} reqs={reqs} out={max_tokens} ===", flush=True)
        # Warmup: same shape, profiler off.
        warm = run_batch(args.base_url, args.model, n_tokens, conc, reqs, min(max_tokens, 8))
        print(f"  warmup: wall={warm['wall_s']:.2f}s ttft={warm['ttft_ms_median']:.0f}ms", flush=True)
        if post(args.base_url, "/start_profile") != 200:
            print("  profiler start failed; skipping", flush=True)
            continue
        row = run_batch(args.base_url, args.model, n_tokens, conc, reqs, max_tokens)
        status = post(args.base_url, "/stop_profile")
        n_files = archive(args.prof_dir, label)
        print(f"  profiled: wall={row['wall_s']:.2f}s ttft={row['ttft_ms_median']:.0f}ms "
              f"tok/s={row['output_tok_s']:.1f} stop={status} traces={n_files}", flush=True)
        summary.append({"label": label, "input_tokens": n_tokens, "concurrency": conc,
                        "requests": reqs, "max_tokens": max_tokens,
                        "warm": warm, "profiled": row, "trace_files": n_files})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
