#!/usr/bin/env python3
"""Serving A/B client for the MXFP8 engine.

Drives the running endpoint with a fixed workload at several concurrency levels,
records per-request latency and aggregate throughput, and writes JSON + a short
summary. Run once against the engine-off server and once against engine-on; the
two result files are compared by ``compare_serving.py``.

Deliberately measures the same shape the earlier tuning work used so the numbers
are comparable: a fixed prompt with 256-token outputs (decode-dominated) plus a
long-prompt prefill probe.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = (
    "You are a meticulous engineer. Explain, in depth and with concrete "
    "examples, how a block-scaled FP8 GEMM differs numerically from a "
    "per-tensor FP8 GEMM, and why a 32x32 power-of-two block scale can be "
    "re-expressed losslessly as 1x32 MX groups. "
)
PREFILL_PROMPT = (
    "Summarise the following engineering log, listing every distinct failure "
    "mode and its mitigation. Be exhaustive.\n\n" + ("fault detected; retry scheduled. " * 900)
)


def post(base, prompt, max_tokens, timeout=600):
    body = json.dumps({
        "model": "deepseek-v41-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    dt = time.perf_counter() - t0
    usage = payload.get("usage", {})
    return dt, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)


def run_level(base, prompt, max_tokens, concurrency, requests):
    lat, ctok, ptok = [], 0, 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(post, base, prompt, max_tokens) for _ in range(requests)]
        t0 = time.perf_counter()
        for f in cf.as_completed(futs):
            dt, c, p = f.result()
            lat.append(dt)
            ctok += c
            ptok += p
        wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_s": round(wall, 3),
        "agg_output_tok_s": round(ctok / wall, 1),
        "agg_total_tok_s": round((ctok + ptok) / wall, 1),
        "median_latency_s": round(statistics.median(lat), 3),
        "p90_latency_s": round(sorted(lat)[int(0.9 * len(lat)) - 1], 3),
        "completion_tokens": ctok,
    }


def prefill_probe(base, max_tokens=16, timeout=900):
    dt, c, p = post(base, PREFILL_PROMPT, max_tokens, timeout=timeout)
    return {"prompt_tokens": p, "ttft_proxy_s": round(dt, 3),
            "output_tokens": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--levels", default="1,4,16,32")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # warmup
    post(args.base_url, DEFAULT_PROMPT, 16)
    post(args.base_url, DEFAULT_PROMPT, 16)

    results = {"label": args.label, "base_url": args.base_url,
               "max_tokens": args.max_tokens, "levels": []}
    for level in [int(x) for x in args.levels.split(",")]:
        r = run_level(args.base_url, DEFAULT_PROMPT, args.max_tokens, level,
                      max(args.requests, level))
        print(json.dumps(r), flush=True)
        results["levels"].append(r)
    results["prefill"] = prefill_probe(args.base_url)
    print(json.dumps(results["prefill"]), flush=True)

    (out / "serving.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out/'serving.json'}")


if __name__ == "__main__":
    main()