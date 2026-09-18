#!/usr/bin/env python3
"""Compare two serving A/B runs (engine off vs on) and print the delta."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p):
    return json.loads((Path(p) / "serving.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    args = ap.parse_args()
    off, on = load(args.off), load(args.on)

    print(f"{'conc':>5} {'off tok/s':>10} {'on tok/s':>10} {'ratio':>7} "
          f"{'off p90':>9} {'on p90':>9} {'lat ratio':>10}")
    rows = []
    for a, b in zip(off["levels"], on["levels"]):
        r = b["agg_output_tok_s"] / a["agg_output_tok_s"]
        lr = b["p90_latency_s"] / a["p90_latency_s"]
        rows.append({"concurrency": a["concurrency"],
                     "off_tok_s": a["agg_output_tok_s"],
                     "on_tok_s": b["agg_output_tok_s"],
                     "throughput_ratio": round(r, 4),
                     "off_p90": a["p90_latency_s"], "on_p90": b["p90_latency_s"],
                     "latency_ratio": round(lr, 4)})
        print(f"{a['concurrency']:>5} {a['agg_output_tok_s']:>10.1f} "
              f"{b['agg_output_tok_s']:>10.1f} {r:>7.3f} {a['p90_latency_s']:>9.3f} "
              f"{b['p90_latency_s']:>9.3f} {lr:>10.3f}")

    pf = {"off": off.get("prefill"), "on": on.get("prefill")}
    if pf["off"] and pf["on"]:
        print(f"\nprefill {pf['off']['prompt_tokens']} tok: "
              f"off={pf['off']['ttft_proxy_s']}s on={pf['on']['ttft_proxy_s']}s "
              f"ratio={pf['on']['ttft_proxy_s']/pf['off']['ttft_proxy_s']:.3f}")
    result = {"throughput": rows, "prefill": pf,
              "summary": {
                  "max_throughput_ratio": max(r["throughput_ratio"] for r in rows),
                  "min_throughput_ratio": min(r["throughput_ratio"] for r in rows),
              }}
    out = Path(args.on) / "comparison.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()