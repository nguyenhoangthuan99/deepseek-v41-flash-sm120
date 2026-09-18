#!/usr/bin/env python3
"""Summarize an extended engine sweep: coverage + per-shape crossover + winners."""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "extended.jsonl")
rows = []
for line in path.open():
    r = json.loads(line)
    if r.get("kind") in ("done", "built", "build_failed", "skip",
                         "candidate_load_error"):
        if r.get("kind") != "done":
            print("EVENT:", json.dumps(r)[:160])
        continue
    rows.append(r)

by_shape = defaultdict(list)
for r in rows:
    by_shape[(r["N"], r["K"])].append(r)

print(f"rows={len(rows)}  shapes={len(by_shape)}\n")
print(f"{'shape (N,K)':>14} {'Ms covered':>28} {'mxfp8 wins':>11} "
      f"{'triton wins':>12} {'cuda wins':>10} {'threshold_M':>12}")
cross = {"kind": "mxfp8_crossover", "shapes": {}}
for (N, K), cells in sorted(by_shape.items()):
    ms = sorted(c["M"] for c in cells)
    wins = defaultdict(int)
    for c in cells:
        wins[c.get("best_engine")] += 1
    pts = []
    for c in sorted(cells, key=lambda c: c["M"]):
        tt, mx = c.get("triton_tuned_us"), c.get("cublas_mxfp8_us")
        if tt and mx:
            pts.append({"M": c["M"], "mxfp8_over_tuned": round(tt / mx, 3)})
    winners = [p["M"] for p in pts if p["mxfp8_over_tuned"] >= 1.05]
    thr = winners[0] if winners else None
    print(f"  ({N:>5},{K:>4}) {str(ms):>28} {wins['cublas_mxfp8']:>11} "
          f"{wins['triton_tuned']:>12} {wins['cuda_custom']:>10} {str(thr):>12}")
    cross["shapes"][f"{N}x{K}"] = {
        "N": N, "K": K,
        "threshold_m": thr,
        "always_wins": bool(winners and winners[0] == pts[0]["M"]),
        "never_wins": not winners,
        "points": pts,
    }

out = path.with_name("mxfp8_crossover.json")
out.write_text(json.dumps(cross, indent=2))
print(f"\ncrossover written to {out}")