#!/usr/bin/env python3
"""Render the extended engine sweep as a human-readable table + crossover summary."""
import json, sys
from collections import defaultdict
from pathlib import Path

rows=[]
for line in Path(sys.argv[1] if len(sys.argv)>1 else "extended.jsonl").open():
    r=json.loads(line)
    if r.get("kind"): continue
    rows.append(r)
by=defaultdict(list)
for r in rows: by[(r["N"],r["K"])].append(r)

print(f"cells={len(rows)} shapes={len(by)}\n")
for (N,K),cells in sorted(by.items()):
    print(f"=== ({N}, {K}) ===")
    print(f"{'M':>5} {'trit_tun':>9} {'trit_gen':>9} {'cuda':>8} {'mxfp8':>8} {'bf16':>8} | {'best':>12} {'mxfp8/tun':>9}")
    for c in sorted(cells,key=lambda c:c["M"]):
        tt=c.get("triton_tuned_us"); tg=c.get("triton_generic_us")
        cu=c.get("cuda_custom_us"); mx=c.get("cublas_mxfp8_us"); bf=c.get("cublas_bf16_dequant_us")
        r_ = f"{tt/mx:.2f}" if tt and mx else "-"
        f=lambda v: f"{v:>9.2f}" if v else "        -"
        print(f"{c['M']:>5} {f(tt)} {f(tg)} {f(cu)} {f(mx)} {f(bf)} | {str(c.get('best_engine','-')):>12} {r_:>9}")
    print()
