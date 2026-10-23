#!/usr/bin/env python3
"""Aggregate PyTorch-profiler Chrome traces from vLLM into kernel/family shares."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import statistics

FAMILIES = [
    ("nccl", re.compile(r"nccl|AllReduce|AllGather|ReduceScatter|AlltoAll|all_to_all|NCCL", re.I)),
    ("sparse_mla_attn", re.compile(r"sparse_mla|sparse_mla|SparseMla|trtllm|TRTLLM|fmha|FMHA|flashinfer|BatchPrefill|batch_prefill|decode_dsv4|dsv4|mla", re.I)),
    ("moe", re.compile(r"m_grouped|grouped_gemm|deep_gemm|DeepGemm|moe|MoE|silu_and_mul|act_and_mul|topk_softmax|moe_align|expand", re.I)),
    ("dense_gemm", re.compile(r"cutlass|CUTLASS|gemm|Gemm|nvjet|b12x|f8f8|fp8|w8a8|mxfp8", re.I)),
    ("indexer_topk", re.compile(r"index|Index|mqa|MQA|topk|TopK|sparse_index|logits|dsa", re.I)),
    ("kv_cache_write", re.compile(r"store_kv|StoreKV|rope_quant|quant|hadamard|compress|kv_cache|insert|reshape_and_cache|concat_and_cache", re.I)),
    ("norm_mhc", re.compile(r"rmsnorm|RMSNorm|layer_norm|mhc|MHC|hc_|sinkhorn|norm", re.I)),
    ("sampler", re.compile(r"sample|Sampler|argmax|multinomial|top_k|top_p", re.I)),
    ("engram_embed", re.compile(r"engram|Engram|embedding|gather|lookup", re.I)),
    ("elementwise_copy", re.compile(r"elementwise|vectorized|copy|cast|fill|memcpy|memcpy|add|mul|silu|gelu|rotary", re.I)),
]


def family_of(name: str) -> str:
    for fam, pat in FAMILIES:
        if pat.search(name):
            return fam
    return "other"


def load_trace(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="ignore") as fh:
            return json.load(fh)
    with open(path, errors="ignore") as fh:
        return json.load(fh)


def analyse_file(path: str):
    data = load_trace(path)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    kernels = []
    cpu_annotations = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if cat == "kernel" or cat == "cuda_kernel":
            dur = float(ev.get("dur") or 0.0)
            if dur > 0:
                name = ev.get("name") or "?"
                kernels.append((name, dur))
                fam = family_of(name)
                cpu_annotations[fam] = cpu_annotations.get(fam, 0.0) + dur
    # Aggregate by name
    per_name: dict[str, float] = {}
    for name, dur in kernels:
        per_name[name] = per_name.get(name, 0.0) + dur
    total = sum(per_name.values())
    return {"file": os.path.basename(path), "kernels": len(kernels),
            "total_us": total, "per_name": per_name, "per_family": cpu_annotations}


def fmt_name(name: str, width: int = 78) -> str:
    name = name.split("(")[0]
    return (name[: width - 1] + "…") if len(name) > width else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof-dir", default="/tmp/prof")
    ap.add_argument("--out", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    report = {}
    md = ["# vLLM SM120 kernel profile (torch profiler, TP8)", ""]
    labels = sorted([d for d in os.listdir(args.prof_dir)
                     if os.path.isdir(os.path.join(args.prof_dir, d))])
    for label in labels:
        files = sorted(glob.glob(os.path.join(args.prof_dir, label, "*.json*")))
        if not files:
            continue
        rank_files = [f for f in files if "_rank" in os.path.basename(f)]
        per_file = [analyse_file(f) for f in rank_files[:2]]
        usable = per_file
        agg: dict[str, float] = {}
        for pf in usable:
            for name, dur in pf["per_name"].items():
                agg[name] = agg.get(name, 0.0) + dur / len(usable)
        total = sum(agg.values())
        fam: dict[str, float] = {}
        for name, dur in agg.items():
            fam[family_of(name)] = fam.get(family_of(name), 0.0) + dur
        top = sorted(agg.items(), key=lambda kv: -kv[1])[: args.top]
        report[label] = {
            "trace_files": len(files),
            "rank0_total_kernel_us": per_file[0]["total_us"],
            "rank0_kernels": per_file[0]["kernels"],
            "mean_total_kernel_us": total,
            "family_us": dict(sorted(fam.items(), key=lambda kv: -kv[1])),
            "top_kernels": [{"name": n, "us": round(v, 1), "pct": round(100 * v / total, 2) if total else 0}
                            for n, v in top],
            "files": [pf["file"] for pf in per_file],
        }
        md.append(f"## {label}")
        md.append(f"- traces: {len(files)}  rank0 kernels: {per_file[0]['kernels']}  "
                  f"rank0 total kernel time: {per_file[0]['total_us']/1000:.1f} ms  "
                  f"(mean over up to 4 ranks: {total/1000:.1f} ms)")
        md.append("")
        md.append("| family | ms | share |")
        md.append("|---|---:|---:|")
        for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
            md.append(f"| {k} | {v/1000:.1f} | {100*v/total:.1f}% |")
        md.append("")
        md.append("| top kernel | ms | share |")
        md.append("|---|---:|---:|")
        for name, v in top:
            md.append(f"| `{fmt_name(name)}` | {v/1000:.2f} | {100*v/total:.2f}% |")
        md.append("")

    out = args.out or os.path.join(args.prof_dir, "profile_summary.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("\n".join(md))
    print("\n".join(md))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
