#!/usr/bin/env python3
"""Weight/VRAM-fit planner for PD (prefill/decode) disaggregation of
DeepSeek-V4.1-Flash.

Answers one question honestly: *can a given prefill/decode GPU split hold the
model, and is it balanced?*

Key architecture fact (see results/weight-breakdown.json): the model has NO
separable encoder/decoder weight sets. The vision encoder (ViT) is ~1 GB; the
"~8B prefill / ~16B decode" asymmetry is a compute-path difference over the SHARED
`layers.*` MoE stack (295.6 GB) plus the Engram memory tables (203.1 GB). In
SGLang PD disaggregation BOTH pools load the FULL model, sharded across their own
GPUs; the split is by phase and KV is transferred prefill->decode over the
interconnect. So per-GPU weight = gpu_resident_weight / pool_gpus, identical in
both pools -- the split balances THROUGHPUT, never weights.

Usage:
    python3 disagg_plan.py                         # default topology sweep
    python3 disagg_plan.py --prefill 4 --decode 4  # check a specific split
    python3 disagg_plan.py --engram-cpu            # assume Engram tables on host
    python3 disagg_plan.py --measure --model-dir /path/to/checkpoint
Exit code is nonzero if the requested --prefill/--decode split does not fit
(used by serve-disagg.sh as a preflight gate).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BREAKDOWN = HERE.parent / "results" / "weight-breakdown.json"


def measure_checkpoint(model_dir: str) -> dict:
    """Recompute group byte sizes from safetensors headers (slow; over NAS ~1 min)."""
    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, dict] = {}

    def header(fname: str) -> dict:
        if fname not in headers:
            with (root / fname).open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                headers[fname] = json.loads(fh.read(n))
        return headers[fname]

    def group(name: str) -> str:
        low = name.lower()
        if name.startswith("vision") or low.startswith("aligner") or "image" in low:
            return "vision_encoder_vit"
        if "engram" in low:
            return "engram_memory"
        if name.startswith("mtp") or "nextn" in low:
            return "mtp_dspark"
        if "indexer" in low or ".index" in low:
            return "indexer_dsa"
        if name.startswith("embed") or name == "head" or low.startswith(("head.", "norm")):
            return "embed_head"
        return "layers_moe_decoder"

    sizes: dict[str, int] = defaultdict(int)
    for name, fname in index.items():
        head = header(fname)
        if name in head:
            a, b = head[name]["data_offsets"]
            sizes[group(name)] += b - a
    total = sum(sizes.values())
    return {
        "model": root.name,
        "total_weight_gb": round(total / 1e9, 1),
        "total_weight_bytes": total,
        "num_tensors": len(index),
        "groups_gb": {k: round(v / 1e9, 2) for k, v in sizes.items()},
    }


def load_breakdown(args) -> dict:
    if args.measure:
        if not args.model_dir:
            sys.exit("--measure requires --model-dir")
        return measure_checkpoint(args.model_dir)
    if not BREAKDOWN.is_file():
        sys.exit(f"Missing {BREAKDOWN}; re-run with --measure --model-dir <ckpt>")
    return json.loads(BREAKDOWN.read_text())


def evaluate(pool_gpus: int, gpu_resident_gb: float, vram_gb: float,
             overhead_gb: float, kv_min_gb: float) -> dict:
    per_gpu_weight = gpu_resident_gb / pool_gpus
    kv_budget = vram_gb - per_gpu_weight - overhead_gb
    return {
        "pool_gpus": pool_gpus,
        "per_gpu_weight_gb": round(per_gpu_weight, 1),
        "kv_budget_gb": round(kv_budget, 1),
        "fits": per_gpu_weight + overhead_gb <= vram_gb and kv_budget >= kv_min_gb,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vram-gb", type=float, default=95.6, help="Per-GPU VRAM (GiB->GB).")
    p.add_argument("--overhead-gb", type=float, default=7.0,
                   help="Per-GPU non-weight reserve: framework, NCCL, CUDA graphs, activations.")
    p.add_argument("--kv-min-gb", type=float, default=4.0,
                   help="Minimum leftover per-GPU VRAM for KV to call a pool servable.")
    p.add_argument("--prefill", type=int, default=None, help="Prefill-pool GPU count to check.")
    p.add_argument("--decode", type=int, default=None, help="Decode-pool GPU count to check.")
    p.add_argument("--engram-cpu", action="store_true",
                   help="Assume the 203 GB Engram tables are offloaded to host RAM.")
    p.add_argument("--measure", action="store_true", help="Recompute sizes from --model-dir.")
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"))
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = p.parse_args()

    bd = load_breakdown(args)
    total = bd["total_weight_gb"]
    engram = bd["groups_gb"].get("engram_memory", 0.0)
    vision = bd["groups_gb"].get("vision_encoder_vit", 0.0)
    gpu_resident = total - (engram if args.engram_cpu else 0.0)

    header = {
        "model": bd.get("model"),
        "total_weight_gb": total,
        "vision_encoder_gb": vision,
        "engram_gb": engram,
        "engram_on_cpu": args.engram_cpu,
        "gpu_resident_weight_gb": round(gpu_resident, 1),
        "vram_gb": args.vram_gb,
        "overhead_gb": args.overhead_gb,
    }

    # Standard topologies to compare. A "pool" holds the full gpu-resident model.
    topos = [
        ("aggregated TP8 (current, 1 pool)", 8),
        ("disagg 4+4 (each pool = 4 GPUs)", 4),
        ("disagg 6+2 / 2+6 (small pool = 2)", 2),
        ("disagg 8+8 two-node (each pool = 8)", 8),
    ]
    rows = []
    for label, pool in topos:
        rows.append((label, evaluate(pool, gpu_resident, args.vram_gb,
                                     args.overhead_gb, args.kv_min_gb)))

    requested = None
    if args.prefill or args.decode:
        pref = evaluate(args.prefill or 0 or 1, gpu_resident, args.vram_gb,
                        args.overhead_gb, args.kv_min_gb) if args.prefill else None
        dec = evaluate(args.decode, gpu_resident, args.vram_gb,
                       args.overhead_gb, args.kv_min_gb) if args.decode else None
        requested = {"prefill": pref, "decode": dec,
                     "fits": all(x["fits"] for x in (pref, dec) if x)}

    if args.json:
        print(json.dumps({"header": header,
                          "topologies": [{"label": l, **r} for l, r in rows],
                          "requested": requested}, indent=2))
    else:
        print(f"\nModel: {header['model']}  total weights {total} GB "
              f"(vision encoder only {vision} GB; Engram {engram} GB)")
        print(f"GPU VRAM {args.vram_gb} GB/card, reserve {args.overhead_gb} GB/card "
              f"for framework+graphs+activations")
        print(f"GPU-resident weight to shard: {header['gpu_resident_weight_gb']} GB "
              f"(Engram {'on CPU' if args.engram_cpu else 'on GPU'})\n")
        print(f"  {'topology':38s} {'GPUs/pool':>9s} {'wt/GPU':>8s} {'KV/GPU':>8s}  fits")
        for label, r in rows:
            mark = "yes" if r["fits"] else "NO"
            print(f"  {label:38s} {r['pool_gpus']:>9d} "
                  f"{r['per_gpu_weight_gb']:>7.1f}G {r['kv_budget_gb']:>7.1f}G  {mark}")
        if requested:
            print(f"\n  requested split prefill={args.prefill} decode={args.decode}: "
                  f"{'FITS' if requested['fits'] else 'DOES NOT FIT'}")
        print("\n  Note: both pools carry the full GPU-resident weight; the split "
              "balances throughput, not weights.\n")

    ok = True if requested is None else requested["fits"]
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
