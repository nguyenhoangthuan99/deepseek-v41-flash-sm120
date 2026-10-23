# Communication optimization options: tests 1, 2, 3, 7 (SM120 vLLM)

Follow-up to `NCCL-SWEEP-REPORT.md` (NCCL AllReduce = 47–68% of GPU kernel time).
Four candidate next steps were tested/assessed.

## 1. Custom allreduce (replace NCCL for small/medium collectives) — REJECTED

vLLM's `CustomAllreduce` is hard-disabled here:
`"Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs."`
We patched the gate in the installed source and probed 8-rank PCIe:

- With the gate removed, `CustomAllreduce` initialises (`world=8`, `max_size=8 MB`,
  `fully_connected=False`) but **silently produces zeros** — the non-fully-connected
  multi-GPU kernel path is not implemented. So the disable gate is protecting a
  correctness bug, not just a policy.
- Forcing `fully_connected=True` (one-shot all-peer-write path) is **correct**
  (8×1 reduced, `sum=163840`), so the one-shot kernel is usable on PCIe P2P — but
  it is **much slower than NCCL** because PCIe has no multicast/NVLink:

| bytes | vLLM custom AR (µs) | NCCL (µs) | ratio |
|---:|---:|---:|---:|
| 40 KB | 473 | 55 | 8.6× slower |
| 160 KB | 2541 | 64 | 39.7× slower |
| 640 KB | 1974 | 146 | 13.5× slower |
| 1.31 MB | 5259 | 293 | 17.9× slower |

**Conclusion:** do not enable vLLM custom allreduce on this PCIe node; NCCL ring
is the right transport. (A correct custom PCIe allreduce would need a new ring
kernel, not the one-shot path, and is not expected to beat NCCL materially.)

## 2. Fused allreduce + residual + RMSNorm — DEFERRED (dependency failed)

Fusing these requires the allreduce itself to be a device-side kernel (custom
allreduce / one-shot) so the residual add + RMSNorm can run in the same launch.
Since option 1 is rejected, NCCL allreduce must stay a separate op; the only
fusion left would be residual-add+RMSNorm (already partly fused via
`fused_add_rms_norm`). Norm/MHC account for ~5.7% of kernel time total, so the
max realistic gain is low single-digit %, not worth the kernel risk now.

## 3. DP attention + EP (remove TP allreduce) — TESTED, KV capacity much larger but currently unstable

vLLM config used: `--data-parallel-size 8 --tensor-parallel-size 1
--enable-expert-parallel` (+ DSpark K3, 1M ctx, 0.85 mem, full decode graphs).

### KV cache pool impact (measured from vLLM logs)
| | TP8 baseline (current) | DP8 attention + EP8 |
|---|---|---|
| per-rank KV memory | 21.63 GiB (deployment log) | **21.28 GiB** |
| per-rank KV tokens | ~10M (per-rank capacity) | **10,107,817 tokens** |
| max concurrency @1M tokens (per rank) | — | **9.64×** |
| KV replication | **replicated across 8 TP ranks** (MLA latent needed by every head shard) | **sharded by request** (each rank owns its requests) |
| aggregate unique KV capacity | ≈ per-rank capacity (no multiplier) | **≈ 8 × 10.1M ≈ 80.9M tokens** |
| max single-request context | ~10M tokens | **10.1M tokens (still ≫ 1M limit)** |

So DP attention roughly **multiplies aggregate KV capacity by ~8×** at the same
per-rank KV budget, because the TP replication of the MLA latent disappears;
per-request max context stays at ~10M tokens, so the 1M-context requirement is
still satisfied. Weight layout: dense/attention weights replicated per DP rank,
MoE experts EP-sharded (54.99 GiB consumed per rank in our run).

### Performance (measured, warmed, same bench harness)
| workload | TP8 baseline | DP attention | change |
|---|---:|---:|---:|
| 512 in, C1 | 151.3 tok/s | 87.5 tok/s | **−42%** |
| 512 in, C4 | 311.0 tok/s | 260.1 tok/s | −16% |
| 512 in, C8 | 472.7 tok/s | **workers crashed** (HTTP 500, gloo peer closed) | unstable |
| higher C / longer ctx | — | not completed (server crash) | unstable |

**Conclusion:** DP attention is the only way to remove the TP AllReduce entirely
and it gives a ~8× larger aggregate KV pool, but in vLLM with DSpark + Engram it
is **slower at C1–C4 and unstable at C8+** (worker crash). Not deployable as-is.
Revisit only if KV capacity/saturation becomes the priority and the crash is
fixed (e.g., disable DSpark for DP runs, investigate Engram DP sync).

## 7. NCCL version upgrade — BLOCKED by ABI

Image ships `nvidia-nccl-cu13 2.30.7` (torch's `torch.cuda.nccl.version()` path);
torch is built against the 2.29/2.30 ABI. The earlier attempt with NCCL 2.32.3
produced symbol-not-found/launch failures (`cudaErrorSymbolNotFound`, ABI
mismatch), which is why the deployment pins 2.30.7 via
`/opt/nccl2307` + `LD_LIBRARY_PATH`. Upgrading requires rebuilding torch or the
vLLM image against the new NCCL; expected PCIe-collective gain is small relative
to that effort/risk. **Not recommended now.**

## Summary
| option | result |
|---|---|
| 1 custom allreduce | **rejected** (correct only with forced one-shot path; 8–18× slower than NCCL on PCIe) |
| 2 fused allreduce+norm | deferred; depends on option 1, max gain low single-digit % |
| 3 DP attention + EP | **KV capacity ~8× (80.9M vs per-rank ~10M tokens), 1M ctx still fits**, but C1 −42%, C4 −16%, unstable at C8+ |
| 7 NCCL upgrade | ABI-blocked; not worth without rebuilding torch/image |

Recommended path remains: NCCL protocol choice per deployment profile
(`LL128` for saturated C≥16, default for low-concurrency) + the remaining
structural options only if DP attention stability is fixed.
