# DP attention: test, profile and debug (SM120 / vLLM fork)

Goal: find out whether DP attention + EP (no TP allreduce) is faster than the
deployed TP8 model-parallel layout on this 8x RTX PRO 6000 PCIe node.

All tests: same image (`dsv41-vllm:sm120`), 1M context, block 64, 0.85 mem,
DSpark disabled unless marked "+spec", warmed client, 512-token prompts.

## 1. Throughput comparison (output tok/s, 512-token prompts)

| concurrency | TP8 no-spec | DP8×TP1 no-spec | DP4×TP2 no-spec | TP8 +DSpark K3 | DP8 +DSpark (light run) |
|---:|---:|---:|---:|---:|---:|
| 1 | **84.5** | 46.4 | 50.0 | **151.3** | 58.2 |
| 2 | **152.2** | 88.2 | 90.2 | — | — |
| 4 | **272.9** | 157.6 | 170.0 | **311.0** | 220.0 |
| 8 | **442.0** | 287.9 | 334.3 | **472.7** | 373.4 |
| 16 | **638.3** | 547.2 | 603.1 | **626.8** | — |
| 32 | 861.5 | **1002.4** | 982.3 | 772.5 | — |

Conclusions:
- **DP attention is not faster at low concurrency on this node** (0.45–0.80× of
  TP8 for C1–C16), regardless of DP8×TP1 or DP4×TP2.
- It only wins at **saturated decode (C32): +16%** (DP8) / +14% (DP4×TP2) vs TP8.
- With DSpark K3, DP is slower than TP8+spec at all measured concurrency
  (C1 58 vs 151, C4 220 vs 311, C8 373 vs 473 tok/s).

## 2. Profile of DP attention (torch profiler, decode C1, rank traces)

Kernel-time families with DP8×TP1 + allgather_reducescatter all2all backend:

| family | share |
|---|---:|
| NCCL comm (AllGather 19.5% + ReduceScatter 12.3% + AllGatherV 1.9%) | **34.1%** |
| sparse-MLA / attention kernels | 24.7% |
| MoE grouped GEMM | 16.6% |
| dense GEMM (b12x/CUTLASS mxfp8) | 11.8% |
| other | 6.5% |
| MHC + norm | 4.7% |
| engram/embed/gather | 1.7% |

vs the TP8 profile (decode C1): AllReduce 58.1%, sparse MLA 13.0%, MoE 11.1%,
dense 6.1%, MHC 3.7%.

**Root cause of DP being slower:** DP attention removes the TP AllReduce, but
vLLM's default EP all-to-all backend (`allgather_reducescatter`) replaces it with
**AllGather + ReduceScatter per MoE layer** (34% of kernel time). On PCIe, those
collectives are latency-bound at decode batch sizes (messages of tens of KB,
8 ranks), so the replacement communication is *worse* than TP AllReduce. In
addition, attention/dense weights are replicated per DP rank, so small decode
batches leave those kernels underutilised (attn 24.7% of time at tiny batch).

## 3. Faster all-to-all backends: DeepEP is unusable on SM120

Tried `--all2all-backend deepep_high_throughput` (and its manager initialises),
but the worker crashes during model warmup:

```
RuntimeError: CUDA runtime exception (/tmp/ep_kernels_workspace/DeepEP/
csrc/kernels/legacy/layout.cu:139): 500 (cudaErrorSymbolNotFound, named symbol not found)
```

Also tried forcing SM120 compilation (`TORCH_CUDA_ARCH_LIST=12.0`,
`CUDAARCHS=120`, clearing the DeepEP JIT workspace) — same failure. This is the
same SM120/DeepEP issue seen earlier; fixing it requires rebuilding/porting
DeepEP's kernels for SM120 (not a config change). `flashinfer_all2allv` and
NIXL paths require NVLink/RDMA and are not applicable to this PCIe node.

## 4. DSpark + DP attention interaction

- Earlier run (C16 max_num_seqs=32, max_tokens=128, rounds=2) crashed at C8 with
  `gloo ... Connection closed by peer` / worker died.
- A lighter repro (C1/C4/C8, max_tokens=64, rounds=1) did **not** crash, so the
  failure is load/timing dependent rather than deterministic at startup.
- No root cause established beyond "worker died during/after cudagraph/DP
  collective activity"; TP8 + DSpark is stable across all our runs.

## 5. KV-cache effect of DP attention (measured from vLLM logs)

- Per rank: **11.85M tokens** KV (DP8 no-spec run), 9.64× concurrency at 1M ctx.
- KV is sharded by request across DP ranks instead of replicated across TP
  ranks → aggregate unique capacity is ~8× TP8's per-rank capacity (same reason
  as the earlier SGLang measurement), with per-request max ~11M tokens (1M ctx
  still fits).
- So DP attention is a **capacity/saturation** option, not a low-latency one.

## 6. Recommendations

1. Keep TP8 for latency-sensitive/mixed serving (current deployment).
2. Use DP attention only for saturated throughput (C≥32) or KV-capacity-heavy
   serving; at C32 it is +14–16% over TP8.
3. If DP attention must be fast at low concurrency, the required fix is a
   working low-latency all-to-all backend on SM120 (DeepEP port/rebuild, or a
   PCIe-optimised all-to-all). The current allgather+reducescatter backend is the
   bottleneck (34% of kernel time) and DeepEP fails with
   `cudaErrorSymbolNotFound` on this hardware.
4. DP4×TP2 is a small improvement over DP8×TP1 (+8–16% at C4–C16) but still
   below TP8 below C32; not worth switching.

## 7. Is DP slow because the kernels are not tuned for DP? (tested)

No — protocol/kernel tuning does not close the gap:

- **NCCL protocol tuning of the DP all-to-all**: DP8 no-spec re-run with
  `NCCL_PROTO=LL128` (the protocol that helps TP8 saturated decode) gives
  C1 44.8 vs 46.4 tok/s, C4 152.0 vs 157.6, C8 288.0 vs 287.9, C16 531.6 vs
  547.2, C32 995.9 vs 1002.4 — i.e. slightly worse at every concurrency.
  The allgather+reducescatter backend is not fixable by NCCL protocol choice.
- **Sparse-MLA decode kernels are already tuned at DP's per-rank batch size**:
  DP splits requests across ranks, so at C1-C8 each rank decodes exactly one
  request (T=1 token/step). The FlashInfer autotune sweep showed the heuristic
  is already optimal for T<=32 (18.46 us, at the launch-latency floor), and our
  chunks_per_block cache only helps the T=64 bucket (C16-equivalent per rank:
  C/8 -> T=8 at C64). There is no per-rank small-T win left.
- **Architecture bound**: a DP rank computes attention/dense for its own request
  only (1 GPU per request) while TP8 spreads one request over 8 GPUs; even if the
  34% DP communication were free, DP C1 step would be ~13 ms vs TP8's 10.6 ms.
- The properly tuned low-latency all-to-all (DeepEP) is unusable on SM120
  (`cudaErrorSymbolNotFound`; also after forcing SM120 arch env), so the only
  "tuned for DP" comm path cannot run here.

Therefore DP attention is not a kernel-tuning problem at low concurrency: it is
a parallelization-layout + missing SM120 all-to-all backend problem. Its wins
(saturated C32 +16%, ~8x aggregate KV) appear only when enough requests per rank
justify the layout.
