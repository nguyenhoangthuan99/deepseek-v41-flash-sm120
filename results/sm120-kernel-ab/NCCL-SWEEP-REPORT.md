# NCCL tuning + EPLB analysis for `deepseek-v41-flash-sm120`

Profiling showed NCCL AllReduce is 47–68% of GPU kernel time in every workload
(see `PROFILE-BOTTLENECK-REPORT.md`). This phase tunes the communication layer
and evaluates EPLB for long-run deployment.

## 1. NCCL protocol sweep (measured, warmed endpoint, TP8, DSpark K3)

Server restarted per config with one env change; benchmark: unique prefixes,
per-cell warmup batch, 2 rounds small / 1 round large; clocks locked 2100 MHz.

Configs: `baseline` (NCCL tuner), `NCCL_PROTO=LL128`, `NCCL_PROTO=Simple`.
`NCCL_ALGO=Tree` could not be evaluated: the server for that config never came
up / exited (Tree allreduce is not supported by this PCIe P2P topology) — not a
usable option here.

### TPOT (ms, lower better)
| len | C | baseline | LL128 | Simple | best |
|---|---:|---:|---:|---:|---|
| 512 | 1 | **5.41** | 6.09 | 7.97 | baseline |
| 512 | 4 | **10.18** | 10.71 | 11.99 | baseline |
| 512 | 8 | **11.73** | 12.03 | 15.98 | baseline |
| 512 | 16 | 17.60 | **14.75** | 16.46 | LL128 (+16%) |
| 512 | 32 | 27.62 | **24.21** | 24.95 | LL128 (+12%) |
| 4096 | 1 | **5.64** | 6.37 | 7.16 | baseline |
| 4096 | 4 | **11.82** | 12.52 | 14.43 | baseline |
| 4096 | 8 | 13.41 | **12.78** | 19.20 | LL128 |
| 4096 | 16 | 30.62 | 29.47 | **28.25** | simple/baseline≈tie |
| 4096 | 32 | 56.98 | **54.09** | 56.30 | LL128 |
| 16384 | 1 | **6.18** | 7.30 | 7.79 | baseline |
| 32768 | 1 | **5.07** | 6.75 | 8.28 | baseline |
| 32768 | 8 | **137.43** | 137.75 | 139.69 | baseline |

### Aggregate output tok/s (geometric mean over all cells)
| config | TTFT ratio vs baseline | TPOT ratio | tok/s ratio |
|---|---:|---:|---:|
| NCCL_PROTO=LL128 | 0.962 (3.8% worse) | 1.029 (2.9% worse) | 0.993 |
| NCCL_PROTO=Simple | 1.025 | 1.124 (12.4% worse) | 0.906 |

**Conclusion**
- Default NCCL tuner is best for latency-sensitive and small-concurrency
  workloads (C1–C8) and long-context decode.
- **`NCCL_PROTO=LL128` is the best throughput setting for saturated decode:
  +10% aggregate tok/s at C32/512 (850 vs 772), +5% at C16, +4% at 4096 C8/C32,
  at the cost of 7–11% TPOT at C1–C4.** Because NCCL env is process-wide, this
  is a deployment-profile choice, not a per-request switch:
  - interactive / low-concurrency: keep default;
  - throughput-oriented (C≥16): export `NCCL_PROTO=LL128`.
- `NCCL_PROTO=Simple` is not recommended here (worse TTFT/TPOT and throughput
  in nearly all cells, despite the older SGLang-era C32 datapoint).

## 2. Can NCCL latency be hidden instead of tuned?

Options evaluated against this fork:

1. **Dual-batch overlap (DBO) — NOT possible for DeepSeek-V4.1 in this fork.**
   vLLM has `--enable-dbo` + `--ubatch-size` + `DBO` token thresholds
   (`parallel.py: enable_dbo`, `dbo_decode_token_threshold=32`,
   `dbo_prefill_token_threshold=512`), which would pipeline two microbatches and
   overlap the other microbatch's compute with this one's AllReduce. But
   `VllmConfig` explicitly rejects it for this model:
   `"DeepSeek V4.1 Engram does not support DBO or microbatching. Disable
   --enable-dbo and set --ubatch-size to 0."` So no supported overlap path via
   DBO while Engram is enabled.
2. **Async scheduling** only overlaps host scheduling, not NCCL kernels.
3. **Reduce allreduce count instead of hiding it**: the alternative architecture
   is DP attention + EP (replaces TP allreduce with all-to-all). Previously
   measured in `results/overlap-dp-attention-exploration.json`: aggregate C1
   3.2× faster, C4 2.1× faster, C16 ~5% faster, but **C32 1.16× slower and 32k
   prefill TTFT 1.51× slower**, plus a 2.4× larger aggregate KV pool. That is a
   workload-dependent architecture change, not an NCCL tune; keep TP8 unless the
   deployment is low-concurrency dominated.
4. **NCCL-level latency reduction** is the practical path: protocol/LL128,
   channel/socket counts, and symmetric-memory collectives. LL128 is measured
   above; the channel/socket and `VLLM_USE_NCCL_SYMM_MEM` sweeps are the next
   step (no model/code changes required).
5. Implementing manual micro-batch pipelining outside DBO would require
   splitting the model forward and duplicating CUDA graphs — high-risk, not
   justified before exhausting NCCL env tuning.

## 3. EPLB (expert-parallel load balancing) in this fork

**Yes, vLLM has EPLB here** (`vllm/config/parallel.py`):
- Enable: `--enable-eplb` (requires `--enable-expert-parallel`).
- `--eplb-config` fields: `window_size`, `step_interval`, `num_redundant_experts`
  (default 0), `log_balancedness` (default off; adds comm overhead),
  `use_async` (default **True**), `policy` ("default"), `communicator`
  (auto: nixl on CUDA, torch_gloo fallback; **async EPLB rejects
  `torch_nccl`/`pynccl` due to NCCL multi-stream conflicts**).
- The shipped launcher (`serve-vllm.sh`) does **not** enable EPLB (no flags), so
  today's deployment runs without it.

**Long-run deployment impact**
- Benefits: under skewed request routing (domain/language hot experts), static
  expert placement causes stragglers; EPLB rebalances and can recover
  steady-state throughput. With `num_redundant_experts > 0` it can also reduce
  the tail of hot experts.
- Costs/risks for a long-running deployment on this node:
  - **Rearrangement traffic is expensive on PCIe-only**: moving expert shards
    between ranks is a large all-to-all-like transfer (no NVLink, no RDMA);
    expect multi-second-ish disruptive windows per rearrangement depending on
    interval and number of moved experts. `use_async=True` reduces pauses but the
    default async path cannot use `torch_nccl`/`pynccl` (needs nixl/gloo) and
    interacts with NCCL streams.
  - **VRAM**: redundant experts consume HBM and shrink the KV-cache pool
    (currently ~21.6 GiB at 1M context); every 1 GiB taken is ~5% of KV budget.
  - **CUDA graphs / DSpark**: expert mapping changes mid-serving must be handled
    in the graph-captured MoE path; validate long-duration soak with DSpark K3
    before trusting it (DSpark draft/target graphs make this riskier).
  - **Operational complexity**: policy/interval tuning, monitoring
    (`log_balancedness` itself costs communication), and nondeterministic
    rearrangements complicate reproducibility.
- **Recommendation**: keep EPLB **off** for the initial deployment (current
  state) — it is safe and has no long-run degradation. Enable it only if
  production traffic shows persistent expert imbalance; then start with
  synchronous EPLB, a modest `num_redundant_experts` (1–2 per rank), a long
  `step_interval`, `use_async=false` (avoid NCCL stream conflicts), and validate
  a multi-hour soak on latency spikes and KV-cache loss. For a fixed traffic
  mix, an offline static expert placement is the lower-risk alternative.

## 4. Files / next steps
- Results: `endpoint_nccl_{baseline,simple,ll128}_{a,b}.json`,
  `nccl_sweep_summary.json`.
- Next NCCL experiments (no model changes): `NCCL_MIN/MAX_NCHANNELS`,
  `NCCL_NSOCKS_PERTHREAD`/`SOCKET_NTHREADS`, `VLLM_USE_NCCL_SYMM_MEM=1`,
  and `NCCL_PROTO=LL128` combined with those; then re-profile to confirm the
  AllReduce share drops.
