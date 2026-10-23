# SM120 serving benchmark + DSpark configuration (VM106 vs VM100)

## 1. DSpark: upstream DeepInfer config vs measured best on this hardware

### Authoritative checkpoint fields (DeepSeek-V4.1-Flash-1017 `config.json`)
| field | value |
|---|---|
| `num_nextn_predict_layers` | **3** |
| `dspark_block_size` | **5** |
| `dspark_target_layer_ids` | **[37, 38, 39]** |
| `dspark_target_attn_input` | True (default) |
| `dspark_markov_rank` | 256 |
| `dspark_noise_token_id` | 128799 |
| `dspark_n_routed_experts` / `num_experts_per_tok` | 128 / 3 |

DeepInfer derives its `dflash_cfg` from those fields (`hugging_face/config.py::_build_dflash_cfg`)
and its runtime knob is `DEEPINFER_USE_NEXTN_DECODE` — **default 5** (“max speculative
guesses per step”; `0` disables speculation). DSpark confidence head is on
(upstream fixed), confidence temperature default 1.0, target+draft CUDA graphs on
(`DEEPINFER_DFLASH_ENFORCE_EAGER=0`), graph bucket interval 8, zero-overhead
scheduling on by default.

⇒ **Upstream-DeepInfer default DSpark config = block 5 / up to 5 guesses, static
equal-block, probabilistic draft + standard rejection, confidence head on,
target+draft graphs on.** No compact/ragged mode.

### Measured best on this 8×RTX PRO 6000 kit (vLLM fork, docs/RESULTS.md)
Block sizes 1/3/5/8 were screened: **block 3 = balanced default**, **block 5 =
code-oriented alternative** (checkpoint blocksize). Static equal-block only —
compact/ragged DSpark fails the Engram equal-block assertion and is not a working
mode. Measured DSpark3 vs no-speculation (warm aggregate output tok/s):

| workload | C | no spec | DSpark block 3 |
|---|---:|---:|---:|
| prose | 1 | 106.7 | **200.8** |
| prose | 4 | 369.1 | **495.4** |
| prose | 16 | 926.7 | **970.6** |
| prose | 32 | **1394.8** | 1368.5 |
| code | 1 | 107.6 | **232.8** |
| code | 4 | 365.2 | **601.7** |
| code | 16 | 914.1 | **1136.4** |
| code | 32 | 1368.1 | **1589.3** |

### Recommended configuration
**vLLM fork (`deepseek-v41-flash-sm120`):**
```
SPEC=1  DSPARK_BLOCK=3
--speculative-config '{"method":"dspark","num_speculative_tokens":3,
                       "draft_sample_method":"probabilistic",
                       "rejection_sample_method":"standard"}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[3,4,6,8,12,16,24,32,48,64,96,128]}'
# prefill graphs stay disabled; NCCL_P2P_LEVEL=PHB
```
Use `SPEC=0` only for saturated prose workloads where C32 aggregate throughput is
the sole objective; use `DSPARK_BLOCK=5` for code-dominated workloads.

**DeepInfer native:** keep checkpoint DSpark fields untouched; set
`DEEPINFER_USE_NEXTN_DECODE=3` to match the measured balanced K3 (default 5 = K5 /
code-oriented); keep `DEEPINFER_DFLASH_ENFORCE_EAGER=0`,
`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE=1.0`,
`DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1`; no compact/ragged speculation.

## 2. Warmed endpoint benchmark: VM106 (this fork + wired kernels) vs VM100
Both endpoints: same model/fork/hardware, TP8, 1M ctx, DSpark K3, FULL_DECODE_ONLY
graphs, block 64, b12x + MoE alignment 64, NCCL PHB. Client direction: VM106
localhost; VM100 `http://10.200.108.100:30000`. **Per-cell warmup batch at the
measured concurrency + 2–3 measured rounds (median)**; unique random prefixes so
no cross-request prefix-cache hits; output 128 tokens, ignore_eos.

### Unique-prefix (cold-cache), warmed
| len | C | VM106 TTFT | VM100 TTFT | VM106 TPOT | VM100 TPOT | VM106 tok/s | VM100 tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 172.5 ms | **104.4** | 5.23 | 5.67 | 153.3 | 155.7 |
| 512 | 16 | 861.1 | **753.3** | 16.19 | 17.75 | 630.6 | 639.4 |
| 512 | 32 | 1212.5 | **1068.9** | 26.63 | 26.65 | 817.0 | 822.3 |
| 1024 | 1 | 176.2 | **125.1** | 5.20 | 5.80 | 153.4 | 147.7 |
| 2048 | 1 | 200.5 | **189.9** | 5.17 | 5.36 | 150.0 | 144.7 |
| 4096 | 1 | 345.0 | 330.5 | 4.88 | 5.61 | 128.5 | 123.0 |
| 4096 | 32 | 6011.0 | **5211.8** | 55.12 | 58.17 | 305.6 | 313.8 |
| 8192 | 1 | 652.2 | **622.7** | 5.20 | 5.08 | 97.7 | 100.0 |
| 16384 | 1 | 1289.8 | **1221.3** | 5.11 | 5.36 | 66.1 | 66.8 |
| 32768 | 1 | 2514.8 | **2384.9** | 4.78 | 5.30 | 40.9 | 41.8 |
| 32768 | 8 | 11414.9 | **10866.4** | 72.92 | 71.57 | 48.9 | 51.2 |

VM100 is ~5–15% faster at small-context TTFT and ~5% at large contexts; TPOT and
aggregate tok/s are within noise — both are the same fork and hardware.

### Prefix-cache effect (shared prefix / repeated prompt)
| case | VM106 | VM100 |
|---|---:|---:|
| 16k unique-prefix TTFT | 1289.8 ms | 1221.3 ms |
| 16k shared-prefix TTFT | **221.5 ms** | **146.2 ms** |
| 16k repeated-prompt TTFT | 221.0 ms | 145.1 ms |
| 32k unique / shared | 2514.8 / **264.5** ms | 2384.9 / **181.7** ms |

⇒ prefix caching is worth ~5.8× (VM106) / 8.4× (VM100) TTFT at 16k; shared-prefix
numbers are the realistic serving case for repeated system prompts.

### Huge single-request contexts (unique prefix, TTFT)
| tokens | VM106 | VM100 |
|---:|---:|---:|
| 65,536 | 5033 ms | **4746 ms** |
| 131,072 | 10210 ms | **9669 ms** |
| 262,144 | 21434 ms | **20347 ms** |

## 3. Kernel switching vs CUDA graphs
- Engine choice is made inside the op, so **a captured decode graph bakes in
  whichever engine was selected at capture time**; different batch buckets may use
  different engines (each bucket has its own graph), and changing the choice for an
  already-captured bucket requires re-capture (process restart). Eager prefill
  calls re-evaluate per call.
- Safety guard: the DeepInfer kernel is never first-used inside capture (its JIT
  and allocations would break capture); capture falls back to FlashInfer until the
  shape has been exercised eagerly.
- Modes (`VLLM_SM120_SPARSE_MLA_KERNEL`): `fast` (default, FlashInfer — best
  measured on the 584 B layout), `accurate` (DeepInfer engine — better numerics,
  slower), `balanced` (per-shape table). Measured A/B says FlashInfer wins all
  supported buckets, so the only accuracy/speed trade-off currently available is
  forcing `accurate` (or `balanced` after table recalibration against the actual
  runtime shape tuple, which differs from the initial guess: heads padded, topk =
  window+index_topk rather than 512).

## 4. Files
Benchmark client: `benchmarks/sm120/bench_endpoint.py` (DeepInfer tree, shared
read-only). Raw results: `results/sm120-kernel-ab/endpoint_{vm106,vm100}_*.json`.
