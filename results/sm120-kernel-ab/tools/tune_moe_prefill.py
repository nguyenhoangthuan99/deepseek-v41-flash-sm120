import os, sys, gc, json, statistics, torch
sys.path.insert(0, "/DI"); sys.path.insert(0, "/DI/benchmarks/sm120"); sys.path.insert(0, "/opt/dsv41/vllm-src")
from benchmarks.sm120 import bench_ab_moe_vllm as bm
import ab_dg_sf as dgsf
try:
    import deep_gemm as dg
except Exception:
    from vllm.third_party import deep_gemm as dg

SHAPES = {"fc1": (4608, 5120), "fc2": (5120, 2304)}
CASES = os.environ.get("CASES", "fp4,fp8").split(",")
TS = [int(x) for x in os.environ.get("TS", "1024,2048,4096,8192").split(",")]
ALIGNS = [int(x) for x in os.environ.get("ALIGNS", "64,128").split(",")]
E = int(os.environ.get("E", "56"))

def bench(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(3):
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        samples.append(st.elapsed_time(en) / iters * 1000)
    return statistics.median(samples)

def main():
    torch.cuda.set_device(0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    for case in CASES:
        for shape, (N, K) in SHAPES.items():
            torch.manual_seed(1234)
            w = torch.randn((E, N, K), device="cuda") * 0.05
            wq, wsf = bm.quant_w_fp8(w) if case == "fp8" else bm.quant_w_fp4(w)
            del w; gc.collect(); torch.cuda.empty_cache()
            sfb = dgsf.pack_sfb(wsf, N)
            for T in TS:
                counts = bm.skewed_counts(T, E, gen)
                act_rows = []
                for e in range(E):
                    r = int(counts[e])
                    if r: 
                        x = torch.randn((r, K), device="cuda").bfloat16()
                        act_rows.append(bm.quant_act_fp8(x)); del x
                    else:
                        act_rows.append((None, None))
                for align in ALIGNS:
                    try:
                        aligned, m_sum, aq, asf, psum, m_ind = bm.make_inputs(case, counts, align, act_rows, K)
                        dg.set_mk_alignment_for_contiguous_layout(align)
                        sfa = dgsf.pack_sfa(asf, m_sum)
                        out = torch.zeros((m_sum, N), dtype=torch.bfloat16, device="cuda")
                        def run():
                            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                                (aq, sfa), (wq, sfb), out, psum,
                                recipe_a=(1, 32), recipe_b=(1, 32),
                                use_psum_layout=True, expected_m_for_psum_layout=m_sum)
                        us = bench(run)
                        flops = 2.0 * m_sum * N * K
                        rows.append({"case": case, "shape": shape, "T": T, "align": align,
                                     "m_sum": m_sum, "us": us, "tflops": flops/us/1e6})
                        print(f"{case} {shape} T={T:5d} align={align:3d} m_sum={m_sum:6d} {us:9.2f}us {flops/us/1e6:7.1f} TF/s", flush=True)
                    except Exception as exc:
                        print(f"{case} {shape} T={T} align={align} FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    out_path = "/DI/benchmarks/sm120/results/moe_prefill_sweep.json"
    json.dump(rows, open(out_path, "w"), indent=2)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
