import os, sys, gc, torch
sys.path.insert(0, "/DI")
sys.path.insert(0, "/DI/benchmarks/sm120")
sys.path.insert(0, "/opt/dsv41/vllm-src")

from benchmarks.sm120 import bench_ab_moe_vllm as bm  # noqa: E402
import ab_dg_sf as dgsf  # noqa: E402
try:
    import deep_gemm as dg  # noqa: E402
except Exception:
    from vllm.third_party import deep_gemm as dg  # noqa: E402

CASE = os.environ.get("CASE", "fp8")
SHAPE = os.environ.get("SHAPE", "fc1")
T = int(os.environ.get("T", "256"))
E = int(os.environ.get("E", "56"))
ALIGN = int(os.environ.get("ALIGN", "64" if T <= 256 else "128"))
SHAPES = {"fc1": ("fc1_gate_up", 4608, 5120), "fc2": ("fc2_down", 5120, 2304)}

def main():
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    name, N, K = SHAPES[SHAPE]
    w = torch.randn((E, N, K), device="cuda") * 0.05
    if CASE == "fp8":
        wq, wsf = bm.quant_w_fp8(w)
    else:
        wq, wsf = bm.quant_w_fp4(w)
    del w; gc.collect(); torch.cuda.empty_cache()
    sfb = dgsf.pack_sfb(wsf, N)
    counts = bm.skewed_counts(T, E, gen)
    act_rows = []
    for e in range(E):
        r = int(counts[e])
        if r == 0:
            act_rows.append((None, None)); continue
        x = torch.randn((r, K), device="cuda").bfloat16()
        act_rows.append(bm.quant_act_fp8(x)); del x
    aligned, m_sum, aq, asf, psum, m_indices = bm.make_inputs(CASE, counts, ALIGN, act_rows, K)
    dg.set_mk_alignment_for_contiguous_layout(ALIGN)
    sfa = dgsf.pack_sfa(asf, m_sum)
    out = torch.zeros((m_sum, N), dtype=torch.bfloat16, device="cuda")
    def run():
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (aq, sfa), (wq, sfb), out, psum,
            recipe_a=(1, 32), recipe_b=(1, 32),
            use_psum_layout=True, expected_m_for_psum_layout=m_sum)
    for _ in range(int(os.environ.get("WARM", "4"))):
        run()
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()
    print(f"done {CASE} {SHAPE} T={T} E={E} align={ALIGN} m_sum={m_sum} out0={float(out[0,0])}")

if __name__ == "__main__":
    main()
