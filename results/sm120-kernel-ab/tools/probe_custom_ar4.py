import os, torch
def main():
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(group=cpu_group, device=torch.device(f"cuda:{rank}"))
    if rank == 0:
        print("CA_PROBE", {"disabled": ca.disabled, "max_size": getattr(ca,'max_size',None),
                           "world": getattr(ca,'world_size',None),
                           "fully_connected": getattr(ca,'fully_connected',None)}, flush=True)
    dist.barrier()
    if ca.disabled:
        dist.destroy_process_group(); return
    rows = []
    for nbytes in (40*1024, 160*1024, 640*1024, 1310*1024):
        n = nbytes // 2
        x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(x)
        def ca_call(): out.copy_(ca.all_reduce(x))
        def nccl_call(): dist.all_reduce(x, op=dist.ReduceOp.SUM)
        for fn in (ca_call, nccl_call):
            for _ in range(10): fn()
        torch.cuda.synchronize()
        res = {}
        for name, fn in (("custom", ca_call), ("nccl", nccl_call)):
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize(); st.record()
            for _ in range(50): fn()
            en.record(); torch.cuda.synchronize()
            res[name] = round(st.elapsed_time(en) / 50 * 1000, 2)
        if rank == 0:
            print("CA_LAT", nbytes, res, "sum_ok", float(out[0].item()), flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__":
    main()
