import os, torch
def main():
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu_group = dist.new_group(backend="gloo")
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(group=cpu_group, device=torch.device(f"cuda:{rank}"))
    print("rank", rank, "disabled", ca.disabled, flush=True)
    if ca.disabled:
        dist.barrier(); return
    n = 20480  # 40KB bf16
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    y = ca.all_reduce(x)
    torch.cuda.synchronize()
    print("rank", rank, "y0", float(y[0]), "y.sum", float(y.sum()), "x0", float(x[0]), flush=True)
    # try 2D input
    x2 = torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda")
    try:
        y2 = ca.all_reduce(x2); torch.cuda.synchronize()
        print("rank", rank, "2D y0", float(y2[0,0]), "sum", float(y2.sum()), flush=True)
    except Exception as e:
        print("rank", rank, "2D err", type(e).__name__, str(e)[:120], flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__":
    main()
