import os, torch, torch.distributed as dist
def main():
    rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    n = int(os.environ.get("NBYTES", 40960)) // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(int(os.environ.get("WARM", "10"))):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    for _ in range(int(os.environ.get("MEAS", "3"))):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0: print("done", float(x[0]))
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__": main()
