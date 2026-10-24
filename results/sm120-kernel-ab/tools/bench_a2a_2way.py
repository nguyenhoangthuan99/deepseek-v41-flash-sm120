import os, statistics, torch, torch.distributed as dist
HID=5120; TOPK=6; WORLD=8; EXPERTS=384
def bench(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    st,en=torch.cuda.Event(True),torch.cuda.Event(True); vals=[]
    for _ in range(3):
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize(); vals.append(st.elapsed_time(en)/iters*1000)
    return statistics.median(vals)
def main():
    rank=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    g=torch.Generator(device="cuda").manual_seed(7)
    rows=[]
    for M in (1,2,4,8,16):
        local=torch.randn((M,HID),device="cuda",dtype=torch.bfloat16)
        experts=torch.randint(0,EXPERTS,(M,TOPK),generator=g,device="cuda")
        dst=experts//(EXPERTS//WORLD)
        send=[int((dst==d).any(dim=1).sum().item()) for d in range(WORLD)]
        cs=[torch.zeros(WORLD,dtype=torch.long,device="cuda") for _ in range(WORLD)]
        dist.all_gather(cs, torch.tensor(send,dtype=torch.long,device="cuda"))
        recv=[int(cs[r][rank].item()) for r in range(WORLD)]
        parts=[local[(dst==d).any(dim=1)] for d,c in enumerate(send) if c>0]
        a2a_in=torch.cat(parts,0) if parts else torch.empty((0,HID),device="cuda",dtype=torch.bfloat16)
        buf1=torch.empty((sum(recv),HID),device="cuda",dtype=torch.bfloat16)
        buf2=torch.empty((sum(send),HID),device="cuda",dtype=torch.bfloat16)
        def f_2way():
            dist.all_to_all_single(buf1,a2a_in,output_split_sizes=recv,input_split_sizes=send)
            dist.all_to_all_single(buf2,buf1,output_split_sizes=send,input_split_sizes=recv)
        ag=torch.empty((WORLD*M,HID),device="cuda",dtype=torch.bfloat16)
        def f_agrs():
            dist.all_gather_into_tensor(ag,local)
            out=ag[rank*M:(rank+1)*M].contiguous()
            dist.reduce_scatter_tensor(out,ag)
        u2=bench(f_2way); ug=bench(f_agrs)
        if rank==0:
            rows.append((M,u2,ug))
            print(f"M={M:3d} (per-rank tokens) dispatch+combine a2a={u2:8.2f}us  allgather+reducescatter={ug:8.2f}us  ratio={u2/ug:.2f}", flush=True)
        dist.barrier()
    dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
