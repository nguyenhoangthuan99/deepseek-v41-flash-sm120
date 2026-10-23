# Attempt: port/rebuild DeepEP for SM120 to make DP attention fast

Goal: try the recommendation "get a working low-latency all-to-all (DeepEP) so
DP attention stops paying the 34% allgather+reducescatter tax".

## What was done

1. **Root cause of the original `cudaErrorSymbolNotFound`**: the shipped
   `deep_ep 2.0.0+local` wheel was compiled for a different arch (Blackwell
   SM100 / Hopper), so its legacy kernels had no SM120 cubins/symbols.
2. **Rebuilt DeepEP from source for SM120** using the repo's
   `vllm/tools/ep_kernels/install_python_libraries.sh`:
   - DeepEP pin `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb`, NVSHMEM 3.3.24,
     `TORCH_CUDA_ARCH_LIST=12.0`, CUDA 13.0.
   - Needed fixes to make the build succeed on this image:
     a. container had no `git` -> apt-get install git;
     b. torch headers need CUDA math headers -> prepend
        `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include` to CPATH;
     c. `-l:libnccl.so.2` not on linker path -> LIBRARY_PATH/LDFLAGS point at
        `/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib`;
     d. skip the script's unconditional `uv pip install torch` to keep the
        image's torch 2.13 ABI.
   - Result: `deep_ep-2.0.0+local-cp312-cp312-linux_x86_64.whl` built with
     `-gencode arch=compute_120,code=sm_120`; baked into a test image
     `dsv41-vllm:sm120-deepep`.
3. **Second blocker after the arch fix**: `intranode.cu` dispatch kernel asked
   for 192 KB dynamic shared memory (768 threads x 8 KB/warp), but SM120's
   opt-in limit is **101,376 B**. Patched the dispatch launch to
   `kNumThreads=256` (8 warps x 8 KB = 64 KB) and rebuilt the wheel.
4. **Third blocker**: after that, DeepEP initialises its managers but dies during
   warmup with:
   ```
   RuntimeError: Worker failed with error 'DeepEP error: CPU recv timeout'
   ```
   also with `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1`.

## Why it cannot be used on this node

DeepEP's intranode (HT) kernels use NVLink-style GPU-to-GPU flag/atomic
ordering for their barriers; its own docs warn that **PCIe connections may lead
to errors due to memory ordering issues** and require NVLink for the
low-latency paths. This VM is 8x RTX PRO 6000 on **PCIe only, no NVLink and no
RDMA**, so the intranode path times out waiting for flags even though the
kernels now exist for SM120. The internode/RDMA path is unavailable (no IB).

=> DeepEP is not fixable by config here; it needs a PCIe-specific all-to-all
implementation (new kernels), not an arch rebuild.

## Artifacts
- SM120 wheel: `/tmp/epbuild/dist/deep_ep-2.0.0+local-cp312-cp312-linux_x86_64.whl`
- Test image: `dsv41-vllm:sm120-deepep` (contains the rebuilt DeepEP; fails at
  warmup with the PCIe ordering timeout)
- Patched DeepEP source tree: `/tmp/epbuild/DeepEP` (intranode dispatch
  kNumThreads=256; wheel built from it)
- Build recipe can be reproduced with the steps above; the stock
  `tools/ep_kernels/install_python_libraries.sh` works once git/math headers/NCCL
  lib path are supplied.

## Conclusion / recommendation
- Keep **TP8 + NCCL** for latency-sensitive serving (unchanged).
- DP attention remains a **saturation/KV-capacity** option (C32 +14-16%, ~8x
  aggregate KV); its low-concurrency deficit is the allgather+reducescatter
  backend (34% of kernel time) plus per-rank small batches, and the faster
  DeepEP backend cannot run on PCIe-only SM120.
- If low-concurrency DP is a hard requirement, the next step is a custom
  PCIe-optimised all-to-all (ring/one-shot with PCIe-safe ordering), which is
  kernel development rather than a port of DeepEP.
