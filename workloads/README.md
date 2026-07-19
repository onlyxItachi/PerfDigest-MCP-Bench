# workloads — cross-backend capture workloads

Tiny, self-contained workloads used to produce the real reports digested in
[`../results/`](../results/).

| Source | Backend(s) | Units | What it stresses |
|---|---|---|---|
| `gpu_bench.cu` | `nsight` (NVIDIA `ncu`) | `saxpy_mem_bound`, `fma_compute_bound` | DRAM-bound vs SM-compute-bound roofline contrast |
| `cpu_bench.c` | `linux_perf` (`perf`) | `mem_bound_chase`, `compute_bound_poly` | cache-bound vs in-register compute contrast |
| `cublas_bench.py` | `nsight` | four cuBLAS GEMM kernel families (fp32 / TF32 / fp16 / bf16) | digest quality on **closed-source vendor kernels** — no source to read, counters only |
| `torch_compile_bench.py` | `chrome_trace` + `nsight` | eager ATen kernels vs Inductor `triton_*_fused_*` | machine-generated kernel names, eager-vs-compiled `kernel_b` compare, Kineto bookkeeping/collision tags |
| `simd_bench.c` | `linux_perf` | `scalar_fma_loop` (0 vector instrs), `vector_fma_loop` (autovectorized) | vendor SIMD event names outside the CPU vocabulary (AMD `fp_ret_sse_avx_ops.*` vs Intel `fp_arith_inst_retired.*`) — the `expand()` escape hatch + a v1.2 vocabulary case |
| `torch_train_bench.py` | `chrome_trace` + `nsight` | eager ATen forward/backward/AdamW kernels vs Inductor `triton_*_backward`-flavored fused kernels + eager `_foreach_*`/`multi_tensor_apply_kernel` optimizer kernels | a full training step (forward+backward+optimizer): Inductor-fused backward kernels, a `compare_metrics` `kernel_b` pairing that reports the wrong sign of a fusion win if read in isolation, `_foreach_*` op identity vs. generic optimizer GPU kernel names, untagged Dynamo/AOTDispatcher bookkeeping spans |

The Python workloads need a CUDA-enabled torch (measured with 2.12.1+cu132,
triton 3.7.1); capture commands are in each file's docstring.

The compiled binaries and the `.ncu-rep` / `.csv` reports are **gitignored**
(regenerable, large/binary). Only the sources live here.

## Reproduce

```bash
# --- GPU (needs an NVIDIA GPU + CUDA toolkit; RmProfilingAdminOnly=0 -> no sudo) ---
nvcc -O3 -arch=sm_89 -o gpu_bench gpu_bench.cu
ncu --set full --launch-skip 2 --launch-count 2 -o gpu_bench --force-overwrite ./gpu_bench

# --- CPU (Linux perf; works at perf_event_paranoid=2, user-scope only) ---
cc -O2 -fno-inline -fno-omit-frame-pointer -o cpu_bench cpu_bench.c -lm
LC_ALL=C perf stat -j -o cpu_stat.json \
  -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,LLC-loads,LLC-load-misses \
  -- ./cpu_bench
perf record -e cycles:u -g -o perf.data -- ./cpu_bench
LC_ALL=C perf report -i perf.data --stdio -n --no-children -g none --percent-limit 0.5 > cpu_report.txt
```

Then digest **through the perfdigest MCP** (never read the raw report into the
agent's context):

```text
list_kernels(report_ref="gpu_bench.ncu-rep", format="ncu-rep")
get_metrics(report_ref="gpu_bench.ncu-rep", format="ncu-rep", kernel="saxpy_mem_bound")
get_metrics(report_ref="cpu_stat.json",     format="perf-stat-json", kernel="0")
get_metrics(report_ref="cpu_report.txt",    format="perf-report",    kernel="compute_bound_poly")
```

## Two capture gotchas this run surfaced (real, host-dependent)

1. **`perf stat -j` under a comma-decimal locale emits invalid JSON**
   (`"counter-value" : "6960,59"`, `"pcnt-running" : 100,00`). Always capture
   with `LC_ALL=C`.
2. **`perf` appends scope modifiers** (`cycles:u`, `instructions:u`) on a
   `perf_event_paranoid>=2` host. The digester now normalizes these to the bare
   counter name (see the report's "Finding" section).
