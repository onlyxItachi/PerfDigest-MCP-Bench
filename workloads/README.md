# workloads — cross-backend capture workloads

Two tiny, self-contained workloads with deliberately **opposite roofline
character**, used to produce the real reports digested in
[`../results/CROSS_BACKEND_2026-06-15.md`](../results/CROSS_BACKEND_2026-06-15.md).

| Source | Backend | Two units | Contrast |
|---|---|---|---|
| `gpu_bench.cu` | `nsight` (NVIDIA `ncu`) | `saxpy_mem_bound`, `fma_compute_bound` | DRAM-bound vs SM-compute-bound |
| `cpu_bench.c` | `linux_perf` (`perf`) | `mem_bound_chase`, `compute_bound_poly` | cache-bound vs in-register compute |

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
