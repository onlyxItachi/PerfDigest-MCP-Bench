# Demanding workloads, part 1: cuBLAS + torch.compile (2026-07-12)

First installment of the demanding-workloads series ([roadmap: cuBLAS, torch.compile,
LLVM lowering, Rust cargo, Vector ISA]). This part covers the two scenarios that run
on **already-shipped backends** (`nsight` ncu-rep + `chrome_trace` torch-trace,
perfdigest-mcp **1.1.1**).

**Host:** ASUS TUF A14 — RTX 4060 Laptop (sm_89), driver 610.43.02, ncu 2026.2.1,
torch 2.12.1+cu132 (Python 3.12, triton 3.7.1).
**Workloads:** [`workloads/cublas_bench.py`](../workloads/cublas_bench.py),
[`workloads/torch_compile_bench.py`](../workloads/torch_compile_bench.py).
**Method note:** digests for v1.1.1-only surfaces (`summarize_report`,
`compare_metrics`, torch-trace) were produced by the v1.1.1 repo code directly
(`uv run`); the registered live MCP server in the measuring session was still a
uvx-cached 1.0.0 env for the ncu calls. Payload bytes are identical either way
(the server is a thin shell). Tokens ≈ bytes / 4, same convention as
[TOKEN_AB_V1.1_2026-07-11.md](TOKEN_AB_V1.1_2026-07-11.md).

---

## Scenario A — cuBLAS: closed-source vendor kernels

**The question this tests:** when the kernels that run are cuBLAS/cuBLASLt internals
with no readable source, can an agent diagnose from the digest alone? This is the
tool's founding thesis with no escape hatch — there is nothing to `grep`.

One 2048×2048×2048 GEMM, four precision paths. The library picked a **different
closed-source kernel family per path**, visible immediately in `list_kernels`:

| path | kernel picked by cuBLAS | duration | derived TFLOP/s |
|---|---|---|---|
| fp32, TF32 off | `ampere_sgemm_64x64_nn` | 2477 µs | 6.9 |
| fp32, TF32 on | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x256_16x4_nn_align4>` | 1608 µs | 10.7 |
| fp16 | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` | 776 µs | 22.1 |
| bf16 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 776 µs | 22.1 |

`get_metrics` on each (default core set) is enough to explain the ranking **without
any source access**:

| metric | sgemm (fp32) | cutlass (TF32) | fp16 | bf16 |
|---|---|---|---|---|
| compute_pct_peak | **71.9** | 46.0 | 47.7 | 47.7 |
| dram_pct_peak | 7.9 | 11.7 | 8.6 | 11.5 |
| achieved_occupancy | 31.6 | **8.3** | 16.1 | 16.1 |
| registers_per_thread | 126 | **226** | 234 | 234 |
| shared_mem_per_block | 8.4 KB | 0 | 32 KB | 32 KB |
| l2_hit_rate | 97.6 | 96.1 | 95.4 | 93.5 |

The story reads straight off the counters: the fp32 kernel saturates the CUDA cores
(72% compute-of-peak — precision-bound, not badly tuned), while the tensor-core
kernels win 1.5–3.2× *at 8–16% occupancy* — the classic tensor-core trade of
occupancy for giant register tiles (226–234 regs/thread). An agent seeing
"faster kernel, 4× lower occupancy" learns not to "fix" occupancy on vendor
tensor-core kernels. None of this needed a single line of source.

### Token cost (measured)

| payload | bytes | ≈ tokens | ratio vs digest path |
|---|---|---|---|
| **digest orient: `list_kernels` + 4× `get_metrics`** | **8,189** | **~2,050** | 1× |
| **`summarize_report` top-5 (one call)** | **3,128** | **~780** | 0.4× |
| raw `ncu --page details` text | 238,679 | ~59,700 | **29×** |
| raw `ncu --page raw --csv` | 378,760 | ~94,700 | **46×** |
| the `.ncu-rep` itself | 132.9 MB | unreadable binary | — |

**Honest finding:** 51% of the `list_kernels` payload (2,772 of 5,435 bytes) is just
TWO noise-kernel names — torch's `randn` launches carry 1,386-char mangled template
names each. The digest is honest (names are identity, we don't truncate), but on
torch workloads the *name payload* dominates the unit list. Recorded as a v1.2
design question (a `short_name` alongside the full name?), not silently changed.

---

## Scenario B — torch.compile: machine-generated kernel names

**The question this tests:** Inductor emits kernels named by a code generator
(`triton_poi_fused_add_mul_relu_sigmoid_0`, `triton_per_fused_prepare_softmax_online_1`)
that change whenever the fused graph changes. Do aggregation, ranking, and the
`kernel_b` rename workflow survive names no human chose?

Model: fusable pointwise chain (`relu·sigmoid+add·mul`) + matmul + softmax-sum,
run eager vs `torch.compile`, both traced with Kineto (`torch-trace` format), and
the compiled run additionally captured with ncu (`-k regex:triton`).

**What the digests showed:**

- **Fusion is visible structurally, not just in timings.** The eager trace's
  gpu_kernel units are four separate ATen elementwise kernels (clamp_min, sigmoid,
  mul, add); the compiled trace has exactly one
  `triton_poi_fused_add_mul_relu_sigmoid_0` — the op chain is literally in the name.
- **`compare_metrics` handled the rename scenario it was designed for:** eager
  `softmax_warp_forward<...>` → compiled `triton_per_fused_prepare_softmax_online_1`
  via `kernel_b`, avg_time_us −83.7%. (Caveat the tool correctly leaves to the
  agent: the compiled path split softmax+sum across *three* kernels, so
  single-kernel deltas understate/overstate — unit equivalence is a judgment call,
  and the tool stays a translator.)
- **Bookkeeping tagging fired in the wild:** Kineto's `PyTorch Profiler (0)` span
  (8.3 ms, longer than any real work) came back as `... [Trace]`, and Inductor's
  CPU-side launch spans collide with the GPU kernel names, so the GPU units came
  back disambiguated as `triton_poi_fused_... [kernel]`. Both marks did their job:
  ranking stays honest, and the agent can discount envelope spans.
- **Cross-report name matching (trace ↔ ncu) works with one friction point:** ncu
  reports the identical generated name (`triton_poi_fused_add_mul_relu_sigmoid_0`),
  but the trace unit carries the ` [kernel]` collision tag, so an exact-name lookup
  across the two formats needs the tag stripped or `#index` addressing. Documented
  workflow note, not a bug — the tag is what keeps the name unique inside its own
  report.
- **ncu on the fused kernel closes the loop:** `get_metrics` on
  `triton_poi_fused_add_mul_relu_sigmoid_0` → dram_pct_peak **92.7**,
  compute_pct_peak 7.9, occupancy 98%, 207 GB/s. A fused elementwise chain should
  be a memory-bound streaming kernel, and the digest says exactly that — Inductor's
  fusion left nothing on the table for this chain.

### Token cost (measured)

| payload | bytes | ≈ tokens | ratio |
|---|---|---|---|
| `summarize_report` top-5, compiled trace | 1,379 | ~345 | **74×** vs its raw trace (101,794 B) |
| `summarize_report` top-5, eager trace | 1,307 | ~327 | **115×** vs its raw trace (150,336 B) |
| `compare_metrics` eager→compiled | 979 | ~245 | — |
| ncu digest orient (list + 1× get_metrics) | 1,846 | ~460 | **99×** vs `--page details` (182,426 B) |

---

## Carry-forward findings

1. **The closed-source thesis holds at its hardest**: vendor GEMM kernels were
   fully diagnosable (precision-bound vs tensor-core occupancy trade) from the
   default core set alone, at 29–46× fewer tokens than the raw pages.
2. **Generated names are fine; long names are the cost.** Inductor's dynamic names
   stressed nothing — aggregation, ranking, collision tags, and `kernel_b` all
   behaved. The real token tax on torch workloads is ATen's 1.4 KB mangled
   template names in unit lists → v1.2 design question (`short_name`?).
3. **Cross-format identity** (same kernel seen by Kineto and ncu) is a real agent
   workflow; today it works via substring/`#index`. Worth a doc note in the main
   repo's usage convention.
4. Live-server nuance for users: a uvx-registered server keeps serving its cached
   version until the environment refreshes — after a release, restart the MCP
   client (or `uv cache clean perfdigest-mcp`) to pick up the new tools.

**Still to come in this series:** Vector-ISA/SIMD CPU counters (`linux_perf`
vocabulary stress; blocked on `kernel.perf_event_paranoid=2` per boot), Rust cargo
analysis (criterion-JSON backend candidate, v1.2), LLVM lowering
(llvm-mca / optimization-remarks backend, v1.2).
