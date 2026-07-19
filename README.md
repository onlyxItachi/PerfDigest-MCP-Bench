# PerfDigest-MCP-Bench

Benchmark workloads and evaluation results for
[`perfdigest-mcp`](https://github.com/onlyxItachi/PerfDigest-MCP) — the MCP server
that digests profiler reports (NVIDIA `ncu`, AMD `rocprof`, Linux `perf`, Apple
Metal, `ptxas`) into a small, token-efficient, cross-vendor JSON signal for LLM
coding agents.

This repo holds **artifacts**, not code: A/B token-efficiency studies, real-hardware
cross-backend runs, the tiny CUDA/C workloads used to produce them, and (going
forward) captured profiler report binaries. The main repo,
[onlyxItachi/PerfDigest-MCP](https://github.com/onlyxItachi/PerfDigest-MCP), holds
the server source **and its unit tests** — unit tests deliberately stay there
because they gate CI on every change; this repo is downstream of it, evaluating
released behavior, not gating it.

## Results, and what version each one measured

| Document | perfdigest version | Host | What it measures |
|---|---|---|---|
| [`results/RESULTS.md`](results/RESULTS.md) | v1.0.0-era | RTX 4060 Laptop (sm_89) | Context-cost A/B: digest vs raw `ncu` payload, same diagnosis, ~14–130x fewer tokens per turn |
| [`results/CROSS_BACKEND_2026-06-15.md`](results/CROSS_BACKEND_2026-06-15.md) | v1.0.0 + the `perf` `:u` scope-modifier fix | Same host (ASUS TUF A14, Ryzen AI 9 HX 370 + RTX 4060 Laptop) | Real captures across `nsight` (GPU) and `linux_perf` (CPU) digested through the same MCP call shape, with the `None` ≠ `0.0` honesty rule verified end to end |
| [`results/TOKEN_AB_V1.1_2026-07-11.md`](results/TOKEN_AB_V1.1_2026-07-11.md) | **v1.1.0-rc** (main `dff3d19`) | Same host | Token A/B across FOUR backends (`ncu`, `torch-trace`, `perf`, `ptxas`) incl. the new `summarize_report`/`compare_metrics` flows: 16–103× on GPU/trace paths, 4× on perf, and the honest 0.8× ptxas floor case |
| [`results/DEMANDING_WORKLOADS_1_CUBLAS_INDUCTOR_2026-07-12.md`](results/DEMANDING_WORKLOADS_1_CUBLAS_INDUCTOR_2026-07-12.md) | **v1.1.1** | Same host | Demanding workloads pt. 1: closed-source cuBLAS GEMMs (four kernel families diagnosed from counters alone, 29–46×) and torch.compile/Inductor generated kernels (eager-vs-compiled `kernel_b` compare, collision tags in the wild, 74–115×) |
| [`results/INTERPRETER_MATRIX_2026-07-18.md`](results/INTERPRETER_MATRIX_2026-07-18.md) | **v1.2.0-rc** (main `76796f0`) | Same host, 6 uv-managed CPythons (3.10–3.14 + 3.14t free-threaded) | Interpreter matrix, not a workload study: digest byte-identity across all six interpreters (10/11 pure-Python backends identical, the one divergence traced to CPython 3.12's `sum()` Neumaier-summation change), reader parse latency per version, and 3.14t cache-lock contention (correctness holds, 2.36× throughput vs the 3.14 GIL build) |
| [`results/DEMANDING_WORKLOADS_2_TORCH_TRAINING_2026-07-19.md`](results/DEMANDING_WORKLOADS_2_TORCH_TRAINING_2026-07-19.md) | **v1.2.0** (released PyPI) | Same host | Demanding workloads pt. 2: torch.compile TRAINING step (MLP+LayerNorm+GELU, AdamW) — Inductor-fused backward kernels diagnosed as memory-bound from counters alone, a `compare_metrics` `kernel_b` pairing caught reporting the *wrong sign* of a fusion win (single kernel vs its true multi-kernel replacement), `_foreach_*` op identity vs. generic `multi_tensor_apply_kernel` GPU names, untagged Dynamo/AOTDispatcher bookkeeping spans in the top-5, 96–450× token reduction |
| [`results/DEMANDING_WORKLOADS_3_CPP_BUILD_LOOP_2026-07-19.md`](results/DEMANDING_WORKLOADS_3_CPP_BUILD_LOOP_2026-07-19.md) | **v1.2.0** (released PyPI) | Same host | Demanding workloads pt. 3: one small real C++ project through the whole BuildDigest loop (`cmake_profile` cold+warm, `ninja_log` cold+incremental, `clang_time_trace`) — cold configure confirmed compiler-detection-dominated vs warm surfacing the project's own CMakeLists costs, a template-heavy TU dominating the ninja ranking 5× over the next edge with its cost attributed to two independently-named clang phases (a single recursive-constexpr call at 37% of compile time, plus STL instantiation pressure), a `compare_metrics` cold-vs-incremental delta proving non-rebuild via literal zeros, a 10.7× nesting-overlap gap on `InstantiateClass`, an honest `.ninja_log` compression floor (0.80×, matching the v1.1 `ptxas` floor exactly) alongside 261–5,440× ratios on the cmake/clang lenses, and a `linux_perf` leg skipped honestly (`perf_event_paranoid=4`, unlock command and expected findings documented) |

The first two runs predate `v1.1.0` — they exercise the tier-1 digest tools
(`list_kernels` / `get_metrics` / `expand`) that are unchanged since v1.0.0.
The v1.1 run measures the agent-loop tools on a local main build (PyPI still
served 1.0.0 at measurement time).

## Layout

```
.
├── workloads/    # tiny, self-contained CUDA/C sources with deliberately
│                 # opposite roofline character, used to produce the reports
│                 # below (workloads/README.md has the capture commands)
└── results/      # the evaluation documents themselves
```

## Reproduce

[`workloads/README.md`](workloads/README.md) has the exact capture commands
(`nvcc` + `ncu` for the GPU workload, `cc` + `perf` for the CPU workload,
including the two host-dependent capture gotchas: `LC_ALL=C` for `perf stat -j`,
and the `perf` scope-modifier normalization). Compile, capture, then digest
**through the perfdigest MCP tools** — never read the raw report into the
agent's context; that defeats the point of the tool this repo evaluates.

## CI capture matrix

[`capture-matrix.yml`](.github/workflows/capture-matrix.yml) enacts
"capture where you can, digest anywhere" on real GitHub-hosted runners, against
the **released PyPI package** (pinned — this repo evaluates released behavior):

- **ptxas on a GPU-less VM** — the full capture→digest pipeline runs on a
  runner with no GPU, because ptxas capture needs the toolkit, not the hardware
  (verified: real register/spill numbers, genuine-0.0 semantics).
- **linux perf as a degraded-PMU honesty test** — Azure VMs hide the hardware
  PMU (measured: EPYC 9V74 runner, *every* hardware counter `<not supported>`,
  even the AVX-512 CPU flags masked). The job asserts those digest as
  `not_available_in_this_export`, never a fabricated 0.0.
- **cross-OS digest** — macOS and Windows runners digest the Linux-captured
  reports, proving the tier-1/tier-2 split on hosts that could never capture them.

What CI *cannot* do here, measured not assumed: GPU captures (no GPU on free
runners) and the SIMD counter study (no vPMU) — those run on local hardware.

## What lands here going forward

This repo is the destination for anything the main repo should not carry:

- Future token-wise A/B studies (new models, new workloads, new backends).
- Captured profiler report artifacts — `.ncu-rep`, `.rocprof-csv`, `perf.data`,
  and similar binary/large files. These must never enter the main repo (they are
  gitignored there); this repo is *for* them, so `.csv` / `.ncu-rep` are
  deliberately **not** ignored here (see `.gitignore`).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
