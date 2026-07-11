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

Both runs predate `v1.1.0` (`compare_metrics`, `summarize_report`, the report
cache, the `ptxas` backend) — they exercise the tier-1 digest tools
(`list_kernels` / `get_metrics` / `expand`) that are unchanged since v1.0.0.

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

## What lands here going forward

This repo is the destination for anything the main repo should not carry:

- Future token-wise A/B studies (new models, new workloads, new backends).
- Captured profiler report artifacts — `.ncu-rep`, `.rocprof-csv`, `perf.data`,
  and similar binary/large files. These must never enter the main repo (they are
  gitignored there); this repo is *for* them, so `.csv` / `.ncu-rep` are
  deliberately **not** ignored here (see `.gitignore`).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
