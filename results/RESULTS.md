# perfdigest A/B benchmark — context efficiency

First measurement of the central thesis: perfdigest replaces the raw profiler
flood with a small, structured signal **without losing diagnostic accuracy**.

## Setup

- **Subject model:** Claude Sonnet 4.6 (`claude-sonnet-4-6`), via Claude Code
  headless (`claude -p --output-format json`). A mid-tier model is the right
  subject — a top model can parse raw CSV anyway and masks the toolkit's value.
- **Workload:** `test_script/kernels.cu` (GAFIME fused map-reduce kernels),
  compiled to `gafime_bench.exe`, profiled with
  `ncu --set full -o gafime.ncu-rep` on an RTX 4060 Laptop (sm_89). 4 kernel
  launches (`gafime_global_continuous_kernel` at arity 1/2/3/5).
- **Identical task, both arms:** "find the highest-duration kernel; decide
  memory- vs compute-bound; name the single biggest limiter with metric
  values; 3 sentences."
- **ON arm:** only the perfdigest MCP tools (`list_kernels`, `get_metrics`,
  `expand`). **OFF arm:** only the `Read` tool over `ncu --page details` text
  (the realistic "agent reads the raw output" baseline). The `.ncu-rep` is
  binary, so the ON arm cannot bypass the tools.

## Result — both arms reached the SAME correct diagnosis

> Longest kernel `gafime_global_continuous_kernel<5>` ≈ 10.08 ms, **memory-bound**:
> DRAM ≈ 96.9% of peak vs compute ≈ 28.0%; limiter = DRAM bandwidth saturation
> with sub-27% L1/L2 hit rates.

Accuracy is not the differentiator — **context cost is**.

## Context cost — profiler payload entering the model per turn

The honest metric is the profiler-derived payload the model must ingest, not
the fixed Claude Code system-prompt overhead (identical in both arms) nor the
single-shot token total (dominated by that overhead, and inflated in the ON
arm purely because tool calls add turns and cache-reads accumulate per turn).

| Payload the agent ingests              | Bytes   | ~Tokens\* | vs digest |
| -------------------------------------- | ------- | --------- | --------- |
| perfdigest `list_kernels`+`get_metrics`|   1,541 |   ~385    | 1.0x      |
| raw `ncu --page details` (OFF arm)     |  55,158 | ~13,790   | **35.8x** |
| raw `ncu --csv --page raw` (full dump) | 163,809 | ~40,952   | **106x**  |

\* ~4 chars/token estimate (no API key for `count_tokens` in this env).
Corroborated by Claude Code's own usage accounting: the OFF arm's file-read
turn created **13,467** cache-input tokens (≈ the ~13,790 estimate), while the
ON arm's tool-result turn created **582**.

## Why this compounds (the real win)

Per turn, perfdigest swaps ~14K tokens of raw noise for ~385 tokens of dense
signal. An agent optimizing a kernel runs measure→edit→measure many times; the
raw output is **permanent and stacks every turn**, while the digest stays
small. The single-shot ratio (≈36x vs the human details page, ≈106x vs the CSV
an agent might `cat`) is the per-turn floor; the cumulative gain over a real
optimization loop is larger. Secondary observation: even single-shot, the OFF
arm cost more (\$0.153 vs \$0.105) — raw context is not free even once.

## Reproduce

The original CUDA workload (GAFIME kernels) has been removed from the repo, but
the ratio reproduces from any `--set full` report — one kernel is enough:

```bash
# profile any CUDA app
ncu --set full -o report.ncu-rep ./your_app

# raw payload an agent would read, vs the perfdigest digest:
ncu --import report.ncu-rep --page details | wc -c     # ~13.6 KB per kernel of raw noise
# vs list_kernels + get_metrics over the same report   # ~1 KB dense signal (≈14–130x fewer tokens)
```

Re-measured 2026-06-13 on an RTX 4060 (sm_89), single kernel: raw details
13,615 B (~3,404 tok) and raw `--csv` dump 124,344 B (~31,086 tok) vs a 963 B
(~241 tok) digest — i.e. 14x and 129x.
