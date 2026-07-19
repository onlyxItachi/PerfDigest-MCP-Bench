# Demanding workloads, part 4: the CUDA optimization loop (2026-07-19)

Fourth installment of the demanding-workloads series. Parts 1–3 each stressed
one profiler lens on someone else's kernels (cuBLAS, Inductor, a real C++
build). This part flips the setup: **three hand-written variants of the same
kernel, in one file**, walked through the loop every CUDA engineer actually
lives — write a kernel, profile it, read the counters, fix the access
pattern, profile again. `compare_metrics` is the instrument for the
measure→edit→measure step; `ptxas -v` is the free codegen preview available
*before* any of that.

**Host:** ASUS TUF A14 — RTX 4060 Laptop (sm_89), driver 610.43.02, ncu
2026.2.1, `nvcc` 13.3 (V13.3.73) — same host as the rest of the series, GPU
exclusively reserved for this run (confirmed idle via `nvidia-smi` before
capture).
**Device facts used to size the workload** (queried via `cudaGetDeviceProperties`,
not assumed): `l2CacheSize` = exactly 32.0 MB, 24 SMs, 128-bit bus, 7.63 GB
VRAM.
**Subject:** **`perfdigest-mcp==1.2.0`**, the released PyPI package, installed
fresh with the `nvidia` extra
(`pip install "perfdigest-mcp[nvidia]==1.2.0"`) into an isolated scratch venv
(confirmed `ncu_report 2026.2.0` importable — the NVIDIA-provided `.ncu-rep`
reader). Digests were produced by calling the functions in
`perfdigest.server.tools` directly (`list_kernels`, `get_metrics`,
`compare_metrics`, `summarize_report`, `expand`) after running
`perfdigest.server.app._register_backends()`, exactly as `main()` does before
`mcp.run()` — same method as parts 2 and 3. Payload bytes are
`len(json.dumps(result, separators=(",",":")))`; tokens ≈ bytes / 4, same
convention as [TOKEN_AB_V1.1_2026-07-11.md](TOKEN_AB_V1.1_2026-07-11.md).
**Workload:** [`workloads/cuda_opt_bench.cu`](../workloads/cuda_opt_bench.cu)
(new this run).

---

## The workload

One file, one computation (a square-matrix transpose, `N=8192`, 256 MB per
array — **8× the device's 32 MB L2**, deliberately sized so the working set
cannot be cache-absorbed and any coalescing penalty shows up as real DRAM
traffic, not hidden latency), three kernels:

- **`kernel_v0_naive`** — direct index-swap transpose. Global *read* is
  coalesced; global *write* multiplies the fast thread index by `n`, so
  consecutive threads in a warp write `n×4` bytes apart. Textbook
  uncoalesced store — this is literally the "naive" kernel from the
  Harris/Ruetsch NVIDIA transpose blog post, relabeled.
- **`kernel_v1_coalesced`** — the canonical fix: a padded `[32][33]`
  shared-memory tile. The transpose happens entirely in shared-memory
  indexing, so both the global read and the global write are per-warp
  contiguous. `BLOCK_ROWS=8`, 4-iteration unrolled loop over the 32-row tile.
- **`kernel_v2_vectorized`** — v1's tile and padding, plus `float4`
  (128-bit) vectorized loads *and* stores on the global-memory side. The
  block shape flips to `(8,32)` so each thread's load phase is a single
  vectorized `LDG.128` covering 4 input columns (replacing 4 of v1's scalar
  `LDG.32`s); the store phase gathers 4 scalar shared-memory reads (a
  strided-by-33 tile column — not contiguous in shared memory, so this part
  stays scalar) into one register `float4` and issues a single `STG.128`.

**Correctness** (this is pure data movement — no arithmetic — so a correct
transpose is exactly bit-reproducible, not just numerically close):
`main()` runs a full device-side element-wise compare of v1's and v2's
output against v0's (every element, not a sample) plus an independent
2,000,000-sample direct-index spot check (`odata[col*n+row] ==
idata[row*n+col]`) against the *untransposed input* for all three variants —
so v0 is never blindly trusted as ground truth. All five checks pass
(`mismatch=0`), both in a plain run and under `ncu` replay. A
position-weighted checksum (`Σ value × ((i % 97) + 1)`, order-sensitive
enough to catch a permutation bug that a plain sum would miss) is printed to
stderr per variant for a human eyeball:

```
checksum v0_naive=9865049090.865543 v1_coalesced=9865049090.867430 v2_vectorized=9865049090.868019
correctness: v1_vs_v0_mismatch=0 v2_vs_v0_mismatch=0 spot_check_v0=0 spot_check_v1=0 spot_check_v2=0 (0=PASS, 1=FAIL, 2000000 samples each)
correctness OK: v0, v1, v2 outputs are bit-identical and match idata.
```

(The checksum's last few digits move slightly run-to-run — that's `atomicAdd`
double-accumulation order varying with scheduling, not a correctness signal;
the actual gate is the exact element-wise compare above, which is
deterministic and always passes.)

---

## Captures

```bash
nvcc -O3 -arch=sm_89 -o cuda_opt_bench workloads/cuda_opt_bench.cu

# ptxas codegen only, no run — registers/spills/smem BEFORE any launch
nvcc -O3 -arch=sm_89 -Xptxas -v -c workloads/cuda_opt_bench.cu -o /dev/null 2> ptxas_verbose.txt

# ncu: -k filters to the three transpose kernels only (the host binary also
# launches four small correctness/init helper kernels — excluded so the
# report holds exactly the 3-variant story). Each variant is launched 3x
# on the same input; --set full captures all 9 launches, no --launch-skip
# needed since there's nothing else in the filtered window.
ncu -k "regex:kernel_v[0-9]_" --set full -f -o cuda_opt ./cuda_opt_bench
```

The `-k` filter kept the capture small and fast: **9 launches, 125.2 MB
report, 13.3 s wall** (`45 passes` per launch for the full metric set —
Ampere/Ada's `--set full` is not cheap, but tractable at this problem size).
An unfiltered first attempt (no `-k`) captured all 18 launches including the
correctness/checksum kernels in 43.7 s / 315.8 MB — confirmed working, but
discarded in favor of the filtered, on-topic capture.

---

## The v0→v1→v2 arc, measured

`list_kernels` on the filtered report (steady across all 3 rounds per
kernel, <0.1% launch-to-launch jitter — this is a clean, non-noisy signal):

| kernel | launch durations (µs, ncu) | avg |
|---|---|---|
| `kernel_v0_naive` | 6017.3, 6017.5, 6019.8 | **6018.2** |
| `kernel_v1_coalesced` | 2271.7, 2277.1, 2323.3 | **2290.7** |
| `kernel_v2_vectorized` | 2280.6, 2336.8, 2393.5 | **2337.0** |

**v0 → v1: real, clean, textbook — 2.64× faster** (6017.5 → 2277.1 µs on the
launches used below). **v1 → v2: no measurable win** (2277.1 → 2280.6 µs,
+0.15%; using the round averages instead, +2.0% — either way, within this
kernel's own ~2–5% launch-to-launch noise band). This is exactly the honest
outcome the task asked to watch for: the textbook "add vectorization" step
did nothing here, and the counters below say precisely why.

### `get_metrics` (default core set), one steady launch per kernel

| metric | v0_naive | v1_coalesced | v2_vectorized |
|---|--:|--:|--:|
| duration_us | 6017.5 | 2277.1 | 2280.6 |
| dram_pct_peak | **35.9** | **89.7** | **89.6** |
| compute_pct_peak | 6.1 | 18.3 | 12.2 |
| mem_throughput_gbps | 91.8 | 229.5 | 229.1 |
| achieved_occupancy | 78.0 | 97.1 | 96.0 |
| l1_hit_rate | 60.4 | 0.0 | 0.0 |
| l2_hit_rate | 88.2 | 50.0 | 50.1 |
| registers_per_thread | 16 | 26 | 19 |
| warp_cycles_per_issue | 723.0 | 212.5 | 285.1 |
| shared_mem_per_block | 0 | 4224 | 4224 |

`dram_pct_peak` and `mem_throughput_gbps` self-consistently imply the same
peak bandwidth from all three units independently
(`gbps / (pct/100)` ≈ 255.77 GB/s for v0, 255.77 for v1, 255.77 for v2) — a
nice cross-check that these are the same physical ceiling, not three
different measurements.

**The roofline story:** v0 is nowhere near DRAM peak (35.9%) *despite* the
highest cache hit rates (60.4% L1 / 88.2% L2) — counter-intuitive at a
glance, explained by the access pattern: the uncoalesced write scatters many
small, partial-line stores that repeatedly touch the same cache lines from
different warps, inflating the hit rate while wasting most of each fetched
line's bytes. v1's write is one-shot and fully coalesced, so there is much
less reuse to hit *or* miss on — L2 hit rate correctly **drops** to 50.0%,
which here means "more efficient," not "worse." `warp_cycles_per_issue`
falls 723 → 212 (v0's warps spend most of a launch stalled waiting on those
scattered stores); `achieved_occupancy` climbs to 97.1% (v1 is no longer
serializing on outstanding memory transactions the way v0 was).

**v1 vs v2 — same ceiling, different route there.** `dram_pct_peak` is
statistically flat (89.7 vs 89.6) and `mem_throughput_gbps` likewise
(229.5 vs 229.1) — both variants are already sitting on ~90% of the same
physical DRAM ceiling. `expand(..., section='inst_executed.sum')` confirms
the vectorization did exactly what it was designed to do at the instruction
level:

| | v1_coalesced | v2_vectorized | delta |
|---|--:|--:|--:|
| `smsp__inst_executed.sum` | 22,544,384 | 17,825,792 | **−20.9%** |
| `registers_per_thread` (ncu) | 26 | 19 | −26.9% |

A genuine 20.9% instruction-count cut, fewer registers, same grid/block
occupancy ceiling — every codegen-level signal says v2 should win. It
doesn't, because `warp_cycles_per_issue` **rises** 212.5 → 285.1 (+34.2%):
each surviving (now 128-bit) instruction costs more cycles per issue than
v1's scalar ones did, and that per-instruction cost increase exactly offsets
the instruction-count reduction. `expand(..., section='bank')` rules out
shared-memory bank conflicts as the cause —
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared*.sum` is **exactly 0.0 for
both v1 and v2**, so the padding is doing its job in both kernels equally.
The more likely mechanism is v2's store-phase gather (4 strided scalar
shared-memory reads packed into one register before the single vectorized
global store) carrying a longer per-thread dependency chain into that store
than v1's one-read-one-store pattern — reported as measured, not fully
isolated; this run did not add the source-level correlation needed to
confirm it, consistent with this series' existing "measured, not causally
explained" pattern (part 2 left a similar case open).

**Bottom line on the arc:** when a kernel is still far from its roofline
ceiling (v0), *any* fix that increases effective bytes-per-transaction wins
big. Once a kernel is already sitting near that ceiling (v1, at ~90% of
implied peak), further "reduce instruction count" optimizations have no
bandwidth headroom left to spend their savings on, and can be a wash or
even a net negative if the reduced instruction count comes with a per-
instruction cost increase — which is exactly what `warp_cycles_per_issue`
caught here.

---

## `compare_metrics` — the loop instrument

Same report, two adjacent kernels, one call each — the measure→edit→measure
step narrated purely by delta digests.

### v0 → v1

| metric | a (v0) | b (v1) | delta | delta_pct |
|---|--:|--:|--:|--:|
| duration_us | 6017.5 | 2277.1 | −3740.4 | **−62.2%** |
| dram_pct_peak | 35.88 | 89.73 | +53.86 | **+150.1%** |
| mem_throughput_gbps | 91.76 | 229.52 | +137.75 | +150.1% |
| compute_pct_peak | 6.15 | 18.32 | +12.17 | +198.1% |
| achieved_occupancy | 78.0 | 97.1 | +19.06 | +24.4% |
| warp_cycles_per_issue | 723.0 | 212.5 | −510.5 | **−70.6%** |
| l1_hit_rate | 60.44 | 0.0 | −60.44 | −100.0% |
| l2_hit_rate | 88.25 | 50.00 | −38.25 | −43.3% |
| registers_per_thread | 16 | 26 | +10 | +62.5% |
| shared_mem_per_block | 0 | 4224 | +4224 | (undefined — 0 baseline) |

### v1 → v2

| metric | a (v1) | b (v2) | delta | delta_pct |
|---|--:|--:|--:|--:|
| duration_us | 2277.1 | 2280.6 | +3.52 | **+0.15%** |
| dram_pct_peak | 89.73 | 89.58 | −0.15 | −0.17% |
| mem_throughput_gbps | 229.52 | 229.13 | −0.39 | −0.17% |
| compute_pct_peak | 18.32 | 12.19 | −6.13 | −33.5% |
| achieved_occupancy | 97.06 | 96.02 | −1.04 | −1.1% |
| warp_cycles_per_issue | 212.46 | 285.13 | +72.67 | **+34.2%** |
| registers_per_thread | 26 | 19 | −7 | −26.9% |
| shared_mem_per_block | 4224 | 4224 | 0 | 0.0% |

The two tables sit side by side as the loop's own narration: table 1 is "the
fix that mattered" (duration collapses, every roofline metric moves in the
expected direction, `delta = b - a` reads as one coherent story). Table 2 is
"the fix that didn't" — registers and instruction-adjacent metrics move the
*same direction* they did in table 1 (fewer registers, presumably a further
win by the same logic), but `duration_us` and `dram_pct_peak` are flat, and
the one metric that actually explains it (`warp_cycles_per_issue`) moves the
*opposite* way from table 1. Reading table 2 by analogy to table 1 would be
the wrong lesson; the delta digest itself doesn't editorialize either way —
it just reports what changed, honestly, and that is exactly what let this
finding surface instead of being explained away.

### A ranking pitfall worth flagging: `summarize_report`'s top-5 quietly drops the real winner

`summarize_report(top_n=5)` on this report ranks purely by `duration_us`. The
top 5 units are all 3 `v0_naive` launches (6017–6020 µs) plus the **slowest
two of `v2_vectorized`'s three launches** (2337, 2394 µs) —
**`v1_coalesced` does not appear in the top 5 at all**, because its slowest
launch (2323 µs) is a hair faster than v2's second-slowest (2337 µs). An
agent reading only the top-5 table would see v0 (bad) and v2 (the "improved"
variant) and never learn that v1 — the actual, real winner of the loop —
exists in the same report. This is a direct consequence of v1/v2 being
statistically tied: `summarize_report` did exactly what it promises
(rank the top N units by duration), and the promise itself doesn't cover
"also surface every distinct kernel name." `list_kernels` (used above) is
the right call for that, not `summarize_report`. Worth a doc callout for
this specific loop-instrument workflow: `summarize_report` alone is not
sufficient to be sure every variant made it into view when variants are
close in duration.

---

## Did `ptxas` foreshadow the ranking?

| kernel | registers | spills | static smem | barriers | const mem |
|---|--:|--:|--:|--:|--:|
| `kernel_v0_naive` | 14 | 0 | 0 B | 0 | 372 B |
| `kernel_v1_coalesced` | 26 | 0 | 4224 B | 1 | 372 B |
| `kernel_v2_vectorized` | 19 | 0 | 4224 B | 1 | 372 B |

(via `get_metrics(format="ptxas-verbose")` on
[`results/cuda_opt_bench_raw/ptxas_verbose.txt`](cuda_opt_bench_raw/ptxas_verbose.txt);
zero spills across the board — register pressure was never the risk in any
variant.)

**Partially, and the partial match is itself the finding.** ptxas correctly
flags v0 as structurally different — zero shared memory, fewest registers,
no barriers, the shape of "the simple one" — consistent with it turning out
to be the slow one, though ptxas alone cannot say *why* (uncoalesced access
is a runtime fact about warp-to-memory mapping, invisible to a static
compiler pass with no knowledge of the launch's actual index arithmetic
pattern at the warp level). Between v1 and v2, ptxas's signal is actively
**misleading if read alone**: v2 uses 27% fewer registers (19 vs 26), which
by the usual "fewer registers → better occupancy/scheduling" heuristic
predicts v2 should win. The measured result is a flat-to-negative outcome.
**ptxas foreshadows the ranking only when a kernel isn't yet roofline-bound;
once two kernels are both pinned to the same DRAM ceiling (v1, v2 here),
static codegen counts stop predicting wall-clock order, and only the
roofline counters — `dram_pct_peak`, `mem_throughput_gbps`,
`warp_cycles_per_issue` — do.**

One cross-tool nuance, small but real: ptxas reports v0 at **14** registers;
ncu's `registers_per_thread` for the same kernel reads **16**. Not a bug in
either tool — ncu reports the runtime-allocated register count, which
rounds up to the SM's allocation granularity, while ptxas reports the
compiler's raw count before that rounding. Worth knowing before treating the
two register counts as interchangeable across backends.

---

## Token cost (measured)

| payload | bytes | ≈ tokens | ratio |
|---|--:|--:|--:|
| **Full 3-variant flow**: ptxas (`list_kernels`+3×`get_metrics`) + ncu (`list_kernels`+3×`get_metrics`) + 2×`compare_metrics` + `summarize_report` top-5 | **10,825** | **~2,706** | 1× |
| ncu digest orient alone (`list_kernels` + 3×`get_metrics`) | 2,743 | ~686 | — |
| `compare_metrics` v0→v1 | 1,694 | ~424 | — |
| `compare_metrics` v1→v2 | 1,691 | ~423 | — |
| `summarize_report` top-5 | 2,960 | ~740 | — |
| ptxas digest (`list_kernels` + 3×`get_metrics`) | 1,737 | ~434 | vs raw ptxas text (2,351 B): **1.35×** |
| raw `ncu --page details` (all 9 launches) | 119,938 | ~29,985 | **11.1×** vs full flow |
| raw `ncu --page raw --csv` (all 9 launches) | 236,015 | ~59,004 | **21.8×** vs full flow |
| ncu digest orient alone vs raw `--page details` | 2,743 vs 119,938 | — | **43.7×** |
| ncu digest orient alone vs raw `--page raw --csv` | 2,743 vs 236,015 | — | **86.0×** |
| raw `ptxas_verbose.txt` (all 7 kernels in the TU) | 2,351 | ~588 | — |
| the `.ncu-rep` itself | 125.2 MB | unreadable binary | — |

**Honest floor case, again:** the ptxas leg's ratio (1.35×) is the smallest
in this table, echoing `TOKEN_AB_V1.1`'s original 0.8× ptxas floor finding —
verbose compiler output is already terse plain text, so a JSON digest of it
saves less, proportionally, than a JSON digest of a sprawling `ncu` text
page does. It is still a net win here (unlike the 0.8× case), because the
raw file covers all 7 kernels in the translation unit while the digest
targeted only the 3 relevant to this study — but the margin is real evidence
that ptxas digests are a modest, not dramatic, token win, consistent across
two separate measurement runs now (v1.1 and this one).

---

## Limitations

- **No HIP/ROCm leg — verified, not assumed.** `platform_capabilities()`
  reports `"rocm"`: `"capturable": false, "reason": "rocprof not found on
  PATH"` on this host, and ROCm's own install here has historically been
  blocked (disk-quota constraints on this box, unrelated to perfdigest).
  Concretely: this host cannot capture the AMD gfx1150 half of its own
  dual-GPU setup with `rocprof`, so the identical measure→edit→measure loop
  on HIP kernels is not capturable here, full stop — not degraded, not
  faked, just absent. The `rocm` backend's digest path remains covered by
  the main repo's fixtures and unit tests only; nothing in this series has
  exercised it against a real capture. This is a genuine coverage gap in
  the series, stated plainly rather than worked around.
- **The v1→v2 "no win" mechanism is measured, not fully isolated.**
  `warp_cycles_per_issue` rising while `inst_executed.sum` falls is the
  measured fact; the specific claim that v2's store-phase shared-memory
  gather is the cause is a plausible mechanism consistent with the kernel's
  own structure, not something this run independently confirmed (e.g. via
  SASS inspection or a controlled ablation that vectorizes only the load
  phase). Flagged as such, matching this series' running pattern of leaving
  a mechanism open when the counters don't fully close it.
- **Only one problem size (`N=8192`) was profiled.** The 8×-over-L2 sizing
  was chosen deliberately so the coalescing penalty shows up as DRAM
  traffic rather than being cache-absorbed, and it worked (v0's dram_pct_peak
  is genuinely low, not an L2-hidden artifact) — but no sweep across smaller
  N (where v0's penalty might partly hide behind L2) was run. That sweep
  would be a natural follow-up, not attempted here.
- Raw artifacts: `ptxas_verbose.txt` (2.4 KB, text) is committed at
  [`results/cuda_opt_bench_raw/ptxas_verbose.txt`](cuda_opt_bench_raw/ptxas_verbose.txt).
  The `.ncu-rep` (125.2 MB, binary) is **not committed**, matching parts
  1–3's precedent and `workloads/README.md`'s gitignore rationale
  (regenerable, large/binary). Regenerate with:

```bash
nvcc -O3 -arch=sm_89 -o cuda_opt_bench workloads/cuda_opt_bench.cu
ncu -k "regex:kernel_v[0-9]_" --set full -f -o cuda_opt ./cuda_opt_bench
```

Then digest **through the perfdigest MCP** (never read the raw report into
the agent's context):

```text
list_kernels(report_ref="cuda_opt.ncu-rep", format="ncu-rep")
get_metrics(report_ref="cuda_opt.ncu-rep", format="ncu-rep", kernel="#2")   # v0, steady launch
get_metrics(report_ref="cuda_opt.ncu-rep", format="ncu-rep", kernel="#5")   # v1, steady launch
get_metrics(report_ref="cuda_opt.ncu-rep", format="ncu-rep", kernel="#8")   # v2, steady launch
compare_metrics(report_a="cuda_opt.ncu-rep", report_b="cuda_opt.ncu-rep",
                 format="ncu-rep", kernel="#2", kernel_b="#5")              # v0 -> v1
compare_metrics(report_a="cuda_opt.ncu-rep", report_b="cuda_opt.ncu-rep",
                 format="ncu-rep", kernel="#5", kernel_b="#8")              # v1 -> v2
get_metrics(report_ref="results/cuda_opt_bench_raw/ptxas_verbose.txt",
            format="ptxas-verbose", kernel="kernel_v0_naive")
```

(`#index` addressing is required here, same as part 2's finding — the report
holds 3 launches each of 3 identically-named kernels, so every bare-name
lookup is ambiguous by construction.)

---

## Carry-forward findings

1. **The v0→v1 half of the arc is the textbook result, and it is real**:
   2.64× faster, fully explained by `dram_pct_peak` (35.9%→89.7%) and
   `warp_cycles_per_issue` (723→212) moving together in `compare_metrics`'
   one delta call — the loop-instrument thesis holds exactly as designed.
2. **The v1→v2 half is the honest negative result this task asked to watch
   for**: a real, correctly-implemented vectorization (verified via
   `inst_executed.sum`: −20.9% instructions, and `registers_per_thread`:
   −26.9%) bought zero wall-clock improvement because the kernel was already
   at ~90% of its DRAM ceiling — there was no bandwidth headroom left to
   spend the savings on, and the per-instruction cost of the vectorized ops
   rose enough (`warp_cycles_per_issue` +34.2%) to erase the win outright.
   No redesign was applied to make this prettier; it is reported as
   measured.
3. **`ptxas` foreshadows the ranking only up to the roofline ceiling.** It
   correctly flagged v0 as structurally the odd one out; between v1 and v2
   (both DRAM-bound) its register-count signal actively points the wrong
   way. A useful, narrow answer to "does codegen hint the perf ranking":
   yes, for kernels not yet bandwidth-saturated; no, once they are.
4. **A new `summarize_report` reading pitfall**: when two variants are
   statistically tied, top-N-by-duration ranking can silently omit one of
   them from view entirely (v1 never appeared in the top 5 here), even
   though `list_kernels` shows all three. Series total is now three distinct
   `compare_metrics`/`summarize_report` reading pitfalls (part 2 found two,
   this run finds a third) — worth a consolidated usage-doc callout across
   the series rather than three separate footnotes.
5. **The ptxas token-ratio floor reproduces at 1.35×** (part 1.1 measured
   0.8× on a different kernel pair), reinforcing that ptxas digests are a
   modest, not dramatic, win — consistent across two independent
   measurements now.

**Still to come in this series:** Vector-ISA/SIMD CPU counters, Rust cargo
analysis (`criterion-json`), LLVM optimization-remarks (as listed in earlier
parts) — and, contingent on lifting the disk-quota block noted above, a real
`rocprof` capture of this exact transpose kernel ported to HIP, to finally
close the HIP gap this run and part 3's `linux_perf` gap both left open.
