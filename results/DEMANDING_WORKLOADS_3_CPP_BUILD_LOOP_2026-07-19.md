# Demanding workloads, part 3: one C++ project through the whole BuildDigest loop (2026-07-19)

Third installment of the demanding-workloads series. Parts 1
([`DEMANDING_WORKLOADS_1_CUBLAS_INDUCTOR_2026-07-12.md`](DEMANDING_WORKLOADS_1_CUBLAS_INDUCTOR_2026-07-12.md))
and 2
([`DEMANDING_WORKLOADS_2_TORCH_TRAINING_2026-07-19.md`](DEMANDING_WORKLOADS_2_TORCH_TRAINING_2026-07-19.md))
both stress a single GPU-profiler lens. This part is a different shape of
"demanding": not one hard report, but **one small real C++ project observed
through every build-channel lens v1.2 has** — configure cost (`cmake_profile`),
per-target build cost cold *and* incremental (`ninja_log`), in-TU compile-phase
cost (`clang_time_trace`), and runtime cost (`linux_perf`, conditionally). The
question is whether the four digests compose into one coherent build-loop
story, and what each lens costs in tokens.

**Host:** ASUS TUF A14 — AMD Ryzen AI 9 HX 370 (24 threads), Ubuntu 26.04 LTS,
kernel 7.0.0-27-generic — same host as the rest of the series.
**Toolchain:** `clang++-22` 22.1.6 (the only compiler used for every capture —
`-DCMAKE_CXX_COMPILER=clang++-22`), `cmake` 4.3.2, `ninja` 1.13.2, `perf` 7.0.6.
`gcc` 15.2.0 is present on the host but was not used for any capture.
**Subject:** **`perfdigest-mcp==1.2.0`**, the released PyPI package, installed
fresh into a scratch venv, no extras (every backend used here — `cmake_profile`,
`ninja_log`, `clang_time_trace` — is pure Python). Digests were produced by
calling the functions in `perfdigest.server.tools` directly (`list_kernels`,
`summarize_report`, `get_metrics`, `compare_metrics`, `expand`) after running
`perfdigest.server.app._register_backends()`, exactly as `main()` does before
`mcp.run()` — same method as part 2. Payload bytes are
`len(json.dumps(result, separators=(",",":")))`; tokens ≈ bytes / 4, same
convention as [TOKEN_AB_V1.1_2026-07-11.md](TOKEN_AB_V1.1_2026-07-11.md).
**Project:** [`workloads/cpp_project/`](../workloads/cpp_project/) (new this
run).

---

## The project

A small but real CMake/ninja/Clang project, purpose-built to give every
BuildDigest lens something genuine to measure:

- `CMakeLists.txt` — `cmake_minimum_required(VERSION 3.18)`, two targets
  (`bl_core` static lib, `bl_bench` executable), `find_package(Threads
  REQUIRED)`, and an `include()`d helper (`cmake/BuildLoopHelpers.cmake`,
  one `function(bl_apply_target_defaults tgt)`).
- Five TUs: `lib_alpha.cpp` (numeric reduction), `lib_beta.cpp` (string
  utilities), `lib_gamma.cpp` (a fixed random permutation), and one
  deliberately expensive TU, `lib_template_heavy.cpp`, plus `main.cpp`.
- **`lib_template_heavy.cpp`** combines three independent, genuine compile-time
  cost sources, calibrated by hand against this host (see the file's own
  header comment for the full rationale):
  1. **Recursive constexpr:** naive non-memoized `fib(27)` (~635,621 calls,
     recursion *depth* only ~27 — safely under clang's default
     `-fconstexpr-depth=512` — but call *count* blows past the default
     `-fconstexpr-steps=1,048,576`, so that flag is raised to 32M for this
     one file, Clang-only, guarded in `CMakeLists.txt`).
  2. **Wide STL instantiation pressure:** a 64-alternative `std::variant`
     and a 40-way `std::tuple` of `std::map<std::string,
     std::vector<Row<I>>>`.
  3. **An 8-deep recursively nested `std::tuple` chain** (each level wraps
     the previous type in `std::tuple<prev, Wrap<int, N>>`). Deeper chains
     (tried up to 500 while calibrating) blow up both compile time *and*
     the `-ftime-trace` JSON size, since each `InstantiateClass` event's
     `detail` arg carries the full nested type name — 8 was chosen
     specifically to keep the trace artifact small while `fib(27)` alone
     already clears the ≥1s target.
- `main.cpp` runs a measurable runtime hot loop on **two threads**
  (the actual reason `find_package(Threads)` is in the project): a
  cache-unfriendly gather pass (a fixed `mt19937`-seeded permutation
  indexing 20M times into an ~8M-double / 64 MB pool, far past L3) and a
  compute-bound pass (`sin`/`cos`/`sqrt` reduction). Every TU's output is
  folded into one checksum printed to **stderr** (anti-DCE); the elapsed
  time goes to stdout.

## Captures

| capture | bytes (scrubbed, committed) | units |
|---|--:|--:|
| `configure_cold.cmake-profile.json` (fresh `build_cold/`, `--profiling-format=google-trace`) | 1,544,826 | 90 |
| `configure_warm.cmake-profile.json` (re-configure, same build dir) | 297,280 | 55 |
| `cold.ninja_log` (full cold build) | 565 | 7 |
| `incremental.ninja_log` (after touching `lib_template_heavy.cpp`) | 776 | 7 (10 raw lines, 3 superseded) |
| `template_heavy.ftime-trace.json` (standalone `clang++ -ftime-trace` on the heavy TU) | 745,515 | 189 |

All five are committed under
[`results/cpp_build_loop_raw/`](cpp_build_loop_raw/) — see **Scrub note**
below. `linux_perf` captured nothing (see the dedicated section near the
end): `perf_event_paranoid` was checked immediately before that leg and was
still 4.

**Scrub note:** the cmake-profile traces embed absolute `location`/
`functionArgs` paths, and the clang time-trace embeds an absolute `detail`
path for the one TU it compiled — all under this host's
`/home/hamza-usta/PerfDigest-MCP-Bench/workloads/cpp_project/...`. Committed
copies had that prefix rewritten to `cpp_project/...` (and
`.../build_cold/...` to `build/...`), the same relativization the main
repo's own `cmake_profile` test fixture uses
(`tests/fixtures/cmake_profile_sample.cmake-profile.json`, `"location":
"proj/CMakeLists.txt:1"`). System paths (`/usr/lib/gcc/...`,
`/usr/share/cmake-4.3/...`) were left untouched — they carry no username.
Verified with `grep -rl "/home/hamza"` / `grep -rl "hamza-usta"` over
`results/cpp_build_loop_raw/` — zero matches. All numbers in this document
were re-measured against the **committed, scrubbed** files (not the
original captures), so they are exactly reproducible from what is in the
repo.

---

## What the digests showed

### 1. cmake configure: cold is compiler detection, warm is your own CMakeLists — confirmed on a real capture

The `cmake_profile` backend's own usage prompt predicts this before any data
is looked at ("a cold first configure is dominated by compiler-detection
try_compile; re-configure profiles show your own CMakeLists costs"). The
capture confirms it on this project, unit for unit.

**Cold** (`summarize_report`, top-5 of 90 units, ranked by `duration_us`):

| unit | duration_us | calls | what it is |
|---|--:|--:|---|
| `configure` | 394,547 | 1 | the whole configure step (expected to rank first — it *is* the step) |
| `project` | 281,971 | 1 | `project(... CXX)` — triggers compiler identification + ABI detection |
| `try_compile` | 207,320 | 2 | two probe compiles CMake runs to determine the compiler |
| `execute_process` | 136,983 | 10 | shelling out during compiler/feature detection |
| `cmake_determine_compiler_id` | 123,088 | 1 | CMake's own compiler-ID module |

Every one of the top 5 (after the whole-step frame) is compiler-detection
machinery — none of it is *our* `CMakeLists.txt`. `total_duration_us` summed
across all 90 units is 2,022,122 µs (2.02s) — nearly **5×** the wall time
CMake itself reported (`-- Configuring done (0.4s)`), the expected
nesting-overlap effect (`configure` contains `project` contains
`try_compile`/`execute_process`/... — the reader's own honesty note, verified
quantitatively here for the first time in this series).

**Warm** (re-configure, same build dir, second `--profiling-output` file):

| unit | duration_us | calls | what it is |
|---|--:|--:|---|
| `configure` | 14,060 | 1 | whole step — 28× cheaper than cold |
| `project` | 8,919 | 1 | compiler already known — just re-verification |
| `include` | 8,740 | 31 | **our** `cmake/BuildLoopHelpers.cmake`, aggregated with CMake's own transitively-included modules (e.g. `Threads`'s dependencies) — `include` groups by command name, not by file |
| `generate` | 3,974 | 1 | the generate step |
| `find_package` | 1,698 | 1 | our `find_package(Threads REQUIRED)` |

Compiler-detection units (`try_compile`, `execute_process`,
`cmake_determine_compiler_id`) are **gone from the top-5 entirely** on warm —
CMake caches them. What replaces them is unambiguously our own
`CMakeLists.txt`: `include` and `find_package`. This is the real, measured
version of the usage prompt's claim, not an assumption.

### 2. ninja cold build: the template-heavy TU dominates the ranking outright

`summarize_report` on `cold.ninja_log`, all 7 units:

| unit | duration_ms |
|---|--:|
| `lib_template_heavy.cpp.o` | **2,053** |
| `main.cpp.o` | 408 |
| `lib_gamma.cpp.o` | 398 |
| `lib_beta.cpp.o` | 328 |
| `lib_alpha.cpp.o` | 259 |
| `bl_bench` (link) | 63 |
| `libbl_core.a` (link) | 24 |

The heavy TU is **5×** the next-hottest edge and accounts for 58% of the
build's total wall time (measured `time ninja`: 2.14s) even though four
other TUs compiled *concurrently* with it on this 24-core box — it is the
single edge ninja's own critical path waits on. `list_kernels` +
`summarize_report` rank it #1 with no manual inspection needed; an agent
asking "why is this build slow" gets the right answer in one call.

### 3. `compare_metrics` on cold vs incremental: an honest zero, and a real domain caveat

After touching `lib_template_heavy.cpp` and rebuilding, `compare_metrics`
was run per-target across `cold.ninja_log` → `incremental.ninja_log`:

| target | duration_ms delta | duration_ms delta_pct |
|---|--:|--:|
| `lib_alpha.cpp.o` | 0.0 | 0.0% |
| `lib_beta.cpp.o` | 0.0 | 0.0% |
| `lib_gamma.cpp.o` | 0.0 | 0.0% |
| `main.cpp.o` | 0.0 | 0.0% |
| **`lib_template_heavy.cpp.o`** | **+19.0** | **+0.93%** |
| `libbl_core.a` (link) | +2.0 | +8.3% |
| `bl_bench` (link) | −1.0 | −1.6% (noise) |

Four of seven targets come back **exactly zero on every metric** — not
"small," literally `0.0`. That is not a coincidence of timing: ninja's log
is append-only, and an untouched output's line is never rewritten, so the
digest is reading the *same bytes* both times. `compare_metrics` is proving
non-rebuild here, which is arguably a more useful signal than a duration
delta would have been. The three touched targets (the recompiled TU plus
its two link steps) show real, small deltas — exactly the supersede
semantics `ninja_log`'s own docstring describes.

**One caveat worth flagging, found by looking at the full delta, not just
`duration_ms`:** the touched targets' `start_ms`/`end_ms` deltas are +19 to
+21ms — *not* because the incremental build ran 19ms later in absolute wall
time, but because `.ninja_log`'s `start_ms`/`end_ms` are relative to **each
ninja invocation's own clock**, not a shared origin. `compare_metrics` diffs
every requested metric uniformly, so it dutifully reports a "delta" on a
field that is not comparable in absolute terms across two separate
invocations. `duration_ms` is the only field in this backend's default core
set safe to diff across builds; an agent trusting `start_ms`/`end_ms` deltas
here would draw a real but meaningless conclusion. Not a tool bug — the
metric genuinely is what it says — but a domain nuance worth documenting
alongside the "under `-j>1` edges overlap" caveat already in the backend's
usage prompt.

### 4. `clang_time_trace` on the heavy TU: two independent, separately-named WHYs — not the one the task guessed

Standalone `clang++-22 -ftime-trace` on `lib_template_heavy.cpp` (same
`-fconstexpr-steps`/`-ftemplate-depth` flags as the real build): 1.727s
wall, `Total ExecuteCompiler [total]` reports 1,707,440 µs — a near-exact
match, the expected cross-check between `time` and the trace's own
top-level span.

| unit | calls | total_time_us (naive sum) | `[total]` bookkeeping | % of ExecuteCompiler |
|---|--:|--:|--:|--:|
| `EvaluateAsInitializer` | **1** | 636,707 | 637,865 | **37.3%** |
| `InstantiateClass` | 412 | 3,496,667 | 326,351 | 19.1% (bookkeeping) |
| `InstantiateFunction` | 255 | 516,233 | 203,364 | 11.9% (bookkeeping) |
| `Backend` (mostly `Optimizer`, `-O2`) | 1 | 426,273 | — | 25.0% |

`EvaluateAsInitializer` fires **exactly once** in this whole TU — there is
only one namespace-scope `constexpr` variable initializer in the file,
`kFibResult`, so this single 636.7ms span is unambiguously the `fib(27)`
evaluation. **Over a third of this file's total compile time is one
constexpr call.** `InstantiateClass` (the variant/map/tuple pressure) is the
*other* independent cost source, visible as a completely separate named
phase, exactly as designed.

Directly answering the framing question this study set out to ask (does the
trace blame `InstantiateFunction`?): **only partly.** The dominant named
phase is `EvaluateAsInitializer` (constexpr), not `InstantiateFunction` — a
finding, not an assumption confirmed. `InstantiateFunction` and
`InstantiateClass` are real secondary contributors but the trace's own
vocabulary correctly separates "evaluating a constexpr value" from
"instantiating a template" even though both stem from the same handful of
lines of source. An agent reading only the unit names, with no source
access, would still correctly diagnose *two* distinct expensive mechanisms,
not one.

**A sharp, quantified nesting-overlap case, worse than anything documented
in this series so far:** `InstantiateClass`'s naive per-call sum
(3,496,667 µs) **overstates** its own `[total]` bookkeeping figure
(326,351 µs) by **10.7×** — the widest gap between "sum the units yourself"
and "trust the `[total]` tag" measured yet (compare `Frontend`, a shallow
2-call phase, where the two agree to within 1 µs: 1,271,492 vs
1,271,491). `InstantiateFunction` shows the same pattern at 2.5×
(516,233 vs 203,364). This is the reader's own "spans nest, not additive"
caveat, no longer theoretical — an agent doing its own arithmetic on raw
`total_time_us` instead of trusting the `[total]`-tagged unit would be off
by an order of magnitude on this file.

`expand(kernel="InstantiateClass", section="detail")` returns
`{"arg:detail": "std::basic_string<char>"}` — the *first* `InstantiateClass`
event in file order, per the reader's documented behavior (`raw_metrics`
returns the aggregate stats plus the **first** event's args), not
necessarily the biggest one (`max_time_us` on the grouped unit was
149,155 µs, almost certainly one of the wide-variant/40-map/nested-tuple
instantiations, not `basic_string`). Real, documented behavior, not a bug —
worth knowing before reading too much into a single `expand` call on a
name that groups hundreds of distinct call sites.

### Cross-lens synthesis

- **Does the template-heavy TU dominate the ninja ranking?** Yes, outright
  (finding 2) — 5× the next edge, 58% of build wall time.
- **Does its `clang_time_trace` explain why?** Yes, and with more precision
  than guessed going in: not one blamed phase but two independently named
  ones (`EvaluateAsInitializer` = the constexpr `fib`, `InstantiateClass` =
  the STL pressure) — the multi-lens design paid off exactly as intended,
  turning "this file is slow" (ninja) into "here are the two separate
  reasons" (clang trace).
- **Does configure cost concentrate in compiler detection on cold vs. your
  own CMakeLists on warm?** Yes, confirmed unit-for-unit (finding 1) — not
  merely asserted by the backend's usage prompt, verified on a real
  two-capture pair.

---

## Token cost (measured)

| payload | bytes | ≈ tokens | ratio vs raw artifact |
|---|--:|--:|--:|
| `cmake_cold`: `summarize_report` top-5 | 1,193 | ~298 | **1,295×** vs raw (1,544,826 B) |
| `cmake_cold`: `get_metrics("project")` | 284 | ~71 | **5,440×** vs raw |
| `cmake_cold`: `list_kernels` (all 90) | 7,701 | ~1,925 | 201× vs raw |
| `cmake_warm`: `summarize_report` top-5 | 1,141 | ~285 | **261×** vs raw (297,280 B) |
| `cmake_warm`: `get_metrics("project")` | 278 | ~70 | 1,069× vs raw |
| `ftime_trace`: `summarize_report` top-5 | 1,250 | ~312 | **596×** vs raw (745,515 B) |
| `ftime_trace`: `get_metrics("InstantiateClass")` | 308 | ~77 | 2,421× vs raw |
| `ftime_trace`: `expand(..., section="detail")` | 267 | ~67 | 2,792× vs raw |
| `ftime_trace`: `list_kernels` (all 189) | 19,182 | ~4,796 | 39× vs raw |
| `ninja_cold`: `list_kernels` (all 7) | 703 | ~176 | **0.80×** vs raw (565 B) |
| `ninja_cold`: `summarize_report` top-5 | 1,144 | ~286 | **0.49×** vs raw |
| `ninja_incremental`: `list_kernels` (all 7) | 703 | ~176 | 1.10× vs raw (776 B) |
| `compare_metrics` (avg of 7 ninja targets) | 604 | ~151 | 2.22× vs raw (both logs, 1,341 B) |

**IMPORTANT honesty point, exactly as this artifact class was expected to
behave going in:** `.ninja_log` is already terse tab-separated text, so
there is nothing to compress. `list_kernels` on `cold.ninja_log` lands at
**0.80×** — the exact same number as the celebrated `ptxas` floor in
[TOKEN_AB_V1.1](TOKEN_AB_V1.1_2026-07-11.md) (coincidence, not a rounding
artifact — 703 B digest vs 565 B raw). `summarize_report` is *worse*, 0.49×
(1,144 B digest vs 565 B raw): its extra structural fields
(`total_duration_us`, `coverage_pct_of_total_duration`, `sorted_by`) are a
fixed JSON-shape tax that a 565-byte file is too small to amortize — the
fancier one-call tool costs *more* than the plain listing on an artifact
this tiny. This is a contract-not-compression case, reported plainly, the
same honest framing the `ptxas` floor got in v1.1: perfdigest's value on
`.ninja_log` is the **normalized cross-backend vocabulary**
(`duration_ms`/`start_ms`/`end_ms` next to the same shape from every other
`build_step`/`build_phase` backend), not raw byte savings.

The `cmake_profile` and `clang_time_trace` ratios land exactly where
precedent predicted (~40×+ for a full listing, hundreds-to-thousands-fold
for a single-unit `get_metrics`/`expand` call) — both backends reuse the
same Chrome-trace-shaped, verbosely-argumented JSON that made the torch
traces expensive in parts 1–2, and the same digest mechanics pay off the
same way here.

---

## The `linux_perf` leg: skipped, honestly

`/proc/sys/kernel/perf_event_paranoid` was checked at study start (**4**)
and again **immediately before** this leg (still **4**, unchanged — the
owner did not lower it mid-study). At paranoid=4:

- `perf stat -j -- ./bl_bench` **runs** (exit 0) but every hardware counter
  reads `<not supported>` — the program executes and prints its own
  checksum/timing correctly, but perfdigest would have nothing but honest
  absence to digest.
- `perf record -e cycles:u -g -- ./bl_bench` **fails outright** (exit 255),
  citing the setting explicitly: `perf_event_paranoid setting is 4: ...
  Failure to open any events for recording.`

Per the study's own instruction, this leg is skipped rather than faked.
**Unlock command:** `sudo sysctl kernel.perf_event_paranoid=2`. **What it
would have measured:** `perf stat -j` program counters (IPC, cache-miss
rate, branch-miss rate) on `bl_bench`, digested via `get_metrics(format=
"perf-stat-json")`, contrasting the cache-unfriendly gather pass against the
compute-bound trig/sqrt pass (the two threads main.cpp spawns are built
specifically to make that IPC/cache-miss contrast the interesting story);
and `perf record -g` + `perf report --stdio --no-children -g none` digested
via `summarize_report(format="perf-report")`, expected to rank
`cache_unfriendly_pass`/`alpha_reduce` as the hottest symbols by self time.
Both flows are already proven on this same host and backend in
[`CROSS_BACKEND_2026-06-15.md`](CROSS_BACKEND_2026-06-15.md) and
[`TOKEN_AB_V1.1_2026-07-11.md`](TOKEN_AB_V1.1_2026-07-11.md) — the gap here
is a host-permission gap, not a backend gap.

---

## Limitations

- The `linux_perf` leg produced no data (see above) — every finding in this
  document is from the three build-time lenses only.
- `configure_cold.cmake-profile.json` (1.54 MB) is the largest artifact
  committed in this study. It is "small" only relative to this series' true
  outliers (part 2's 33.65 MB `.ncu-rep`, left uncommitted) — it is
  committed here because the task explicitly calls for it and because its
  size is a real, honest measurement (a cold configure of even a
  two-target project emits ~4,500 B/E compiler-detection events with CMake
  4.3.2), not an artifact of anything this project's own `CMakeLists.txt`
  does.
- The heavy TU's three cost sources (constexpr, variant/map pressure,
  nested-tuple chain) are not independently ablated in the capture data —
  `EvaluateAsInitializer` is cleanly attributable to `fib(27)` alone
  (exactly one call), but `InstantiateClass`'s 412 calls mix the
  64-alternative variant, the 40-way map tuple, and the 8-deep chain
  together; the digest cannot and does not separate them further than the
  phase-name granularity clang itself emits.
- `perf`/`ninja`/`cmake` version numbers are this host's installed versions
  at capture time (`cmake` 4.3.2, `ninja` 1.13.2, `perf` 7.0.6); the
  `ninja_log` reader's own docstring notes the installed `ninja` already
  writes log format v7, one version past the v5/v6 the backend's spec
  reference names — both are accepted, confirmed working here.

## Reproduce

```bash
cd workloads/cpp_project

# (a) cmake configure — cold, then warm re-configure of the same build dir
cmake -S . -B build_cold -G Ninja -DCMAKE_CXX_COMPILER=clang++-22 \
  -DCMAKE_BUILD_TYPE=Release \
  --profiling-format=google-trace --profiling-output=configure_cold.cmake-profile.json
cmake -S . -B build_cold \
  --profiling-format=google-trace --profiling-output=configure_warm.cmake-profile.json

# (b) ninja — cold build, copy the log, touch the heavy TU, rebuild, copy again
cmake --build build_cold   # cold
cp build_cold/.ninja_log cold.ninja_log
touch src/lib_template_heavy.cpp
cmake --build build_cold   # incremental
cp build_cold/.ninja_log incremental.ninja_log

# (c) clang time-trace — standalone, same flags CMakeLists.txt applies to this file
clang++-22 -std=c++17 -O2 -Wall -Wextra -I include \
  -fconstexpr-steps=33554432 -ftemplate-depth=1200 \
  -ftime-trace -c src/lib_template_heavy.cpp -o /tmp/lib_template_heavy.o
# -> lib_template_heavy.json next to the object file

# (d) runtime perf (needs kernel.perf_event_paranoid <= 2)
LC_ALL=C perf stat -j -o bl_bench.perf-stat.json -- ./build_cold/bl_bench
perf record -e cycles:u -g -o bl_bench.perf.data -- ./build_cold/bl_bench
LC_ALL=C perf report -i bl_bench.perf.data --stdio -n --no-children -g none > bl_bench.perf.txt
```

Then digest **through `perfdigest.server.tools`** (never read the raw
report into the agent's context):

```text
summarize_report(report_ref="cold.ninja_log", format="ninja-log", top_n=5)
compare_metrics(report_a="cold.ninja_log", report_b="incremental.ninja_log",
                 format="ninja-log", kernel="CMakeFiles/bl_core.dir/src/lib_template_heavy.cpp.o")
summarize_report(report_ref="configure_cold.cmake-profile.json", format="cmake-profile", top_n=5)
get_metrics(report_ref="lib_template_heavy.json", format="clang-time-trace", kernel="InstantiateClass")
```

**Still to come in this series:** Vector-ISA/SIMD CPU counters, Rust cargo
analysis (`criterion`/`cargo_diag` are already v1.2 backends), LLVM
lowering, and — pending an unlocked `perf_event_paranoid` — the runtime
`linux_perf` leg this run had to skip.
