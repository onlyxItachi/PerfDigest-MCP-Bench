# PerfDigest Coordinated Benchmark Schemas (`v1`)

This directory holds the JSON Schema (draft 2020-12) definitions shared across the PerfDigest repositories (`onlyxItachi/PerfDigest-MCP`, `onlyxItachi/PerfDigest-MCP-Bench`, and `onlyxItachi/perfdigest-plugin`) so benchmark manifests and result documents validate against a single contract without conflating architectural layers.

## Files

| Path | Schema ID (`schema` field) | Purpose |
|---|---|---|
| `benchmark-result.v1.schema.json` | `perfdigest.benchmark-result.v1` | Result document schema: host provenance, tool versions, `layerSummary`, `benchCompatibility`, measured `cells[]`, falsifiable `claims[]`, and non-empty `limits[]`. |
| `benchmark-manifest.v1.schema.json` | `perfdigest.benchmark-manifest.v1` | Manifest schema declaring cell IDs, layers, groups, backends, scenarios, and iteration defaults. |
| `examples/interpreter-matrix-2026-07-18.benchmark-result.v1.json` | `perfdigest.benchmark-result.v1` | Deterministic v1 export of `results/interpreter_matrix_raw/` produced by `ci/export_benchmark_v1.py`. |

## Three-Layer Separation (`cell.layer` & `layerSummary`)

Every cell in a `perfdigest.benchmark-result.v1` or `perfdigest.benchmark-manifest.v1` document carries an explicit `layer` field, and top-level `layerSummary` records per-layer cell and parity counts (`engine`, `orchestration`, `agent_e2e`):

1. **`engine`** (`PerfDigest-MCP` / `PerfDigest-MCP-Bench`):
   - Measures pure profiler-report parsing, digest construction (`load_units`, `build_digest`, `build_summary`), interpreter startup (`import perfdigest.server.app`), `report_store` cache contention, and single-session MCP server startup (`uvx perfdigest-mcp`).
   - Independent of any outer TypeScript/Node/Bun/Deno/Go plugin wrapper or coding agent.
2. **`orchestration`** (`perfdigest-plugin`):
   - Measures the PerfDigest Evidence plugin's own build and runtime surfaces: TypeScript compilation (`tsc` vs `tsgo`), CLI startup, offline demo orchestration (`--agent none`), MCP server tool discovery, child spawn overhead, and receipt hashing workers.
3. **`agent_e2e`** (`perfdigest-plugin`):
   - Measures end-to-end offline optimization/investigation workflows driven through a real `perfdigest-agent/v1` CLI agent process (`--agent generic --agent-sessions`), overlapping analysis turns with profiler capture, MCP digesting, verification, and same-session evidence delivery.

## Memory Accounting Provenance (`memoryCoverage`)

Each cell declares `memoryCoverage` so single-process RSS is never confused with concurrent process-tree peak memory:

- `"cgroup_v2_tree_peak"` — Linux kernel cgroup v2 `memory.peak` (`cgroupTreePeakKb`) across the entire concurrent process tree inside a transient `systemd-run --user --scope` unit, alongside GNU time `peakRssKb` (`ru_maxrss` of the single largest child).
- `"max_single_process_only_incomplete_tree"` — only GNU `/usr/bin/time -f '%M'` (`peakRssKb`) was available.
- `"none"` — memory was not sampled in the underlying study (used by `ci/export_benchmark_v1.py` rather than fabricating RSS numbers).

## Bridging Existing Raw Studies (`ci/export_benchmark_v1.py`)

Existing raw artifacts in `results/*_raw/` are preserved untouched. `ci/export_benchmark_v1.py` bridges `results/interpreter_matrix_raw/` into `schemas/examples/interpreter-matrix-2026-07-18.benchmark-result.v1.json` using the mapping recorded in `benchCompatibility.fieldMapping`:

- `results/interpreter_matrix_raw/<ver>_latency.json` → `cells[layer="engine", group="runtime"].wall` (`unit: "us"`, `median`, `q1`, `q3`, `n`).
- `results/interpreter_matrix_raw/<ver>_identity.json` → `parity.checks[name="digest-sha256:<backend>:<format>"]` (`sha256`, `bytes`, `total_units`) + `extra.identity`.
- `results/interpreter_matrix_raw/<ver>_startup.json` → `cells[layer="engine", scenario="interpreter-startup"].wall` (`unit: "s"`, `median`).
- `results/interpreter_matrix_raw/3.14*_concurrency.json` → `extra.concurrency` and concurrency correctness checks.
- `results/interpreter_matrix_raw/latency_loadavg.json` → `host.loadavg1AtStart`.

### Usage

```bash
# Regenerate the v1 example document from results/interpreter_matrix_raw/
python3 ci/export_benchmark_v1.py

# Verify structural invariants and byte-identical regeneration in CI
python3 ci/export_benchmark_v1.py --check

# Run unit tests
python3 -m unittest discover -s ci -p "test_*.py"
```
