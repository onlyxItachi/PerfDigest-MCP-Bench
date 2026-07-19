"""interpreter_matrix.py — perfdigest across the Python interpreter matrix.

Three legs, driven against a LOCAL perfdigest checkout installed non-editable
into one venv per interpreter (env setup is NOT this script's job — see
"Usage" below for the exact commands the reader is expected to have already
run):

  (i)   digest byte-identity across interpreters (the token-invariance proof)
  (ii)  reader parse/digest latency per version (cache bypassed)
  (iii) 3.14t free-threaded cache-lock contention vs a GIL build

This file is dual-mode: it is spawned as a `worker` INSIDE each venv (so the
measurement happens under that interpreter's real perfdigest import), and it
runs as the `report` orchestrator (any interpreter — it only shells out and
formats) to fan out to every worker and render the three result tables as
markdown to stdout.

Usage
-----
Env setup (once; NOT performed by this script):

    uv python install 3.10 3.11 3.12 3.13 3.14 3.14t
    for v in 3.10 3.11 3.12 3.13 3.14 3.14t; do
        uv venv --python "$v" /path/to/interp-bench/"$v"
        uv pip install --python /path/to/interp-bench/"$v"/bin/python \\
            /path/to/MCP_Developing_REPO   # non-editable, local checkout
    done

Then, from any interpreter with no extra deps (stdlib only):

    python workloads/interpreter_matrix.py report \\
        --venv-dir /path/to/interp-bench \\
        --repo /path/to/MCP_Developing_REPO \\
        --raw-out results/interpreter_matrix_raw \\
        > results/INTERPRETER_MATRIX_2026-07-18.md

A version whose venv is missing, or whose `import perfdigest` fails (e.g. the
package's `requires-python` floor, or a wheel gap on a free-threaded build),
is reported as could-not-run with the captured reason — never silently
skipped and never fabricated.

Worker mode (normally invoked BY the orchestrator, not by hand):

    <venv>/bin/python workloads/interpreter_matrix.py worker \\
        --leg {identity,latency,concurrency} --repo /path/to/MCP_Developing_REPO
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture map — one REAL fixture per pure-Python backend (skip the native
# `ncu-rep` path: it needs the `ncu-report` vendor wheel, out of scope here).
# Paths are relative to <repo>/tests/fixtures/.
# ---------------------------------------------------------------------------
FIXTURES: list[dict[str, str]] = [
    {"backend": "linux_perf", "format": "perf-stat-json", "fixture": "cpu_stat_sample.perf-stat.json"},
    {"backend": "linux_perf", "format": "perf-report", "fixture": "cpu_report_sample.perf-report.txt"},
    {"backend": "ptxas", "format": "ptxas-verbose", "fixture": "gpu_codegen_sample.ptxas.txt"},
    {"backend": "chrome_trace", "format": "torch-trace", "fixture": "torch_sample.torch-trace.json"},
    {"backend": "clang_time_trace", "format": "clang-time-trace", "fixture": "template_heavy.ftime-trace.json"},
    {"backend": "cmake_profile", "format": "cmake-profile", "fixture": "cmake_profile_sample.cmake-profile.json"},
    {"backend": "ninja_log", "format": "ninja-log", "fixture": "ninja_log_sample.ninja_log"},
    {"backend": "cargo_diag", "format": "cargo-diag", "fixture": "cargo_build_warnings_sample.cargo-diag.jsonl"},
    {"backend": "gha_log", "format": "gha-log", "fixture": "ci_test_green_sample.gha.log"},
    {"backend": "git_numstat", "format": "git-numstat", "fixture": "repo_session_sample.numstat"},
    {"backend": "criterion", "format": "criterion", "fixture": "criterion_sample"},
]

# Leg (iii) fixtures: file-based only (a directory-shaped report_ref bypasses
# the cache entirely by design — report_store/cache.py — so it can't exercise
# the lock this leg is measuring). "same_path" is the mid-size single target;
# "mixed_paths" round-robins threads across a small spread of sizes.
CONCURRENCY_SAME_PATH = {"backend": "chrome_trace", "format": "torch-trace", "fixture": "torch_sample.torch-trace.json"}
CONCURRENCY_MIXED_PATHS = [
    {"backend": "ptxas", "format": "ptxas-verbose", "fixture": "gpu_codegen_sample.ptxas.txt"},
    {"backend": "gha_log", "format": "gha-log", "fixture": "ci_test_green_sample.gha.log"},
    {"backend": "ninja_log", "format": "ninja-log", "fixture": "ninja_log_sample.ninja_log"},
    {"backend": "clang_time_trace", "format": "clang-time-trace", "fixture": "template_heavy.ftime-trace.json"},
]

ALL_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t"]
DEFAULT_TOP_N = 5  # mirrors summarize_report's own default — "the real code path"


# ===========================================================================
# Worker-side helpers (only imported/callable once running INSIDE a venv that
# has perfdigest installed; the orchestrator never imports perfdigest itself).
# ===========================================================================

def _digest_bytes(units, fmt: str, backend, top_n: int = DEFAULT_TOP_N) -> bytes:
    """The exact canonical payload the identity/concurrency legs hash."""
    from perfdigest.core.digest import build_summary

    summary = build_summary(units, fmt, list(backend.default_core_set), top_n)
    return json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_backend(name: str):
    from perfdigest.adapters import registry
    from perfdigest.server.app import _register_backends

    _register_backends()
    return registry.get_by_name(name)


def worker_identity(repo: Path) -> dict:
    fixtures_dir = repo / "tests" / "fixtures"
    results = {}
    for entry in FIXTURES:
        backend = _load_backend(entry["backend"])
        path = str(fixtures_dir / entry["fixture"])
        units = backend.load_units(path)
        payload = _digest_bytes(units, entry["format"], backend)
        key = f"{entry['backend']}:{entry['format']}"
        results[key] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "total_units": len(units),
        }
    return {"python_version": sys.version.split()[0], "results": results}


def worker_latency(repo: Path, iters: int = 50, warmup: int = 5) -> dict:
    fixtures_dir = repo / "tests" / "fixtures"
    results = {}
    for entry in FIXTURES:
        backend = _load_backend(entry["backend"])
        path = str(fixtures_dir / entry["fixture"])
        for _ in range(warmup):
            backend.load_units(path)  # cache BYPASSED — direct reader call
        samples_s = []
        for _ in range(iters):
            t0 = time.perf_counter()
            backend.load_units(path)
            samples_s.append(time.perf_counter() - t0)
        samples_us = sorted(s * 1e6 for s in samples_s)
        q1, _, q3 = statistics.quantiles(samples_us, n=4, method="inclusive")
        key = f"{entry['backend']}:{entry['format']}"
        results[key] = {
            "median_us": statistics.median(samples_us),
            "q1_us": q1,
            "q3_us": q3,
            "iqr_us": q3 - q1,
            "n": iters,
        }
    return {"python_version": sys.version.split()[0], "results": results}


def _concurrency_run(repo: Path, entries: list[dict], n_threads: int, iters: int) -> dict:
    """N threads x iters calls to report_store.cache.cached_units, spread over
    `entries` round-robin (len(entries)==1 -> every thread hits the SAME path).
    Correctness = every returned digest hash equals that path's single-
    threaded reference hash — the cache lock's actual job.
    """
    from perfdigest.report_store import cache as report_cache

    fixtures_dir = repo / "tests" / "fixtures"
    report_cache.clear()

    backends = {e["backend"]: _load_backend(e["backend"]) for e in entries}
    paths = {i: str(fixtures_dir / e["fixture"]) for i, e in enumerate(entries)}
    refs = {}
    for i, e in enumerate(entries):
        backend = backends[e["backend"]]
        units = backend.load_units(paths[i])
        refs[i] = hashlib.sha256(_digest_bytes(units, e["format"], backend)).hexdigest()
    report_cache.clear()  # measure from a cold cache — the contended fill included

    barrier = threading.Barrier(n_threads)
    ok = [True] * n_threads
    mismatches = [0] * n_threads

    def run(idx: int) -> None:
        entry_i = idx % len(entries)
        entry = entries[entry_i]
        backend = backends[entry["backend"]]
        path = paths[entry_i]
        ref_hash = refs[entry_i]
        barrier.wait()
        for _ in range(iters):
            units = report_cache.cached_units(backend.name, backend.load_units, path)
            got = hashlib.sha256(_digest_bytes(units, entry["format"], backend)).hexdigest()
            if got != ref_hash:
                mismatches[idx] += 1
                ok[idx] = False

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t0

    total_calls = n_threads * iters
    return {
        "n_threads": n_threads,
        "iters_per_thread": iters,
        "wall_time_s": wall_s,
        "throughput_calls_per_s": total_calls / wall_s if wall_s > 0 else float("inf"),
        "correct": all(ok),
        "total_mismatches": sum(mismatches),
        "total_calls": total_calls,
    }


def worker_concurrency(repo: Path, threads: int = 8, iters: int = 25) -> dict:
    try:
        gil_enabled = sys._is_gil_enabled()  # type: ignore[attr-defined]
    except AttributeError:
        gil_enabled = True  # no toggle API -> GIL always on (pre-3.13)

    same_path_1 = _concurrency_run(repo, [CONCURRENCY_SAME_PATH], 1, iters)
    same_path_n = _concurrency_run(repo, [CONCURRENCY_SAME_PATH], threads, iters)
    mixed = _concurrency_run(repo, CONCURRENCY_MIXED_PATHS, threads, iters)
    return {
        "python_version": sys.version.split()[0],
        "gil_enabled": gil_enabled,
        "same_path_1thread": same_path_1,
        "same_path_nthreads": same_path_n,
        "mixed_paths_nthreads": mixed,
    }


def run_worker(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    if args.leg == "identity":
        out = worker_identity(repo)
    elif args.leg == "latency":
        out = worker_latency(repo, iters=args.iters, warmup=args.warmup)
    elif args.leg == "concurrency":
        out = worker_concurrency(repo, threads=args.threads, iters=args.iters)
    else:  # pragma: no cover - argparse constrains choices
        raise ValueError(args.leg)
    print(json.dumps(out))


# ===========================================================================
# Orchestrator side — spawns the worker above inside each venv's python and
# renders markdown. Uses only the stdlib; never imports perfdigest itself.
# ===========================================================================

def _venv_python(venv_dir: Path, label: str) -> Path:
    return venv_dir / label / "bin" / "python"


def _check_installed(python_exe: Path) -> tuple[bool, str | None]:
    if not python_exe.exists():
        return False, f"no venv at {python_exe.parent.parent} (env setup step was not run for this version)"
    proc = subprocess.run(
        [str(python_exe), "-c", "import perfdigest.server.app"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        reason = (proc.stderr.strip().splitlines() or ["unknown import error"])[-1]
        return False, reason
    return True, None


def _run_worker(python_exe: Path, script: Path, repo: Path, leg: str, extra: list[str] | None = None) -> dict:
    cmd = [str(python_exe), str(script), "worker", "--leg", leg, "--repo", str(repo)]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _startup_time_s(python_exe: Path, reps: int = 5) -> float | None:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(python_exe), "-c", "import perfdigest.server.app"],
            capture_output=True, text=True, timeout=60,
        )
        samples.append(time.perf_counter() - t0)
        if proc.returncode != 0:
            return None
    return statistics.median(samples)


def render_report(
    venv_dir: Path,
    repo: Path,
    versions: list[str],
    gil_baseline: str,
    freethreaded: str,
    raw_out: Path | None = None,
    legs: tuple[str, ...] = ("identity", "latency", "concurrency"),
    reuse_raw: Path | None = None,
) -> str:
    """Render the three-leg markdown report.

    `legs` restricts which legs are actually MEASURED this call (identity and
    concurrency are load-insensitive and safe to run any time; latency is a
    wall-clock microbenchmark and should only run on a settled host — see
    `os.getloadavg()` guidance in the docstring). A leg not in `legs` is
    instead loaded from `reuse_raw` (previously saved `--raw-out` JSON), so a
    split run (e.g. identity+concurrency now, latency once load settles)
    still produces one complete report.
    """
    script = Path(__file__).resolve()
    lines: list[str] = []

    def _save_raw(leg: str, label: str, payload: dict) -> None:
        if raw_out is None:
            return
        raw_out.mkdir(parents=True, exist_ok=True)
        (raw_out / f"{label}_{leg}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    def _load_raw(leg: str, label: str) -> dict | None:
        if reuse_raw is None:
            return None
        p = reuse_raw / f"{label}_{leg}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # --- host / status probing -------------------------------------------------
    availability: dict[str, tuple[bool, str | None]] = {}
    for label in versions:
        py = _venv_python(venv_dir, label)
        availability[label] = _check_installed(py)

    identity_results: dict[str, dict] = {}
    latency_results: dict[str, dict] = {}
    startup_s: dict[str, float | None] = {}

    # Provenance for the latency leg: a wall-clock microbenchmark is only as
    # honest as the host it ran on. Recorded once, up front, for this pass.
    latency_loadavg: tuple[float, float, float] | None = None
    if "latency" in legs:
        latency_loadavg = os.getloadavg()
        if raw_out is not None:
            raw_out.mkdir(parents=True, exist_ok=True)
            (raw_out / "latency_loadavg.json").write_text(
                json.dumps({"loadavg_1_5_15": list(latency_loadavg)}, indent=1) + "\n"
            )
    elif reuse_raw is not None:
        p = reuse_raw / "latency_loadavg.json"
        if p.exists():
            latency_loadavg = tuple(json.loads(p.read_text())["loadavg_1_5_15"])

    for label in versions:
        ok, reason = availability[label]
        py = _venv_python(venv_dir, label)
        if not ok:
            continue
        if "identity" in legs:
            identity_results[label] = _run_worker(py, script, repo, "identity")
            _save_raw("identity", label, identity_results[label])
        else:
            reused = _load_raw("identity", label)
            if reused is not None:
                identity_results[label] = reused
        if "latency" in legs:
            latency_results[label] = _run_worker(py, script, repo, "latency")
            _save_raw("latency", label, latency_results[label])
            startup_s[label] = _startup_time_s(py)
            _save_raw("startup", label, {"median_startup_s": startup_s[label]})
        else:
            reused = _load_raw("latency", label)
            if reused is not None:
                latency_results[label] = reused
            reused_startup = _load_raw("startup", label)
            if reused_startup is not None:
                startup_s[label] = reused_startup["median_startup_s"]

    concurrency_results: dict[str, dict] = {}
    for label in (gil_baseline, freethreaded):
        ok, _ = availability.get(label, (False, "not in versions list"))
        if not ok:
            continue
        py = _venv_python(venv_dir, label)
        if "concurrency" in legs:
            concurrency_results[label] = _run_worker(py, script, repo, "concurrency")
            _save_raw("concurrency", label, concurrency_results[label])
        else:
            reused = _load_raw("concurrency", label)
            if reused is not None:
                concurrency_results[label] = reused

    # --- (i) identity table ------------------------------------------------
    lines.append("## Leg (i) — digest byte-identity across interpreters\n")
    ok_versions = [v for v in versions if availability[v][0]]
    identity_versions = [v for v in ok_versions if v in identity_results]
    if not identity_versions:
        lines.append("(not measured this pass — rerun with `--legs identity` or supply `--reuse-raw`)\n")
    else:
        header = "| backend:format | fixture | bytes | ~tokens | " + " | ".join(identity_versions) + " | verdict |"
        sep = "|---|---|--:|--:|" + "|".join(["---"] * len(identity_versions)) + "|---|"
        lines.append(header)
        lines.append(sep)
        for entry in FIXTURES:
            key = f"{entry['backend']}:{entry['format']}"
            hashes = {v: identity_results[v]["results"][key]["sha256"] for v in identity_versions}
            first_hash = next(iter(hashes.values()))
            all_same = all(h == first_hash for h in hashes.values())
            nbytes = identity_results[identity_versions[0]]["results"][key]["bytes"]
            cells = " | ".join(hashes[v][:10] for v in identity_versions)
            verdict = "IDENTICAL" if all_same else "**DIFFERS**"
            lines.append(
                f"| `{key}` | `{entry['fixture']}` | {nbytes} | ~{round(nbytes / 4)} | {cells} | {verdict} |"
            )
        lines.append("")
        diffs = []
        for entry in FIXTURES:
            key = f"{entry['backend']}:{entry['format']}"
            hashes = {v: identity_results[v]["results"][key]["sha256"] for v in identity_versions}
            if len(set(hashes.values())) > 1:
                diffs.append((key, hashes))
        if diffs:
            lines.append("**Hash divergence found — investigated below, not papered over:**\n")
            for key, hashes in diffs:
                lines.append(f"- `{key}`: " + ", ".join(f"{v}={h[:10]}" for v, h in hashes.items()))
            lines.append("")
        else:
            lines.append(
                "All backends hash IDENTICAL across every installable interpreter — "
                "the digest contract (and therefore its byte/token cost) is version-invariant.\n"
            )

    # --- (ii) latency table --------------------------------------------------
    lines.append("## Leg (ii) — reader parse/digest latency per version (cache bypassed)\n")
    lines.append(
        "Median [Q1, Q3] microseconds, 50 iterations after 5 warmups, `time.perf_counter()`, "
        "direct `backend.load_units()` calls (not `cached_units`). Microbenchmarks of parse "
        "cost on small real fixtures — a digest-latency budget, not a general interpreter "
        "shootout.\n"
    )
    if latency_loadavg is not None:
        lines.append(
            f"Host load average at measurement time (1/5/15 min): "
            f"{latency_loadavg[0]:.2f}, {latency_loadavg[1]:.2f}, {latency_loadavg[2]:.2f} "
            "(provenance for the wall-clock numbers below — see Limitations if elevated).\n"
        )
    latency_versions = [v for v in ok_versions if v in latency_results]
    if not latency_versions:
        lines.append("(not measured this pass — rerun with `--legs latency` or supply `--reuse-raw`)\n")
    else:
        header = "| backend:format | " + " | ".join(latency_versions) + " |"
        sep = "|---|" + "|".join(["---"] * len(latency_versions)) + "|"
        lines.append(header)
        lines.append(sep)
        for entry in FIXTURES:
            key = f"{entry['backend']}:{entry['format']}"
            cells = []
            for v in latency_versions:
                r = latency_results[v]["results"][key]
                cells.append(f"{r['median_us']:.1f} [{r['q1_us']:.1f}, {r['q3_us']:.1f}]")
            lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Interpreter startup/import time (`import perfdigest.server.app`), median of 5 reps:\n")
    lines.append("| version | median startup (s) |")
    lines.append("|---|--:|")
    for v in ok_versions:
        s = startup_s.get(v)
        lines.append(f"| {v} | {s:.3f} |" if s is not None else f"| {v} | n/a |")
    lines.append("")

    # --- (iii) concurrency table -----------------------------------------
    lines.append("## Leg (iii) — 3.14t free-threaded cache contention\n")
    lines.append(
        f"`report_store.cache.cached_units` under {gil_baseline} (GIL) vs {freethreaded} "
        "(free-threaded). same_path_1thread is the single-threaded reference point; "
        "same_path_nthreads/mixed_paths_nthreads use N=8 threads. `correct` asserts every "
        "thread's digest hash equals the single-threaded reference — the lock's actual job, "
        "not a speed claim. Library-level behavior, not a statement about current MCP server "
        "concurrency (FastMCP runs tools inline today).\n"
    )
    lines.append("| version | GIL | scenario | threads | calls | wall (s) | throughput (calls/s) | correct |")
    lines.append("|---|---|---|--:|--:|--:|--:|---|")
    for v in (gil_baseline, freethreaded):
        r = concurrency_results.get(v)
        if r is None:
            reason = availability.get(v, (False, "not run"))[1]
            lines.append(f"| {v} | - | (could not run) | - | - | - | - | {reason} |")
            continue
        gil = "on" if r["gil_enabled"] else "off"
        for scen_key, scen_label in (
            ("same_path_1thread", "same-path, 1 thread"),
            ("same_path_nthreads", "same-path, N threads"),
            ("mixed_paths_nthreads", "mixed-paths, N threads"),
        ):
            s = r[scen_key]
            correctness = "yes" if s["correct"] else f"NO ({s['total_mismatches']} mismatches)"
            lines.append(
                f"| {v} | {gil} | {scen_label} | {s['n_threads']} | {s['total_calls']} | "
                f"{s['wall_time_s']:.4f} | {s['throughput_calls_per_s']:.0f} | {correctness} |"
            )
    lines.append("")
    if gil_baseline in concurrency_results and freethreaded in concurrency_results:
        base = concurrency_results[gil_baseline]["same_path_nthreads"]["throughput_calls_per_s"]
        free = concurrency_results[freethreaded]["same_path_nthreads"]["throughput_calls_per_s"]
        ratio = free / base if base else float("nan")
        lines.append(
            f"Same-path, 8 threads: {freethreaded} throughput is {ratio:.2f}x the "
            f"{gil_baseline} (GIL) build's.\n"
        )

    # --- limitations / could-not-run ---------------------------------------
    lines.append("## Could-not-run\n")
    not_run = [(v, availability[v][1]) for v in versions if not availability[v][0]]
    if not_run:
        for v, reason in not_run:
            lines.append(f"- **{v}**: could not run — {reason}")
    else:
        lines.append("- None — every requested interpreter installed and ran all applicable legs.")
    lines.append("")

    return "\n".join(lines)


def run_report(args: argparse.Namespace) -> None:
    venv_dir = Path(args.venv_dir).resolve()
    repo = Path(args.repo).resolve()
    versions = args.versions.split(",") if args.versions else ALL_VERSIONS
    raw_out = Path(args.raw_out).resolve() if args.raw_out else None
    reuse_raw = Path(args.reuse_raw).resolve() if args.reuse_raw else None
    legs = tuple(args.legs.split(",")) if args.legs else ("identity", "latency", "concurrency")
    md = render_report(
        venv_dir, repo, versions, args.gil_baseline, args.freethreaded,
        raw_out=raw_out, legs=legs, reuse_raw=reuse_raw,
    )
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_worker = sub.add_parser("worker", help="run one leg inside a single venv, emit JSON to stdout")
    p_worker.add_argument("--leg", choices=["identity", "latency", "concurrency"], required=True)
    p_worker.add_argument("--repo", required=True, help="path to the MCP_Developing_REPO checkout")
    p_worker.add_argument("--iters", type=int, default=50)
    p_worker.add_argument("--warmup", type=int, default=5)
    p_worker.add_argument("--threads", type=int, default=8)
    p_worker.set_defaults(func=run_worker)

    p_report = sub.add_parser("report", help="fan out to every venv and render markdown to stdout")
    p_report.add_argument("--venv-dir", required=True, help="dir containing one venv per version label")
    p_report.add_argument("--repo", required=True, help="path to the MCP_Developing_REPO checkout")
    p_report.add_argument("--versions", default=",".join(ALL_VERSIONS))
    p_report.add_argument("--gil-baseline", default="3.14")
    p_report.add_argument("--freethreaded", default="3.14t")
    p_report.add_argument("--raw-out", default=None, help="dir to dump each worker's raw JSON (the captured artifacts)")
    p_report.add_argument(
        "--legs", default=None,
        help="comma list restricting which legs are MEASURED this run (default: all three); "
             "legs left out are pulled from --reuse-raw instead of re-measured — use this to "
             "split latency (host-load-sensitive) from identity/concurrency (load-insensitive)",
    )
    p_report.add_argument(
        "--reuse-raw", default=None,
        help="dir of previously saved --raw-out JSON to reuse for legs not in --legs",
    )
    p_report.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
