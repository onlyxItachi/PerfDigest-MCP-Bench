#!/usr/bin/env python3
"""Deterministic exporter for the cross-repo real-workload regression corpus
(``perfdigest.regression-corpus.v1``).

Catalogs the committed raw profiler reports in ``results/cpp_build_loop_raw/``
and ``results/cuda_opt_bench_raw/`` along with their SHA-256 digests, byte
sizes, formats, and expected ``PerfDigest-MCP`` digest invariants so both
``PerfDigest-MCP-Bench`` CI and downstream ``perfdigest-plugin`` regression
suites can verify end-to-end provenance and digest honesty without modifying
historical raw artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    REPO_ROOT / "schemas" / "examples" / "real-workloads-corpus.v1.json"
)
CORPUS_CREATED_AT = "2026-07-19T00:00:00Z"

STUDY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "cpp_build_loop",
        "title": "Demanding Workloads Pt. 3: C++ Build Loop (CMake + Ninja + Clang -ftime-trace)",
        "studyDoc": "results/DEMANDING_WORKLOADS_3_CPP_BUILD_LOOP_2026-07-19.md",
        "rawDir": "results/cpp_build_loop_raw",
        "question": (
            "Which translation unit and compiler frontend phase dominate the C++ build "
            "loop, and what causes the template instantiation bottleneck?"
        ),
        "expectedSelectedReportId": "template-heavy",
        "reports": (
            {
                "id": "cold-ninja",
                "path": "results/cpp_build_loop_raw/cold.ninja_log",
                "format": "ninja-log",
                "expectedDigest": {
                    "totalUnits": 7,
                    "topUnitName": "CMakeFiles/bl_core.dir/src/lib_template_heavy.cpp.o",
                    "topUnitIndex": 4,
                    "topUnitDomain": "build_step",
                    "topUnitMetrics": {
                        "duration_ms": 2053.0,
                        "start_ms": 1.0,
                        "end_ms": 2054.0,
                    },
                },
            },
            {
                "id": "configure-cold",
                "path": "results/cpp_build_loop_raw/configure_cold.cmake-profile.json",
                "format": "cmake-profile",
                "expectedDigest": {
                    "totalUnits": 90,
                    "topUnitName": "configure",
                    "topUnitIndex": 0,
                    "topUnitDomain": "build_phase",
                    "topUnitMetrics": {
                        "calls": 1.0,
                        "total_time_us": 394547.0,
                        "avg_time_us": 394547.0,
                        "max_time_us": 394547.0,
                    },
                },
            },
            {
                "id": "configure-warm",
                "path": "results/cpp_build_loop_raw/configure_warm.cmake-profile.json",
                "format": "cmake-profile",
                "expectedDigest": {
                    "totalUnits": 55,
                    "topUnitName": "configure",
                    "topUnitIndex": 0,
                    "topUnitDomain": "build_phase",
                    "topUnitMetrics": {
                        "calls": 1.0,
                        "total_time_us": 14060.0,
                        "avg_time_us": 14060.0,
                        "max_time_us": 14060.0,
                    },
                },
            },
            {
                "id": "incremental-ninja",
                "path": "results/cpp_build_loop_raw/incremental.ninja_log",
                "format": "ninja-log",
                "expectedDigest": {
                    "totalUnits": 7,
                    "topUnitName": "CMakeFiles/bl_core.dir/src/lib_template_heavy.cpp.o",
                    "topUnitIndex": 4,
                    "topUnitDomain": "build_step",
                    "topUnitMetrics": {
                        "duration_ms": 2072.0,
                        "start_ms": 1.0,
                        "end_ms": 2073.0,
                    },
                },
            },
            {
                "id": "template-heavy",
                "path": "results/cpp_build_loop_raw/template_heavy.ftime-trace.json",
                "format": "clang-time-trace",
                "expectedDigest": {
                    "totalUnits": 189,
                    "topUnitName": "InstantiateClass",
                    "topUnitIndex": 159,
                    "topUnitDomain": "build_phase",
                    "topUnitMetrics": {
                        "calls": 412.0,
                        "total_time_us": 3496667.0,
                        "avg_time_us": 8487.055825242718,
                        "max_time_us": 149155.0,
                    },
                },
            },
        ),
    },
    {
        "id": "cuda_opt_bench",
        "title": "Demanding Workloads Pt. 4: CUDA Optimization Loop (ptxas static codegen)",
        "studyDoc": "results/DEMANDING_WORKLOADS_4_CUDA_OPT_LOOP_2026-07-19.md",
        "rawDir": "results/cuda_opt_bench_raw",
        "question": (
            "What do the ptxas kernel codegen metrics (register pressure, spill stores/loads, "
            "and constant memory) show across the compiled CUDA kernels?"
        ),
        "expectedSelectedReportId": "ptxas-verbose",
        "reports": (
            {
                "id": "ptxas-verbose",
                "path": "results/cuda_opt_bench_raw/ptxas_verbose.txt",
                "format": "ptxas-verbose",
                "expectedDigest": {
                    "totalUnits": 7,
                    "topUnitName": "_Z15checksum_kernelPKflPd [sm_89]",
                    "topUnitIndex": 0,
                    "topUnitDomain": "kernel_codegen",
                    "topUnitMetrics": {
                        "registers_per_thread": 14.0,
                        "spill_stores_bytes": 0.0,
                        "spill_loads_bytes": 0.0,
                        "stack_frame_bytes": 0.0,
                        "static_smem_bytes": 0.0,
                        "const_mem_bytes": 376.0,
                        "barriers": 0.0,
                        "compile_time_ms": 3.685,
                    },
                },
            },
        ),
    },
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_regression_corpus_v1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    studies: list[dict[str, Any]] = []
    for spec in STUDY_SPECS:
        reports: list[dict[str, Any]] = []
        for r in spec["reports"]:
            abs_path = repo_root / r["path"]
            if not abs_path.exists():
                raise FileNotFoundError(f"Missing committed raw report: {abs_path}")
            reports.append(
                {
                    "id": r["id"],
                    "path": r["path"],
                    "format": r["format"],
                    "bytes": abs_path.stat().st_size,
                    "sha256": _sha256_file(abs_path),
                    "expectedDigest": r["expectedDigest"],
                }
            )
        studies.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "studyDoc": spec["studyDoc"],
                "rawDir": spec["rawDir"],
                "question": spec["question"],
                "expectedSelectedReportId": spec["expectedSelectedReportId"],
                "reports": reports,
            }
        )
    return {
        "schema": "perfdigest.regression-corpus.v1",
        "createdAt": CORPUS_CREATED_AT,
        "upstreamRepo": "onlyxItachi/PerfDigest-MCP-Bench",
        "studies": studies,
    }


def serialize_regression_corpus_v1(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2) + "\n"


def validate_invariants(
    doc: dict[str, Any],
    repo_root: Path = REPO_ROOT,
    expected_path: Path = DEFAULT_OUTPUT,
) -> list[str]:
    errors: list[str] = []
    if doc.get("schema") != "perfdigest.regression-corpus.v1":
        errors.append(
            f"schema must be 'perfdigest.regression-corpus.v1', got {doc.get('schema')!r}"
        )
    if doc.get("upstreamRepo") != "onlyxItachi/PerfDigest-MCP-Bench":
        errors.append("upstreamRepo must be 'onlyxItachi/PerfDigest-MCP-Bench'")

    studies = doc.get("studies", [])
    if not isinstance(studies, list) or len(studies) < 2:
        errors.append("studies must contain at least cpp_build_loop and cuda_opt_bench")
        return errors

    for study in studies:
        sid = study.get("id", "<unknown>")
        study_doc = repo_root / str(study.get("studyDoc", ""))
        if not study_doc.exists():
            errors.append(f"{sid}: studyDoc {study.get('studyDoc')!r} does not exist")
        reports = study.get("reports", [])
        if not reports:
            errors.append(f"{sid}: reports must be non-empty")
            continue
        report_ids = {r.get("id") for r in reports}
        if study.get("expectedSelectedReportId") not in report_ids:
            errors.append(
                f"{sid}: expectedSelectedReportId {study.get('expectedSelectedReportId')!r} "
                f"not in report ids {sorted(report_ids)}"
            )
        for r in reports:
            rid = f"{sid}/{r.get('id', '<unknown>')}"
            rpath = repo_root / str(r.get("path", ""))
            if not rpath.exists():
                errors.append(f"{rid}: path {r.get('path')!r} does not exist")
                continue
            actual_bytes = rpath.stat().st_size
            actual_sha = _sha256_file(rpath)
            if r.get("bytes") != actual_bytes:
                errors.append(f"{rid}: bytes mismatch ({r.get('bytes')} != {actual_bytes})")
            if r.get("sha256") != actual_sha:
                errors.append(f"{rid}: sha256 mismatch ({r.get('sha256')} != {actual_sha})")

    if not expected_path.exists():
        errors.append(f"expected output file {expected_path} does not exist")
    else:
        rendered = serialize_regression_corpus_v1(doc).encode("utf-8")
        on_disk = expected_path.read_bytes()
        if rendered != on_disk:
            errors.append(
                f"{expected_path} is not byte-identical to freshly exported corpus "
                f"(expected sha256={hashlib.sha256(rendered).hexdigest()}, "
                f"got sha256={hashlib.sha256(on_disk).hexdigest()})"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the regression-corpus.v1 JSON document",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify structural invariants and byte-identical regeneration without writing",
    )
    args = parser.parse_args(argv)

    doc = build_regression_corpus_v1(repo_root=REPO_ROOT)
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialize_regression_corpus_v1(doc), encoding="utf-8")

    errors = validate_invariants(doc, repo_root=REPO_ROOT, expected_path=args.output)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    total_reports = sum(len(s["reports"]) for s in doc["studies"])
    mode_label = "verified (--check)" if args.check else "exported"
    print(
        f"OK: {mode_label} {args.output.relative_to(REPO_ROOT)} "
        f"({len(doc['studies'])} studies, {total_reports} raw reports, "
        f"sha256={_sha256_file(args.output)[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
