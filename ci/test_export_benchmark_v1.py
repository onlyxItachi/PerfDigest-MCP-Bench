"""Unit tests for ``ci/export_benchmark_v1.py`` and the coordinated v1 schemas."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ci import export_benchmark_v1 as exporter
from ci import export_regression_corpus as corpus_exporter

REPO_ROOT = Path(__file__).resolve().parent.parent


class ExportBenchmarkV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = exporter.build_benchmark_result_v1()

    def test_schema_files_present_and_define_three_layer_contract(self) -> None:
        result_schema_path = REPO_ROOT / "schemas" / "benchmark-result.v1.schema.json"
        manifest_schema_path = REPO_ROOT / "schemas" / "benchmark-manifest.v1.schema.json"
        self.assertTrue(result_schema_path.exists())
        self.assertTrue(manifest_schema_path.exists())

        result_schema = json.loads(result_schema_path.read_text(encoding="utf-8"))
        manifest_schema = json.loads(manifest_schema_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result_schema["properties"]["schema"]["const"],
            "perfdigest.benchmark-result.v1",
        )
        self.assertEqual(
            manifest_schema["properties"]["schema"]["const"],
            "perfdigest.benchmark-manifest.v1",
        )

        cell_props = result_schema["properties"]["cells"]["items"]["properties"]
        self.assertEqual(
            cell_props["layer"]["enum"],
            ["engine", "orchestration", "agent_e2e"],
        )
        self.assertEqual(
            cell_props["memoryCoverage"]["enum"],
            [
                "cgroup_v2_tree_peak",
                "max_single_process_only_incomplete_tree",
                "none",
            ],
        )
        self.assertIn("cgroupTreePeakKb", cell_props)
        self.assertIn("layerSummary", result_schema["properties"])
        self.assertIn("benchCompatibility", result_schema["properties"])

    def test_export_invariants_and_layer_summary(self) -> None:
        self.assertEqual(self.doc["schema"], "perfdigest.benchmark-result.v1")
        cells = self.doc["cells"]
        self.assertEqual(len(cells), 72)
        self.assertEqual(self.doc["layerSummary"]["engine"]["total"], len(cells))
        self.assertEqual(self.doc["layerSummary"]["engine"]["measured"], len(cells))
        self.assertEqual(self.doc["layerSummary"]["engine"]["parityPass"], len(cells))
        self.assertEqual(self.doc["layerSummary"]["orchestration"]["total"], 0)
        self.assertEqual(self.doc["layerSummary"]["agent_e2e"]["total"], 0)

        for cell in cells:
            self.assertEqual(cell["layer"], "engine")
            self.assertEqual(cell["group"], "runtime")
            self.assertEqual(cell["memoryCoverage"], "none")
            self.assertEqual(cell["status"], "measured")
            self.assertEqual(cell["parity"]["status"], "pass")

    def test_wall_parity_startup_concurrency_and_loadavg_mappings(self) -> None:
        raw_dir = REPO_ROOT / "results" / "interpreter_matrix_raw"
        loadavg = json.loads((raw_dir / "latency_loadavg.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            self.doc["host"]["loadavg1AtStart"],
            loadavg["loadavg_1_5_15"][0],
        )

        by_id = {c["id"]: c for c in self.doc["cells"]}

        # Check a parse cell against raw 3.10_latency.json + 3.10_identity.json
        raw_310_lat = json.loads((raw_dir / "3.10_latency.json").read_text(encoding="utf-8"))
        raw_310_id = json.loads((raw_dir / "3.10_identity.json").read_text(encoding="utf-8"))
        cell_310 = by_id["engine.cpython-3.10.cargo_diag.cargo-diag"]
        self.assertEqual(cell_310["wall"]["unit"], "us")
        self.assertEqual(cell_310["wall"]["n"], 50)
        self.assertAlmostEqual(
            cell_310["wall"]["median"],
            raw_310_lat["results"]["cargo_diag:cargo-diag"]["median_us"],
        )
        self.assertEqual(
            cell_310["extra"]["identity"]["sha256"],
            raw_310_id["results"]["cargo_diag:cargo-diag"]["sha256"],
        )
        self.assertTrue(cell_310["extra"]["identity"]["matchesGilBaseline"])

        # Check the documented 3.10/3.11 vs 3.12+ Neumaier sum() divergence on torch-trace
        torch_310 = by_id["engine.cpython-3.10.chrome_trace.torch-trace"]
        torch_314 = by_id["engine.cpython-3.14.chrome_trace.torch-trace"]
        self.assertFalse(torch_310["extra"]["identity"]["matchesGilBaseline"])
        self.assertTrue(torch_314["extra"]["identity"]["matchesGilBaseline"])

        # Check startup cell + concurrency metadata for 3.14t
        raw_314t_startup = json.loads((raw_dir / "3.14t_startup.json").read_text(encoding="utf-8"))
        startup_314t = by_id["engine.cpython-3.14t.interpreter-startup"]
        self.assertEqual(startup_314t["wall"]["unit"], "s")
        self.assertAlmostEqual(
            startup_314t["wall"]["median"],
            raw_314t_startup["median_startup_s"],
        )
        conc = startup_314t["extra"]["concurrency"]
        self.assertFalse(conc["gil_enabled"])
        self.assertTrue(conc["same_path_nthreads"]["correct"])
        self.assertTrue(conc["mixed_paths_nthreads"]["correct"])

    def test_check_mode_and_byte_identical_detection(self) -> None:
        self.assertEqual(exporter.main(["--check"]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "drifted.json"
            bad_path.write_text("{}", encoding="utf-8")
            errors = exporter.validate_invariants(self.doc, expected_path=bad_path)
            self.assertTrue(any("not byte-identical" in e for e in errors))

    def test_existing_raw_and_study_files_untouched(self) -> None:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "results/", "workloads/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "")

    def test_regression_corpus_schema_and_export_invariants(self) -> None:
        corpus_schema_path = REPO_ROOT / "schemas" / "regression-corpus.v1.schema.json"
        self.assertTrue(corpus_schema_path.exists())
        corpus_schema = json.loads(corpus_schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            corpus_schema["properties"]["schema"]["const"],
            "perfdigest.regression-corpus.v1",
        )

        corpus_doc = corpus_exporter.build_regression_corpus_v1()
        self.assertEqual(corpus_doc["schema"], "perfdigest.regression-corpus.v1")
        self.assertEqual(corpus_doc["upstreamRepo"], "onlyxItachi/PerfDigest-MCP-Bench")
        self.assertEqual(len(corpus_doc["studies"]), 2)
        self.assertEqual(
            [s["id"] for s in corpus_doc["studies"]],
            ["cpp_build_loop", "cuda_opt_bench"],
        )
        self.assertEqual(corpus_exporter.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
