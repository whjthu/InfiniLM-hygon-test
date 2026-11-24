"""Exercise the new deployment gates and reader without Slurm or torch."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import precheck
import read_result as reader


def render_runtime(path):
    source = HERE.with_name("zimage-square-rhs-pack16-v1") / "prepare.py"
    spec = importlib.util.spec_from_file_location("pack16_original_renderer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files, manifest = module.build()
    path.mkdir()
    for name, data in files.items():
        (path / name).write_bytes(data)
    (path / "SCOPED_FILES.json").write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def write(path, value):
    data = precheck.encode(value)
    path.write_bytes(data)
    return sha256(data).hexdigest()


class Precheck(unittest.TestCase):
    def test_exact_coexisting_source_argv(self):
        self.assertRegex(precheck.REQUEST, "[0-9a-f]{32}")
        argv = precheck.srun_argv(precheck.STAGE / "precheck.py", "b" * 64)
        self.assertTrue({"--gres=none", "--exact", "--overlap", "--cpus-per-task=1", "--mem=256M",
                         "--time=00:01:00", "--nodelist=f12r2n10"}.issubset(argv))
        self.assertFalse(any("\\" in arg for arg in argv))
        self.assertNotIn("--gres=dcu", " ".join(argv))
        self.assertIn("--rocm", precheck.container_argv(precheck.STAGE / "precheck.py"))

    def test_actual_step_matching_refuses_duplicate_lost_or_nonzero_gres(self):
        member = {"StepId": precheck.STANDBY_STEP, "NodeList": precheck.NODE, "State": "RUNNING",
                  "Name": precheck.STANDBY_LABEL, "TRES": "cpu=8,node=1,gres/dcu=8,mem=16G"}
        step = "1103606.3800"
        observer = {"StepId": step, "NodeList": precheck.NODE, "State": "RUNNING",
                    "Name": precheck.LABEL, "TRES": "cpu=1,mem=256M,node=1"}
        with patch.object(precheck, "rows", return_value=[member, observer]):
            precheck.live_pair(step)
        for snapshot in ([member, observer, observer], [observer],
                         [member, observer | {"TRES": "cpu=1,gres/dcu=1,mem=256M,node=1"}],
                         [member | {"State": "COMPLETED"}, observer]):
            with patch.object(precheck, "rows", return_value=snapshot):
                with self.assertRaises(ValueError):
                    precheck.live_pair(step)

    def test_torch_metadata_reads_literal_without_importing(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "torch"
            package.mkdir()
            init = package / "__init__.py"
            init.write_text("raise RuntimeError('module must not be imported')")
            (package / "version.py").write_text("__version__: str = '2.10.0'\nhip = '6.3.26113'\n")
            (package / "_C.cpython-310-x86_64-linux-gnu.so").touch()
            spec = SimpleNamespace(origin=str(init), submodule_search_locations=[str(package)])
            with patch.object(precheck.PathFinder, "find_spec", return_value=spec):
                result = precheck.torch_metadata()
                self.assertFalse(result["package_imported"])
                self.assertEqual(result["declared_hip"], "6.3.26113")
                (package / "version.py").write_text("__version__='2.10.1'\nhip='6.3.26113'\n")
                with self.assertRaisesRegex(ValueError, "PRECHECK_TORCH_VERSION"):
                    precheck.torch_metadata()

    def test_matching_stage_is_rehashed_and_wrong_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "runtime"
            render_runtime(stage)
            self.assertEqual(precheck.runtime(stage)["members"], 5)
            (stage / "candidate.py").write_bytes(b"x=1\n")
            with self.assertRaisesRegex(ValueError, "PRECHECK_SOURCE_PIN"):
                precheck.runtime(stage)


class Terminal(unittest.TestCase):
    def terminal(self, root, status):
        scope = {"request": precheck.GPU_REQUEST, "label": reader.LABEL, "node": precheck.NODE,
                 "parent": precheck.PARENT, "source_publication": precheck.GPU_PUBLICATION,
                 "runtime_SHA256": precheck.RUNTIME_SHA, "root_admitted": True, "PCI": "0000:01:00.0",
                 "epoch": 300, "deadline_epoch": 540}
        step = "1103606.3910"
        scope_file = root / "scope.json"
        scope_sha = write(scope_file, scope)
        rows = [[step, reader.LABEL, status, "0:0", "8", precheck.NODE, "cpu=4,gres/dcu=1,mem=8G,node=1"]]
        accounting = root / "accounting.json"
        accounting_sha = write(accounting, {"accounting_columns": reader.COLUMNS, "accounting": rows})
        runtime, work = root / "runtime", root / "worker-output"
        manifest = render_runtime(runtime)
        work.mkdir()
        report = {"request": precheck.GPU_REQUEST, "step": step, "node": precheck.NODE, "PCI": scope["PCI"],
                  "scope_SHA256": scope_sha, "source_training_step": "1103606.1787", "training_performance": False,
                  "training_RHS_reuse_identity_verified": False, "same_rhs_reuse_per_burst": 16,
                  "packed_RHS_payload_bytes": 29491200, "planned_extra_pack_guard_bytes": 1024,
                  "complete": True, "passed": True, "routes": {}, "pairs": []}
        for mode, symbol in manifest["stock_kernel_by_arm"].items():
            cell = work / mode
            cell.mkdir()
            witness = {"layout": mode, "arm": "cublas", "physical_PCI": scope["PCI"], "trace_device_index": 0,
                       "complete": True, "profiled_mm_calls": 1, "start_rc": 0, "stop_rc": 0,
                       "bitwise_CPU_reference": True, "requested_preference_verified": True, "pid": 56,
                       "realtime_ns": [1000, 1050]}
            pin = write(cell / "witness.json", witness)
            raw_trace = {"traceEvents": [{"ph": "X", "cat": "HIPOPS",
                                          "args": {"pid": 56, "devId": 0, "name": symbol, "BeginNs": 1010, "EndNs": 1029}}]}
            trace_sha = write(cell / "trace.json", raw_trace)
            report["routes"][mode] = {"state": "QUALIFIED", "unique_contraction": symbol, "tool_exit": 0,
                                     "witness": witness, "witness_SHA256": pin, "raw_trace_SHA256": trace_sha,
                                     "counts": {"complete_HIPOPS": 1, "other_owner": 0, "wrong_device": 0,
                                                "outside_window": 0, "attached": 1, "recognized_contractions": 1}}
        for i, order in enumerate(reader.ORDER, 1):
            arms = []
            for mode in order:
                packed = mode == "packed"
                amount = 9 if packed else 10
                burst = {"wall_seconds": amount, "total_event_ms": amount, "GEMM_event_ms": amount - int(packed),
                         "first_interval_ms": int(packed), "copies": int(packed)}
                arms.append({"layout": mode, "verified_measured_calls": 64, "verified_warm_calls": 32,
                             "peak_allocated_bytes": 120000000, "peak_reserved_bytes": 125000000,
                             "bursts": [burst.copy() for _ in range(4)]})
            report["pairs"].append({"pair": i, "order": list(order), "status": "COMPLETE", "arms": arms})
        write(work / "terminal.json", report)
        return (work, runtime, scope_file, scope_sha, accounting, accounting_sha, step)

    def test_actual_gate_prior_to_terminal_and_preserves_failed_accounting(self):
        for status in ("RUNNING", "FAILED"):
            with tempfile.TemporaryDirectory() as tmp:
                data = self.terminal(Path(tmp), status)
                row = reader.read_result(*data)
                self.assertIs(row["real_GPU_terminal"], status == "FAILED")
                self.assertNotEqual(row["actual_accounting"][2], "COMPLETED")
                self.assertFalse(row["worker_result_passed"])
                self.assertFalse(row["promotion_allowed"])

    def test_actual_trace_and_three_pairs_recomputed_before_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.terminal(Path(tmp), "COMPLETED")
            row = reader.read_result(*data)
            self.assertTrue(row["worker_result_passed"])
            self.assertTrue(row["component_timing_candidate"])
            self.assertEqual(row["pairs"]["verified_mm_calls"], 384)
            self.assertEqual(row["pairs"]["measured_copy_receipts"], 12)
            self.assertEqual(len(row["pairs"]["arm_allocator_peaks"]), 6)
            self.assertEqual(row["independent_post"], "UNVERIFIED")
            self.assertFalse(row["promotion_allowed"])
            (data[0] / "packed/trace.json").write_text('{"traceEvents":[]}')
            bad = reader.read_result(*data)
            self.assertFalse(bad["worker_result_passed"])
            self.assertEqual(bad["reader_failure"]["gate"], "ACTUAL_ROUTE_OR_TRACE")


if __name__ == "__main__":
    unittest.main()
