"""Validate exact staging and distinct-route terminal admission with CPU fixtures."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


class Check(unittest.TestCase):
    def test_stage_and_producer_binding(self):
        label = prepare.HERE / "CLAIM.json"
        claim = json.loads(label.read_bytes())
        self.assertEqual([claim["cpu_request"], claim["gpu_request"]], [prepare.CPU_REQUEST, prepare.GPU_REQUEST])
        archive = prepare.RESULT.with_name("O01-zimage-transposed-lt-native-original1882-terminal.json")
        raw = prepare.pinned(archive, claim["source_evidence"]["original_compile_SHA256"])
        with self.assertRaises(ValueError):
            prepare.build_binding(archive, prepare.digest(raw))
        fake = json.loads(raw)
        cpu = prepare.cpu_inputs()
        fake.update(request=prepare.CPU_REQUEST)
        record = fake["files"]["build-result.json"]["record"]
        record.update(request=prepare.CPU_REQUEST, step="1103606.999", cpp_SHA256=prepare.digest(cpu["probe.cpp"]))
        record["artifact"]["exports"] = sorted(prepare.EXPORTS)
        record["artifact"]["SHA256"] = "d" * 64
        fake["accounting"][0][0] = record["step"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "fixture-build.json"
            done.write_text(json.dumps(fake), encoding="utf8")
            sha = prepare.digest(done.read_bytes())
            adopted = prepare.build_binding(done, sha)
            source = prepare.stage(root / "cpu", cpu, prepare.CPU_REQUEST)
            self.assertEqual(source["stage_SHA256"], claim["cpu_stage"]["manifest_SHA256"])
            result = prepare.stage(root / "gpu", prepare.gpu_inputs(adopted), prepare.GPU_REQUEST)
            self.assertEqual(result["request"], prepare.GPU_REQUEST)
            self.assertIn(b"d" * 64, (root / "gpu" / "worker.py").read_bytes())
            with self.assertRaises(FileExistsError):
                prepare.stage(root / "gpu", prepare.gpu_inputs(adopted), prepare.GPU_REQUEST)
            with self.assertRaises(ValueError):
                prepare.build_binding(done, "e" * 64)

    def test_actual_rendered_terminal_marks_alias_and_absence(self):
        fake_binding = {"step": "1103606.999", "SHA256": "d" * 64, "bytes": 10011, "archive_SHA256": "e" * 64}
        raw = prepare.gpu_inputs(fake_binding)["worker.py"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "contract.py").write_bytes(prepare.gpu_inputs(fake_binding)["contract.py"])
            source = root / "worker.py"
            scope = dict(node="f12r2n11", PCI="0000:09:00.0", deadline_epoch=400)
            step = "1103606.998"
            worker = {"__name__": "fixture", "__file__": str(source)}
            initial_import_path = sys.path[:]
            exec(compile(raw, str(source), "exec"), worker)
            worker["admission"] = lambda *_: (scope, step)

            def run(directory, attempt, symbol):
                def in_trace(_command, out, _deadline):
                    witness = dict(step=step, physical_PCI=scope["PCI"], attempted_calls=attempt,
                                   solutions=[{"solution": 99}] if attempt else [], close_rc=0,
                                   create_rc=0, query_rc=0, workspace_cap_bytes=32 << 20)
                    (out / "candidate-witness.json").write_text(json.dumps(witness), encoding="utf8")
                    return 0

                worker["in_trace"] = in_trace
                worker["audit"] = lambda *_: ([{"symbol": symbol}] if symbol else [], None)
                args = [str(source), "--scope", str(root / "scope.json"), "--scope-sha256", "a" * 64,
                        "--out", str(directory)]
                with patch.object(sys, "argv", args):
                    code = worker["main"]()
                return code, json.loads((directory / "terminal.json").read_bytes())["decision"]

            try:
                self.assertEqual(run(root / "none", 0, None), (1, "NO_SUPPORTED_NEW_IN_PREFIX"))
                self.assertEqual(run(root / "alias", 1, prepare.original_routes()[4021]),
                                 (1, "NEW_ID_REUSES_PREVIOUS_KERNEL"))
                self.assertEqual(run(root / "missing", 1, None), (1, "NEW_SOLUTION_NUMERIC_OR_ROUTE_UNQUALIFIED"))
                self.assertEqual(run(root / "new", 1, "Cijk_Ailk_Bljk_fixture"), (0, "NEW_BF16_KERNEL_QUALIFIED"))
            finally:
                sys.path[:] = initial_import_path


if __name__ == "__main__":
    unittest.main()
