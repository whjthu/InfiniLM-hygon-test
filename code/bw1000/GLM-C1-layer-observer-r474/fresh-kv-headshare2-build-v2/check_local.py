"""Exercise the generated admission gate and the original pilot renderer offline."""

import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare
import prepare_gpu


class Revision(unittest.TestCase):
    def test_generated_cpu_gate(self):
        stage = prepare.build_inputs()
        previous = prepare.frozen_stage()
        self.assertEqual(stage["fresh_kv_headshare.cu"], previous["fresh_kv_headshare.cu"])
        plan = json.loads(stage["PLAN.json"])
        old_plan = json.loads(previous["PLAN.json"])
        changes = {key for key, value in plan.items() if old_plan.get(key) != value}
        self.assertEqual(changes, {"label", "compile_entry_SHA256", "optional_exact_hip_version_header"})
        worker = {"__name__": "worker", "__file__": __file__}
        exec(compile(stage["compile_once.py"], "compile_once.py", "exec"), worker)
        gate = worker["admit_hip_version"]
        self.assertEqual(worker["HIP_VERSION_SHA"], prepare.HEADER_SHA)
        with TemporaryDirectory() as tmp:
            header = Path(tmp).resolve() / "hip_version.h"
            raw = b"a" * prepare.HEADER_BYTES
            other = "/different/unmapped.h"
            worker.update(HIP_VERSION=str(header), HIP_VERSION_SHA=prepare.digest(raw))
            original = {"/original": "old"}
            gate({"/original", other}, original)
            self.assertEqual(original, {"/original": "old"})
            with self.assertRaises(FileNotFoundError):
                gate({str(header)}, original)
            header.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, "DTK_HIP_VERSION_HEADER_IDENTITY"):
                gate({str(header)}, original)
            header.write_bytes(b"b" + raw[1:])
            with self.assertRaisesRegex(ValueError, "DTK_HIP_VERSION_HEADER_IDENTITY"):
                gate({str(header)}, original)
            header.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "DTK_HIP_VERSION_HEADER_CONFLICT"):
                gate({str(header)}, {str(header): "historically conflicting"})
            gate({str(header)}, original)
            self.assertEqual(original, {"/original": "old", str(header): prepare.digest(raw)})

    def test_original_gpu_pilot_adopts_v2_only(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cpu = prepare.stage(folder / "cpu")
            gpu = prepare_gpu.stage(folder / "gpu")
            self.assertEqual(cpu["compile_manifest_SHA256"], gpu["build_manifest_SHA256"])
            self.assertEqual(gpu["pilot_request"], prepare_gpu.PILOT_REQUEST)
            with self.assertRaises(FileExistsError):
                prepare.stage(folder / "cpu")
            with self.assertRaises(FileExistsError):
                prepare_gpu.stage(folder / "gpu")
            launch = runpy.run_path(str(folder / "gpu" / "launch_plan.py"))
            # Load the actual standalone sibling imports before restoring the fixture environment.
            original_path = sys.path[:]
            sys.modules.pop("prepare", None)
            try:
                device = runpy.run_path(str(folder / "gpu" / "device.py"))
            finally:
                sys.modules["prepare"] = prepare
                sys.path[:] = original_path
            self.assertEqual(device["DSO"].name, "libGLM_fresh_kv_headshare.so")
            self.assertEqual(device["DSO"].parent.name, prepare.LABEL)
            self.assertEqual(device["BUILD_MANIFEST"], cpu["compile_manifest_SHA256"])
            self.assertEqual(launch["BUILD_REQUESTS"], (prepare_gpu.V1_REQUEST, prepare.REQUEST))
            root = str(launch["ROOT"])

            def specification(kind, request):
                admitted = dict(node="f12r2n13", parent="1103606", request=request, pre_step="1103606.99",
                                root_admitted=True, epoch=100, deadline_epoch=250, scope_SHA256="b" * 64)
                return dict(kind=kind, request=request, package=root + "/incoming/source",
                            campaign=root + "/results/test-" + request, project_owned=True,
                            fresh_protected_excluded=True, epoch=100, deadline_epoch=250,
                            nodes={"f12r2n13": admitted}, build_receipt=root + "/candidates/" + prepare.LABEL + "/CPU-result.json",
                            build_receipt_SHA256="a" * 64)

            original = launch["derive"](specification("pilot", prepare_gpu.PILOT_REQUEST), now=102)
            self.assertEqual(original["planned_cards"], 8)
            self.assertEqual(original["submissions_performed"], 0)
            for request in (*launch["BUILD_REQUESTS"], prepare_gpu.PILOT_REQUEST):
                with self.assertRaisesRegex(ValueError, "new sharded scope differs"):
                    launch["derive"](specification("followup", request), now=102)


if __name__ == "__main__":
    unittest.main()
