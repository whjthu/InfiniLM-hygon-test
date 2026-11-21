"""New reader boundary tests using synthetic ELF and recorded passive command output."""

from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import read_glm474_fresh_kv_explicit_ar_native_terminal as reader
import test_glm474_fresh_kv_explicit_ar_native as build_fixture


def serialize(value):
    return json.dumps(value, sort_keys=True).encode()


def local_elf(kind):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", header, 16, kind, 62, 1)
    return bytes(header)


class ExplicitTerminalReaderTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.out = self.directory / "out"
        self.out.mkdir()
        self.intent = self.directory / "intent.json"
        self.kernel = self.directory / "kernel-reconciliation.json"
        baseline = build_fixture.ExplicitNativeTest()
        baseline.setUp()
        self.backend = dict(baseline.build)
        self.origin = baseline.origin
        self.plan_path = reader.explicit.CALLER.parent / "depth-stratified-observer-CPU-v1/PLAN.json"
        plan_raw = self.plan_path.read_bytes()
        self.plan = json.loads(plan_raw)
        self.epoch = 1795300000.5
        self.row = ["1103606.791", reader.explicit.LABEL, "COMPLETED", "0:0", "261", reader.NODE,
                    "cpu=4,mem=16G,node=1"]
        self.kworker = dict(passed=True, completed=True, original_step=baseline.kernel["slurm_step"],
                            manifest_SHA256=self.backend["KERNEL_MANIFEST"], exports=self.backend["EXPORTS"],
                            object=baseline.kernel["path"], object_SHA256=baseline.kernel["SHA256"],
                            actual_device_TUs=1, actual_GPU_steps=0)
        self.proof = dict(request=reader.KERNEL_REQUEST, state="COMPILE_VERIFIED", independent_observation=True,
                          actual_GPU_steps=0, object_SHA256=baseline.kernel["SHA256"],
                          receipt_SHA256=sha256(serialize(self.kworker)).hexdigest(),
                          row=[self.kworker["original_step"], self.backend["KERNEL_LABEL"], "COMPLETED", "0:0",
                               "28", reader.NODE, "cpu=4,mem=8G,node=1"])
        self.kernel.write_bytes(serialize(self.proof))
        self.write(self.intent, dict(request=reader.REQUEST, claim_SHA256=reader.CLAIM_PIN, placement=reader.PLACEMENT,
                                    epoch=self.epoch, deadline=self.epoch+1800, work_deadline=self.epoch+1680,
                                    kernel_step=self.kworker["original_step"], kernel_reconciliation_SHA256=sha256(
                                        self.kernel.read_bytes()).hexdigest(), replay_allowed=False))
        self.recipe = dict(baseline.recipe, pinned_system_header_map=self.plan["system_pins"],
                           expected_providers=self.plan["original169"]["providers"])
        self.recipe_pin = self.write(self.out / "BUILD_RECIPE.json", self.recipe)
        self.original_library = local_elf(3) + b"fixture reference"
        candidate = local_elf(3) + b"fixture candidate"
        (self.out / "libGLM_fresh_kv_pack.so").write_bytes(candidate)
        self.backend["ORIGIN_LIBRARY_SHA"] = sha256(self.original_library).hexdigest()

        def storage(path, expected, *_):
            if str(path) == self.recipe["original_library"]["path"]:
                self.assertEqual(expected, sha256(self.original_library).hexdigest())
                return self.original_library
            if path == self.backend["ORIGIN"] / "build-result.json":
                self.assertEqual(expected, self.backend["ORIGIN_BUILD_SHA"])
                return serialize(self.origin)
            if path == self.plan_path:
                self.assertEqual(expected, sha256(plan_raw).hexdigest())
                return plan_raw
            if path == self.backend["KERNEL"] / "CPU-result.json":
                self.assertEqual(expected, sha256(serialize(self.kworker)).hexdigest())
                return serialize(self.kworker)
            self.assertEqual((path, expected), (baseline.kernel["path"], baseline.kernel["SHA256"]))
            return b"kernel fixture already pinned"

        self.backend["checked"] = storage
        self.hosts = {}
        for index, unit in enumerate(sorted(reader.explicit.HOST)):
            (self.out / f"{unit}.o").write_bytes(local_elf(1) + str(index).encode())
            path = self.recipe["host"][unit]["compile"][-1]
            deps = self.dep_list(unit)
            self.hosts[unit] = dict(paths=len(deps), SHA256=f"{index:x}" * 64)
            for suffix in ("before", "actual"):
                (self.out / f"{unit}.{suffix}.d").write_text(path + ": " + " ".join(deps) + "\n", encoding="utf8")
        passes = [dict(phase=f"{phase}-{unit}", argv=self.recipe["host"][unit][key],
                       returncode=0, owned_child_reaped=True)
                  for phase, key in (("fresh", "preprocess"), ("compile", "compile")) for unit in sorted(self.hosts)]
        passes.append(dict(phase="link", argv=self.recipe["link"], returncode=0, owned_child_reaped=True))
        linked = dict(self.deps(self.recipe_pin, "actual"), library_bytes=len(candidate),
                      library_SHA256=sha256(candidate).hexdigest(), strong_exports=284, providers=71,
                      explicit_Runtime_calls=0, implicit_ldd_initialization="UNKNOWN")
        self.worker = dict(label=reader.explicit.LABEL, request=reader.REQUEST, step=self.row[0], node=reader.NODE,
                           epoch=self.epoch, deadline=self.epoch+1800, work_deadline=self.epoch+1680,
                           claim_SHA256=reader.CLAIM_PIN, planned=reader.explicit.PLANNED, actual=reader.explicit.PLANNED,
                           passed=True, completed=True, phase="complete", GPU_GO=False, Model_GO=False,
                           replay_allowed=False, recipe_SHA256=self.recipe_pin, commands=passes,
                           attempted=dict(fresh_M=3, HOST_compile=3, links=1), linked=linked,
                           fresh_dependency_evidence=self.deps(self.recipe_pin, "fresh"),
                           compile_dependency_evidence=self.deps(self.recipe_pin, "actual"))
        self.symbols = [f"7A50 T original_{i}\n" for i in range(281)]
        self.tools = []

    def dep_list(self, unit):
        return [self.recipe["host"][unit]["generated_source"], "/opt/dtk/include/known.hpp"]

    def deps(self, pin, phase):
        self.assertEqual(pin, self.recipe_pin)
        return dict(phase=phase, recipe_SHA256=pin, hosts=self.hosts)

    def write(self, path, data):
        content = serialize(data)
        path.write_bytes(content)
        return sha256(content).hexdigest()

    def query(self, argv, *, drift=None):
        self.tools.append(argv)
        role = "reference" if argv[-1] == self.recipe["original_library"]["path"] else "candidate"
        if argv[0] == "/usr/bin/nm":
            extras = self.backend["LINK_EXPORTS"] if role == "candidate" else []
            if role == "candidate" and drift == "export":
                extras = self.backend["EXPORTS"]
            return "".join(self.symbols + [f"9c90 T {name}\n" for name in extras])
        self.assertEqual(argv[0], "/usr/bin/readelf")
        dependency = "false.so" if role == "candidate" and drift == "needed" else "libc.so.6"
        return f" (SYMBOLIC) 0x00\n (NEEDED) Shared library: [{dependency}]\n"

    def audit(self, drift=None):
        self.write(self.out / "CPU-result.json", self.worker)
        return reader.audit(self.intent, ("|".join(self.row) + "\n").encode(), self.kernel,
                            self.out, self.backend, query=lambda argv: self.query(argv, drift=drift), deps=self.deps)

    def test_three_unit_reader_uses_only_independent_passive_commands(self):
        packet = self.audit()
        self.assertEqual(packet["state"], "NATIVE_ABI_AND_DEPENDENCIES_VERIFIED")
        self.assertEqual(packet["strong_exports"], 284)
        self.assertEqual(packet["original_worker_provider_count"], 71)
        self.assertEqual(set(packet["dependency_file_SHA256"]), set(reader.explicit.HOST))
        self.assertEqual(set(packet["fresh_host_object_SHA256"]), set(reader.explicit.HOST))
        self.assertIn(reader.explicit.DECODE_EXPORT, packet["added_exports"])
        self.assertEqual([argv[0] for argv in self.tools], ["/usr/bin/nm"] * 2 + ["/usr/bin/readelf"] * 2)
        self.assertFalse(packet["GPU_GO"])

    def test_old_two_unit_or_epoch_cannot_qualify(self):
        self.worker["planned"] = dict(self.worker["planned"], fresh_M=2)
        with self.assertRaisesRegex(ValueError, "worker, epoch or phase"):
            self.audit()
        self.assertEqual(self.tools, [])

    def test_third_fresh_graph_drift_rejects_even_if_summary_equal(self):
        path = self.out / "u00.actual.d"
        path.write_bytes(path.read_bytes() + b" /wrong.h")
        with self.assertRaisesRegex(ValueError, "per-unit M/MD payload differs: u00"):
            self.audit()
        self.assertEqual(self.tools, [])

    def test_compiled_link_export_and_needed_must_match(self):
        for drift in ("export", "needed"):
            with self.subTest(drift=drift), self.assertRaisesRegex(ValueError, "exports or ELF NEEDED"):
                self.audit(drift)

    def test_nonterminal_or_foreign_tres_never_load_artifacts(self):
        for original in ("RUNNING", "COMPLETING"):
            self.row[2] = original
            with self.assertRaisesRegex(ValueError, "foreign or nonterminal"):
                self.audit()
        self.row[2] = "COMPLETED"
        self.row[-1] += ",gres/dcu=0,gres/gpu=1"
        with self.assertRaisesRegex(ValueError, "zero-GRES"):
            self.audit()
        self.assertEqual(self.tools, [])

    def test_terminal_failure_is_preserved_with_no_worker_or_pack_reads(self):
        self.row[2:4] = ["FAILED", "1:0"]
        packet = reader.audit(self.intent, ("|".join(self.row) + "\n").encode(), self.directory / "missing",
                              self.directory / "missing", self.backend)
        self.assertEqual(packet["state"], "ORIGINAL_TERMINAL_UNVERIFIED")
        self.assertEqual(packet["planned"], reader.explicit.PLANNED)
        self.assertIsNone(packet["actual"])
        self.assertEqual(self.tools, [])

    def test_kernel_must_be_bound_to_original_intent(self):
        self.kernel.write_bytes(serialize(dict(self.proof, object_SHA256="a" * 64)))
        with self.assertRaisesRegex(ValueError, "pack reconciliation differs"):
            self.audit()
        self.assertEqual(self.tools, [])


if __name__ == "__main__":
    unittest.main()
