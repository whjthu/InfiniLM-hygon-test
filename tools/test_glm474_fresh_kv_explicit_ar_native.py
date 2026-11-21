from copy import deepcopy
from hashlib import sha256
from pathlib import PurePosixPath
from unittest import TestCase

import prepare_glm474_fresh_kv_explicit_ar_build as explicit
import run_glm474_fresh_kv_explicit_ar_once as worker
import test_prepare_glm474_fresh_kv_native_build as original


class ExplicitNativeTest(TestCase):
    def setUp(self):
        self.build = explicit.backend()
        self.run = worker.runner()
        fixture = original.FreshKVNativeBuildTest()
        fixture.setUp()
        self.origin = fixture.build
        self.kernel = fixture.kernel
        plan = fixture.plan
        u00 = next(row for row in plan["units"] if row["name"] == "u00")
        original_src = str(self.build["ORIGIN"] / "B/project-mirror") + u00["source"].removeprefix("/refactor")
        self.origin["commands"].append(dict(label="B-u00-compile", rc=0, owned_child_reaped=True,
            argv=["/usr/bin/g++", *plan["new_common_flags"]["HOST"], "-MD", "-MF", "/owned.d",
                  "-MT", self.origin["objects"]["u00"]["path"], "-c", original_src,
                  "-o", self.origin["objects"]["u00"]["path"]]))
        _, self.pins = explicit.caller_sources()
        self.recipe = self.build["make_recipe"](self.origin, self.pins, self.kernel)

    def test_link_replaces_only_the_three_frozen_callers(self):
        self.assertEqual(self.recipe["planned"], explicit.PLANNED)
        self.assertEqual(set(self.recipe["host"]), set(explicit.HOST))
        self.assertEqual(len(self.recipe["reuse"]), 29)
        native = self.build["NATIVE"]
        replaced = [str(native / (name + ".o")) if name in explicit.HOST else entry["path"]
                    for name, entry in self.origin["objects"].items()]
        self.assertEqual([arg for arg in self.recipe["link"] if arg.endswith(".o")], [*replaced, self.kernel["path"]])
        self.assertLess(self.recipe["link"].index(self.kernel["path"]), self.recipe["link"].index("-lcudart"))
        self.assertEqual(self.recipe["linked_library"], str(native / "libGLM_fresh_kv_pack.so"))
        self.assertEqual(PurePosixPath(self.recipe["host"]["u00"]["generated_source"]).name,
                         "full_bridge_observer.cpp")

    def test_bridge_source_and_phase_headers_are_required(self):
        build = deepcopy(self.origin)
        build["commands"][-1]["owned_child_reaped"] = False
        with self.assertRaisesRegex(ValueError, "unqualified original command"):
            self.build["make_recipe"](build, self.pins, self.kernel)
        namespace = self.build["NATIVE"]
        unit = self.recipe["host"]["u00"]
        prior = {unit["original_source"]: "old", "/old/runtime.hpp": "runtime"}
        graph = {unit["generated_source"]: unit["generated_source_SHA256"], "/old/runtime.hpp": "runtime"}
        graph.update({str(namespace / "src" / name): pin for name, pin in unit["required_generated_includes"].items()})
        self.build["check_host_graph"](prior, graph, unit, {})
        graph.pop(str(namespace / "src/fresh_kv_explicit_ar.hpp"))
        with self.assertRaisesRegex(ValueError, "dependency differs"):
            self.build["check_host_graph"](prior, graph, unit, {})

    def test_link_requires_decode_export_without_changing_pack_object_exports(self):
        pack = self.build["EXPORTS"]
        self.assertEqual(pack, ["glm_r474_fresh_kv_pack_t128", "glm_r474_fresh_kv_pack_t64"])
        known = {f"original_{n}": "T" for n in range(281)}
        pack_only = {**known, **dict.fromkeys(pack, "T")}
        with_decode = {**pack_only, explicit.DECODE_EXPORT: "T"}
        providers = {f"/opt/lib{n}.so": "pin" for n in range(71)}
        check = self.build["linked_scope"]
        check(known, with_decode, ["libc.so"], ["libc.so"], providers, providers)
        for symbols in (pack_only, dict(with_decode, unknown="T")):
            with self.subTest(count=len(symbols)), self.assertRaisesRegex(ValueError, "strong export"):
                check(known, symbols, ["libc.so"], ["libc.so"], providers, providers)

    def exercise_phases(self, break_at=None):
        actual = dict.fromkeys(explicit.PLANNED, 0)
        state = dict(actual=actual, attempted=dict(fresh_M=0, HOST_compile=0, links=0), passed=False)
        seen = []

        def check(phase):
            seen.append(phase)
            if break_at == phase:
                raise ValueError(phase)
            return {}

        def run():
            self.run["phase_runner"](self.recipe, lambda label, _, timeout: seen.append(label),
                check, lambda _: check("reused"), lambda: check("verify-link"), state, lambda: None)

        return run, seen, state

    def test_all_fresh_M_and_actual_MD_ordered_before_their_consumers(self):
        run, seen, state = self.exercise_phases()
        run()
        self.assertEqual(seen, ["fresh-u00", "fresh-u15", "fresh-u17", "fresh", "compile-u00",
                                "compile-u15", "compile-u17", "actual", "reused", "link", "verify-link"])
        self.assertEqual(state["actual"], explicit.PLANNED)
        self.assertTrue(state["passed"])

    def test_dependency_errors_keep_failed_phases_out_of_link(self):
        for phase, expected_compiles in (("fresh", 0), ("actual", 3), ("reused", 3)):
            run, seen, state = self.exercise_phases(phase)
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, phase):
                run()
            self.assertEqual(state["actual"]["HOST_compile"], expected_compiles)
            self.assertNotIn("link", seen)
            self.assertFalse(state["passed"])

    def test_actual_claim_checks_new_worker_and_isolation(self):
        claim = self.run["CLAIM"].read_bytes()
        bound = self.run["validate_claim"](sha256(claim).hexdigest())
        self.assertEqual(bound["host_units"], ["u00", "u15", "u17"])
        self.assertEqual(bound["placement"]["DCU"], 0)
        self.assertFalse(bound["GPU_GO"])
        self.assertTrue(callable(self.run["owned_cleanup"]()))
        environment = dict(SLURM_JOB_ID="1103606", SLURM_STEP_ID="222", HIP_VISIBLE_DEVICES="-1",
                           CUDA_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
        self.run["validate_placement"](environment, self.run["NODE"], 4)
        with self.assertRaisesRegex(ValueError, "zero-GRES"):
            self.run["validate_placement"](dict(environment, SLURM_STEP_GPUS="0"), self.run["NODE"], 4)
