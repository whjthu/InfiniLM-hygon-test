"""Exercise the exact staged T512 HOST package using CPU-only fixtures."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[3] / "tools")]
import host_once as H
import prepare_stage as S
import stage_packets as X
import worker as W


def origin_fixture(recipes):
    directory = S.REPO / S.DEPTH
    plan = json.loads((directory / "PLAN.json").read_bytes())
    binding = json.loads((directory / "BASELINE_BINDING.json").read_bytes())
    path = str(recipes["ORIGIN"] / "B/libGLM_depth_observer_B.so")
    build = dict(passed=True, step="1103606.308",
                 library=dict(path=path, SHA256=recipes["ORIGIN_LIBRARY_SHA"]),
                 actual={"objects":32}, objects={}, fresh_M={}, commands=[])
    remap = {}
    for unit in plan["units"]:
        name = unit["name"]
        output = (str(recipes["ORIGIN"] / "B/objects" / (name+".o")) if name in plan["affected_units"]
                  else binding["units"][name]["original_object"])
        build["objects"][name] = dict(path=output, SHA256="b"*64, bytes=32)
        build["fresh_M"][name] = dict(pins_file=str(recipes["ORIGIN"] / "B" / (name+".dependency-pins.json")),
                                      pins_SHA256="f"*64)
        remap[unit["baseline169_object"]] = output
        if name in W.HOST:
            source = str(recipes["ORIGIN"] / "B/project-mirror") + posixpath.normpath(unit["source"])[len("/refactor"):]
            command = ["/usr/bin/g++", *plan["new_common_flags"]["HOST"], "-MD", "-MF", "/fixture.d",
                       "-MT", output, "-c", source, "-o", output]
            build["commands"].append(dict(label="B-"+name+"-compile", rc=0, owned_child_reaped=True, argv=command))
    args = [remap.get(arg,arg) for arg in plan["original169"]["link_argv"]]
    args[args.index("-o")+1] = path
    build["commands"].append(dict(label="B-link", rc=0, owned_child_reaped=True, argv=args))
    return build, plan


class PackageTest(unittest.TestCase):
    def test_noncanonical_requests_are_rejected(self):
        W.require_request(W.REQUEST)
        for bad in (W.REQUEST[:-1], W.REQUEST+"0", W.REQUEST.upper(), "a"*32, None):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "canonical original 32-hex"):
                W.require_request(bad)

    def test_exact_stage_and_finite_receiver_from_normal_generator(self):
        files, receipt = S.generate()
        self.assertEqual(len(files), 39)
        self.assertIsNotNone(re.fullmatch("[0-9a-f]{32}", receipt["request"]))
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            checkout = root / "new"
            self.assertEqual(S.materialize(checkout), receipt)
            packets = X.programs(checkout, receipt["claim_SHA256"], receipt["stage_manifest_SHA256"])
            self.assertLessEqual(max(map(len, packets)), 131072)
            self.assertLessEqual(len(packets), 13)
            destination = root / "target"
            (destination / "incoming").mkdir(parents=True)
            for packet in packets:
                namespace = W.execute(Path("stage_packet.py"), packet)
                state = namespace["receive"](destination)
                self.assertEqual(state["actual_new_Slurm_steps"], 0)
            self.assertEqual(state["files"], len(files))
            installed = destination / "incoming" / (W.LABEL+"-reviewed-checkout")
            self.assertEqual({name:(installed/name).read_bytes() for name in files}, files)
            with self.assertRaises(FileExistsError):
                S.materialize(checkout)
            with self.assertRaises(ValueError):
                X.programs(checkout, "0"*64, receipt["stage_manifest_SHA256"])

    def test_new_layout_dependencies_and_only_original_objects_after_substitution(self):
        kernel, prior, recipes = W.backend()
        baseline = prior["backend"]()
        origin, plan = origin_fixture(recipes)
        files, _ = S.generate()
        old_manifest = json.loads((prior["CALLER"] / "SOURCE_FILES.json").read_bytes())
        new_manifest = json.loads(files[(S.CALLER / "SOURCE_FILES.json").as_posix()])
        obj = dict(path=str(recipes["KERNEL"] / "fresh_kv_pack.o"), SHA256=kernel.OBJECT_SHA, slurm_step=kernel.ROW[0])
        original = baseline["make_recipe"](origin, old_manifest["generated_sources_SHA256"], obj)
        changed = recipes["make_recipe"](origin, new_manifest["generated_sources_SHA256"], obj)
        self.assertEqual(original["reuse"], changed["reuse"])
        self.assertEqual(original["kernel"], changed["kernel"])
        self.assertEqual(len(changed["reuse"]), 29)
        swapped = lambda args: [arg.replace(str(original_root), str(recipes["NATIVE"])) for arg in args]
        original_root = baseline["NATIVE"]
        self.assertEqual(swapped(original["link"]), changed["link"])
        for name in W.HOST:
            self.assertEqual(swapped(original["host"][name]["compile"]), changed["host"][name]["compile"])
            self.assertEqual(set(changed["host"][name]["required_generated_includes"]),
                             set(original["host"][name]["required_generated_includes"]) | set(W.MORE_INCLUDES[name]))
        self.assertEqual(len(plan["original169"]["providers"]), 71)
        unit = changed["host"]["u17"]
        old = {unit["original_source"]:"0"*64, "/fixture/known_header.hpp":"1"*64}
        actual = {"/fixture/known_header.hpp":"1"*64, unit["generated_source"]:unit["generated_source_SHA256"]}
        actual.update({str(recipes["NATIVE"]/"src"/n):pin for n,pin in unit["required_generated_includes"].items()})
        recipes["check_host_graph"](old, actual, unit, {})
        actual.pop(str(recipes["NATIVE"]/"src/fresh_kv_prefill512_materialize.inc"))
        with self.assertRaisesRegex(ValueError, "dependency differs"):
            recipes["check_host_graph"](old, actual, unit, {})

    def test_verified_builder_fences_before_link(self):
        kernel, _, recipes = W.backend()
        source = json.loads((W.CALLER/"SOURCE_FILES.json").read_bytes())["generated_sources_SHA256"]
        origin, _ = origin_fixture(recipes)
        obj = dict(path=str(recipes["KERNEL"]/"fresh_kv_pack.o"), SHA256=kernel.OBJECT_SHA, slurm_step=kernel.ROW[0])
        recipe = recipes["make_recipe"](origin, source, obj)
        phase = W.worker()["phase_runner"]
        trace = []
        report = dict(attempted={"fresh_M":0,"HOST_compile":0,"links":0}, actual=dict.fromkeys(W.PLANNED,0))

        def deps(name):
            trace.append(name)
            if name == "actual":
                raise ValueError("actual MD differs from the fresh M")

        with self.assertRaisesRegex(ValueError, "actual MD"):
            phase(recipe, lambda label,*_:trace.append(label), deps, lambda _:trace.append("reuse"),
                  lambda:trace.append("link-proof"), report, lambda:None)
        self.assertEqual(trace, ["fresh-u00","fresh-u15","fresh-u17","fresh","compile-u00","compile-u15","compile-u17","actual"])
        self.assertEqual(report["attempted"]["links"], 0)

    def test_original_918_archive_exact_scope_and_added_symbol_only(self):
        original = W.verify_original_archive()
        actual = original["actual_accounting"] + [["1103606.920","post","COMPLETED","0:0","4","f12r2n13","cpu=1,node=1"]]
        W.verify_original_archive(lambda:actual)
        bad = deepcopy(actual)
        bad[-1][-1] += ",gres/dcu=1"
        with self.assertRaisesRegex(ValueError, "zero-GRES"):
            W.verify_original_archive(lambda:bad)
        base = {f"strong_{index}":"T" for index in range(284)}
        new = {**base, W.NEW_EXPORT:"T"}
        W.additions(base, new, ["libexample.so"], ["libexample.so"])
        for modified in ({**new,"extra":"T"}, {**new,W.NEW_EXPORT:"W"}):
            with self.assertRaisesRegex(ValueError, "strong map"):
                W.additions(base, modified, [], [])

    def test_exact_nested_HOST_keeps_zero_GRES_and_single_epoch(self):
        kernel, prior = W.baseline()
        driver = H.original_controller("a"*40,"b"*64,kernel)
        argv = driver["command"](PurePosixPath("/cluster"), PurePosixPath("/cluster/incoming/reviewed"),
                                 PurePosixPath("/cluster/results/target"), 1795330000., "5"*64, 1740)
        self.assertIn("--cpu-bind=threads",argv)
        self.assertIn("--gres=none",argv)
        self.assertIn("--job-name="+W.LABEL,argv)
        self.assertIn("1795331680.0",argv[-1])
        self.assertIn("1795331800.0",argv[-1])
        self.assertIn(W.LABEL+"-checkout",argv[-1])
        self.assertNotIn("explicit-ar-Native-CPU-v2",argv[-1])
        reader = W.reader_template(prior)
        self.assertTrue(callable(reader["verified_artifacts"]))


if __name__ == "__main__":
    unittest.main()
