"""Exercise ownership, order, memory and copy timing with a recording dispatcher."""

from copy import deepcopy
from contextlib import contextmanager
import ast
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class Mode:
    current = None

    def __enter__(self):
        if Mode.current is not None:
            raise AssertionError("nested fixture dispatch")
        Mode.current = self
        return self

    def __exit__(self, *_):
        Mode.current = None


class Mask:
    dtype = "bool"
    device = "cuda:0"
    shape = (1, 4)
    requires_grad = False
    copies = 0

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 8, data_ptr=lambda: id(self))

    def numel(self):
        return 4

    def stride(self):
        return (4, 1)

    def detach(self):
        return self

    def cpu(self):
        Mask.copies += 1
        if Mode.current is not None:
            raise AssertionError("host copy occurred inside a native dropout")
        return self

    def contiguous(self):
        return self

    def reshape(self, _):
        return self

    def view(self, _):
        return self

    def numpy(self):
        return SimpleNamespace(tobytes=lambda: bytes([1, 0, 0, 1]))


class Op:
    def __init__(self, name, result):
        self._schema = SimpleNamespace(name=name)
        self.result = result
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


class Dropout:
    p = .05
    inplace = False
    training = True

    def __init__(self, op):
        self.op = op

    def forward(self, value):
        return Mode.current.__torch_dispatch__(self.op, (), (value, .05, True), {})[0]


def load():
    torch = ModuleType("torch")
    torch.bool, torch.uint8, torch.nn = "bool", "uint8", SimpleNamespace(Dropout=Dropout)
    utility = ModuleType("torch.utils._python_dispatch")
    utility.TorchDispatchMode = Mode
    spec = importlib.util.spec_from_file_location("native_mask_fixture", Path(__file__).with_name("tap.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "torch.utils._python_dispatch": utility}):
        spec.loader.exec_module(module)
    return module


class Tap(unittest.TestCase):
    def setUp(self):
        self.tap = load()
        self.output = object()
        self.a = Op("aten::native_dropout", (self.output, Mask()))
        self.b = Op("aten::native_dropout", (self.output, Mask()))
        self.ma, self.mb = Dropout(self.a), Dropout(self.b)
        self.model = SimpleNamespace(named_modules=lambda: [
            ("layer1.lora_dropout.default", self.ma), ("layer2.lora_dropout.default", self.mb)])
        self.value = SimpleNamespace(shape=(1, 4), device="cuda:0")
        Mask.copies = 0

    def observe(self, limit):
        return self.tap.MaskTap(self.model, max_retained_bytes=limit, expected_calls=2)

    def test_observation_preserves_result_and_defers_copies(self):
        with self.observe(16) as run:
            self.assertIs(self.ma.forward(self.value), self.output)
            self.assertIs(self.mb.forward(self.value), self.output)
            self.assertEqual(Mask.copies, 0)
        self.assertIs(self.ma.forward.__func__, Dropout.forward)
        self.assertIs(self.mb.forward.__func__, Dropout.forward)
        result = run.seal()
        self.assertEqual((self.a.calls, self.b.calls, Mask.copies), (1, 1, 2))
        self.assertEqual(result["charged_device_storage_bytes"], 16)
        self.assertEqual(result["logical_mask_bytes_moved_to_host"], 8)
        self.assertEqual([row["ordinal"] for row in result["mask_fingerprints"]], [0, 1])
        self.assertNotIn("mask", result["mask_fingerprints"][0])
        self.assertEqual(run.records, [])
        with self.assertRaises(ValueError):
            run.seal()

    def test_budget_and_unexpected_provider_fail_closed(self):
        run = self.observe(15)
        with self.assertRaisesRegex(ValueError, "budget"), run:
            self.ma.forward(self.value)
            self.mb.forward(self.value)
        self.assertEqual(run.records, [])
        self.assertIs(self.ma.forward.__func__, Dropout.forward)
        self.a._schema.name = "aten::bernoulli_"
        with self.assertRaisesRegex(ValueError, "bool-mask"), self.observe(16):
            self.ma.forward(self.value)
        self.assertIs(self.mb.forward.__func__, Dropout.forward)

    def test_repeated_or_partial_module_releases_masks(self):
        run = self.observe(16)
        with self.assertRaises(ValueError), run:
            self.ma.forward(self.value)
            self.ma.forward(self.value)
        self.assertFalse(run.records)
        with self.observe(16) as partial:
            self.ma.forward(self.value)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            partial.seal()
        self.assertFalse(partial.records)

    def test_compare_full_order_and_first_mismatch(self):
        rows = [dict(module=f"layer{i}", ordinal=i, shape=[4], mask_stride=[1], provider="aten::native_dropout",
                     mask_bytes=4, mask_SHA256="a" * 64) for i in range(186)]
        baseline = dict(native_mask_calls=186, mask_fingerprints=rows)
        changed = deepcopy(baseline)
        changed["mask_fingerprints"][155]["mask_SHA256"] = "b" * 64
        self.assertEqual(self.tap.compare(baseline, changed), {
            "compared_native_masks": 186, "different_native_masks": 1,
            "first_different_module": {"module": "layer155", "ordinal": 155}})
        changed["mask_fingerprints"][155]["ordinal"] = 156
        with self.assertRaises(ValueError):
            self.tap.compare(baseline, changed)
        rows[0]["mask_SHA256"] = "invalid"
        with self.assertRaises(ValueError):
            self.tap.compare(baseline, baseline)

    def test_profile_scope_only_observes_treatment_and_seals_after_profiler(self):
        source = Path(__file__).with_name("native_plugin.py")
        parsed = ast.parse(source.read_bytes(), filename=source.name)
        wrapper = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "profile_scope")
        calls = []

        @contextmanager
        def profiler():
            calls.append("profiler-enter")
            yield "native-profiler"
            calls.append("profiler-exit")

        class Observer:
            def __init__(self, model, max_retained_bytes):
                self.assertion = (model, max_retained_bytes)

            def __enter__(self):
                calls.append(("mask-enter", self.assertion))

            def __exit__(self, *_):
                calls.append("mask-exit")

            def seal(self):
                calls.append("seal")
                return {"mask": "digest-only"}

        scope = dict(contextmanager=contextmanager, MaskTap=Observer, require=self.tap.require,
                     ARMS=("u1", "c1", "c2", "u2"))
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), source.name, "exec"), scope)
        for arm in ("u1", "c1", "c2", "u2"):
            calls.clear()
            observed = []
            with scope["profile_scope"](profiler, "model", arm, 50, observed) as value:
                self.assertEqual(value, "native-profiler")
                calls.append("real-forward")
            if arm.startswith("u"):
                self.assertEqual((calls, observed), (["profiler-enter", "real-forward", "profiler-exit"], [None]))
            else:
                self.assertEqual(calls, ["profiler-enter", ("mask-enter", ("model", 50)),
                                         "real-forward", "mask-exit", "profiler-exit", "seal"])
                self.assertEqual(observed, [{"mask": "digest-only"}])

    def test_pins_match_the_actual_original_callback_sources(self):
        plugin = Path(__file__).with_name("native_plugin.py").resolve()
        syntax = ast.parse(plugin.read_bytes(), filename=plugin.name)
        binder = next(node for node in syntax.body if isinstance(node, ast.Assign)
                      and any(isinstance(item, ast.Name) and item.id == "ORIGINAL" for item in node.targets))
        expected = ast.literal_eval(binder.value)
        older = plugin.parent.with_name("training-minicpm-checkpoint20-first-forward-source-v1")
        self.assertEqual(set(expected), {"native_observer.py", "probe.py"})
        for name, (size, pin) in expected.items():
            raw = (older / name).read_bytes()
            self.assertEqual((len(raw), sha256(raw).hexdigest()), (size, pin))


if __name__ == "__main__":
    unittest.main(verbosity=2)
