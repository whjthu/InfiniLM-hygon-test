"""Offline regression of the generated callbacks and the one-forward metadata observer."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from first_tail import FirstTail, for_rank, rng_hashes
from prepare import CONTINUOUS, RESUME, WITNESS, overlay, plan


def reject(fn, expected):
    try:
        fn()
    except ValueError as error:
        assert str(error) == expected, str(error)
    else:
        raise AssertionError(expected)


def cpu_fixture(directory):
    python, numpy = random.Random(42), np.random.RandomState(42)
    draws = {"rng": 0, "context": 0, "layout": 0}

    def fingerprint(value):
        return {"SHA256": sha256(repr(value).encode()).hexdigest()}

    def snapshot():
        draws["rng"] += 1
        value = dict(python=python.getstate(), numpy=numpy.get_state(), cpu=b"fixture CPU RNG",
                     cuda=[b"fixture assigned HIP RNG"])
        return rng_hashes(value, fingerprint, lambda item: item)

    def context(*_, **__):
        draws["context"] += 1
        return dict(module_training={"internal": True}, training_modules=1, total_modules=1, grad_enabled=True)

    def layout(*_, **__):
        draws["layout"] += 1
        return {"kwargs/input_ids": dict(dtype="torch.int64", shape=[1, 512], stride=[512, 1])}

    initial = snapshot()
    draws["rng"] = 0
    probe = FirstTail(0, "resumed", None, snapshot, context, layout)
    model_input = dict(ordinal=1, full_input={"SHA256": "a" * 64})
    loss = dict(ordinal=1, loss={"SHA256": "b" * 64}, loss_scalar=.01)
    path = directory / "first-tail-forward.json"
    for step, micro in ((19, 1), (20, 2), (20, 3), (21, 1)):
        probe.before(step, micro, (), {}, model_input)
        probe.after(step, micro)
        probe.write(step, micro, loss, path)
    assert set(draws.values()) == {0} and not path.exists()
    assert all(for_rank(rank, "resumed", None, None, None, None, None) is None for rank in range(2, 8))
    probe.before(20, 1, (), {"ignored_by_fixture": "do-not-serialize"}, model_input)
    reject(lambda: probe.before(20, 1, (), {}, model_input), "ONE_EXACT_FIRST_FORWARD_ENTRY")
    probe.after(20, 1)
    reject(lambda: probe.after(20, 1), "ONE_EXACT_FIRST_FORWARD_RETURN")
    probe.write(20, 1, loss, path)
    reject(lambda: probe.write(20, 1, loss, path), "MATCHED_FIRST_FORWARD_INPUT_AND_LOSS")
    assert draws == {"rng": 2, "context": 2, "layout": 1}
    raw = path.read_bytes()
    value = json.loads(raw)
    assert value["before"]["RNG"] == value["after"]["RNG"] == initial == snapshot()
    assert value["loss"]["loss_scalar"] == .01 and probe.SHA256 == sha256(raw).hexdigest()
    assert b"do-not-serialize" not in raw and b"module_training" not in raw
    assert set(value["before"]["RNG"]) == {"cpu", "assigned_HIP", "numpy", "python"}


def extract_method(file, cls, name):
    tree = ast.parse(file.read_bytes())
    type_def, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    method, = (node for node in type_def.body if isinstance(node, ast.FunctionDef) and node.name == name)
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(file), "exec"), ns)
    return ns[name]


def integration_fixture(out):
    stop = extract_method(out / WITNESS, "ResumeWitness", "request_diagnostic_stop")
    end = extract_method(out / RESUME, "GemmaDomainResume", "on_step_end")
    state = SimpleNamespace(global_step=21)
    control = SimpleNamespace(should_training_stop=False, should_save=True)
    witness = SimpleNamespace(trainer=SimpleNamespace(state=state), lightweight_first_tail=True,
                              diagnostic_stop_step=21, stop_step=21, entry_state_captured=False,
                              steps=[], outputs=[0, 1, 2, 3], planned_inputs=4, expected_saved=[],
                              first_tail=SimpleNamespace(written=True), diagnostic_stop_requested=False,
                              progress=lambda: None, guarded=lambda fn: fn)
    witness.step_end = lambda: witness.steps.append({"global_step": 21})
    witness.request_diagnostic_stop = lambda c: stop(witness, c)
    plugin = SimpleNamespace(witness=witness, skip_seen=[True])
    assert end(plugin, None, state, control) is control
    assert control.should_training_stop and not control.should_save and witness.diagnostic_stop_requested
    witness.outputs.pop()
    reject(lambda: stop(witness, control), "cannot stop before the declared complete observation update")
    witness.first_tail.written = False
    reject(lambda: end(plugin, None, state, control), "missing rank0/1 first-tail forward metadata")
    for name in (CONTINUOUS, RESUME):
        tree = ast.parse((out / name).read_bytes())
        assert any(isinstance(node, ast.Attribute) and node.attr == "first_tail" for node in ast.walk(tree))
    assert "entry_state_step=20" not in (out / RESUME).read_text()


def main():
    repo = Path(__file__).resolve().parents[3]
    code = repo / "code/bw1000"
    original = code / "training-gemma-usgs72-fresh-initial-domain25-v1"
    context = code / "training-minicpm-forward-entry-state-CPU-v1/framework/vlm/runtime/forward_context_state.py"
    compiled = plan(original / "resumed-plan.json", "a" * 32, 40921)
    source = json.loads((original / "resumed-plan.json").read_bytes())
    assert (compiled["restored_global_step"], compiled["native_skip"], compiled["nominal_max_steps"],
            compiled["stop_after_update"], compiled["forwards_per_rank_planned"]) == (20, 80, 25, 21, 4)
    assert compiled["resume_from_checkpoint"] == source["resume_from_checkpoint"]
    assert compiled["training_ready"] is False and compiled["environment_overrides"]["NPROC_PER_NODE"] == "8"
    limits = {"--max_steps": "25", "--gradient_accumulation_steps": "4", "--dataloader_num_workers": "4"}
    for key, value in limits.items():
        command = compiled["command"]
        assert command[command.index(key) + 1] == value
    assert sum(a != b for a, b in zip(compiled["command"], source["command"], strict=True)) == 3
    with TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        cpu_fixture(root)
        out = root / "overlay"
        overlay(original / "overlay", context, out)
        integration_fixture(out)
    print("exact source derivation, real Python/NumPy RNG read-only, rank exclusion, first-tail-only witness and normal step-21 stop PASS; CPU/HIP tensor RNG accessors are fixtures")


if __name__ == "__main__":
    main()
