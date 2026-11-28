"""Check the real answerless contract and output reader with local fixtures."""

import ast
import json
import runpy
import sys
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from contract import ARMS, CAP, PROMPT, SLOTS, generation, inputs, put, record, seal, sha, wire
import native


def rejects(fn, reason):
    try:
        fn()
    except ValueError as error:
        assert str(error) == reason, str(error)
    else:
        raise AssertionError(reason)


def exercise_base_gate(root):
    class Model:
        _vlm_onevision_rope_initialized = True
        config = SimpleNamespace(_attn_implementation="eager")

        def parameters(self):
            return (value for _, value in self.named_parameters())

        def named_parameters(self):
            return iter(self.members)

    bf16, fp32 = "torch.bfloat16", "torch.float32"
    backend = SimpleNamespace(bfloat16=bf16)
    expected = SimpleNamespace(device="cuda:0", dtype=bf16, is_floating_point=lambda: True)
    unexpected = SimpleNamespace(device="cuda:0", dtype=fp32, is_floating_point=lambda: True)
    model = Model()
    members = [(f"hidden_{i}.weight", expected) for i in range(698)]
    model.members = members.copy()
    processor = SimpleNamespace(model_meta=SimpleNamespace(template="llava_onevision1_5"))
    assignment = {field: {"count": 0, "examples": ["secret_key_never_logged"]} for field in native.FINALIZER_FIELDS}
    scope = {"active": False, "calls": 0}
    audited = []
    @contextmanager
    def observe_checkpoint_assignment():
        assert not scope["active"]
        scope["active"] = True
        try:
            recover.audit({"examples": ["secret_key_never_logged"]})
            yield [assignment]
        finally:
            scope["active"] = False

    recover = SimpleNamespace(audit=audited.append, observe_checkpoint_assignment=observe_checkpoint_assignment)
    original_log = recover.audit
    def loader(path, **kwargs):
        assert scope["active"] and path == "/scope/model"
        assert kwargs == dict(model_type="llava_onevision1_5", load_model=True, download_model=False,
                              device_map={"": 0}, attn_impl="eager", torch_dtype=bf16)
        scope["calls"] += 1
        return model, processor

    def checked(records, expected_rejection=None):
        finalizer, predicates = native.assignment_projection(records)
        current = {"base_diagnostic": dict(checkpoint_finalizer=finalizer, predicates=predicates)}
        check = lambda: native.original_base_gate(model, processor, Model, backend, current)
        if expected_rejection is None:
            check()
        else:
            rejects(check, "ACTUAL_ORIGINAL_EAGER_BASE")
            assert current["base_diagnostic"]["predicates"][expected_rejection] is False
        return current

    good = {}
    loaded = native.load_original_model(recover, loader, "/scope/model", backend, good)
    assert loaded == (model, processor) and scope == {"active": False, "calls": 1}
    assert audited == [] and recover.audit is original_log
    native.original_base_gate(model, processor, Model, backend, good)
    assert len(good["base_diagnostic"]["predicates"]) == 13
    assert all(good["base_diagnostic"]["predicates"].values())
    for count in (697, 699, 1202):
        model.members = [(f"hidden_{i}.weight", expected) for i in range(count)]
        checked([assignment], expected_rejection="parameter_tensors_698")
    model.members = members.copy()
    model.members[0] = ("hidden.lora_A.default.weight", expected)
    checked([assignment], expected_rejection="zero_lora_parameters")
    model.members = members.copy()
    for inventory in ([], [assignment, assignment]):
        failed = checked(inventory, expected_rejection="finalizer_called_once")
        assert failed["base_diagnostic"]["checkpoint_finalizer"]["calls"] == len(inventory)
    for field in native.FINALIZER_FIELDS:
        item = dict(assignment, **{field: {"count": 1}})
        failed = checked([item], expected_rejection=f"finalizer_{field}_zero")
        assert failed["base_diagnostic"]["checkpoint_finalizer"][field] == 1

    model.members[0] = ("secret_but_lora.lora_A.default.weight", unexpected)
    model.peft_config = {"secret_peft_key": object()}
    output = root / "diagnostic"
    opts = SimpleNamespace(request="synthetic-seal-fixture", output=output, requests_sha="d" * 64, images_sha="f" * 64)
    def installed_gate(args, owner, facts, current):
        current["phase"] = "load_base"
        native.load_original_model(recover, loader, "/scope/model", backend, current)
        facts["model_loads"] += 1
        native.original_base_gate(model, processor, Model, backend, current)

    with patch.object(native, "authority", return_value={"step": "1103606.9000", "deadline": 10**10}), \
         patch.object(native, "native", installed_gate), patch.object(native.signal, "signal"), \
         patch.object(native.signal, "setitimer", create=True), \
         patch.object(native.signal, "SIGALRM", 14, create=True), \
         patch.object(native.signal, "ITIMER_REAL", 0, create=True), redirect_stdout(io.StringIO()):
        assert native.run(opts) == 2
    data = (output / "result.json").read_bytes()
    report = json.loads(data)
    diag = report["failure"]["base_diagnostic"]
    assert report["failure"]["detail_SHA256"] == sha(b"ACTUAL_ORIGINAL_EAGER_BASE")
    assert len(diag["predicates"]) == 13 and {k for k, v in diag["predicates"].items() if not v} == {
        "peft_config_absent", "only_floating_bfloat16", "zero_lora_parameters"}
    assert diag["lora_parameter_tensors"] == 1 and diag["peft_config"] == {"present": True, "type": "dict", "count": 1}
    assert {row["dtype"]: row["tensors"] for row in diag["dtype_device_counts"]} == {bf16: 697, fp32: 1}
    assert b"secret" not in data and report["facts"]["model_loads"] == 1 and report["facts"]["actual_generate_calls"] == 0
    assert report["planned"] == 32 and all(arm["missing"] == 16 for arm in report["counts"].values())
    assert len((output / "predictions.jsonl").read_bytes().splitlines()) == 32


def exercise_paired_cache_loop(tree):
    """Run the exact compiled pairing loop with a stand-in for native Swift inference."""
    native_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "native")
    loop = next(node for node in native_fn.body if isinstance(node, ast.With))
    code = compile(ast.Module(body=[loop], type_ignores=[]), "actual_val16_cache_scope", "exec")
    counters = {arm: dict(slots=0, decisions=0, none_positions=0, supplied_positions=0) for arm in ARMS}
    observed, rows = [], []
    state = SimpleNamespace(events=[])
    current = {}
    model = object()

    @contextmanager
    def install(actual_model, *_):
        assert actual_model is model
        observed.append("installed")
        try:
            yield state
        finally:
            observed.append("restored")

    def infer(*_, adapter_request, **__):
        assert observed == ["installed"]
        assert not state.events
        assert (adapter_request is None) is (current["arm"] == "base")
        state.events.extend([dict(position="none_native_forward_fallback", drop_visual=False),
                             dict(position="supplied_unchanged", drop_visual=True)])
        current.update(ids=[4, 9], eos=[9])
        return [SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", token_ids=[4, 9],
                                  message=SimpleNamespace(content="fixture"))])]

    env = dict(install=install, model=model, native_class=None, torch=None, dynamic_cache=None,
               generation_mixin=None, ARMS=ARMS, AdapterRequest=SimpleNamespace, ADAPTER=native.ADAPTER,
               args=SimpleNamespace(input=Path("fixture"), output=None, checkpoint="CP25"), prompts=[
                   dict(slot=0, messages=[], images=["fixture.png"])], current=current,
               engine=SimpleNamespace(infer=infer, tokenizer=SimpleNamespace(pad_token_id=0)),
               InferRequest=SimpleNamespace, cfg=None, generation=lambda *args: "eos", need=native.need,
               record=lambda _, row: rows.append(row), cache_counts=native.cache_counts, decisions=counters)
    exec(code, env)
    assert observed == ["installed", "restored"]
    assert [(row["arm"], row["outcome"]) for row in rows] == [("base", "eos"), ("cp25", "eos")]
    assert all(value == dict(slots=1, decisions=2, none_positions=1, supplied_positions=1)
               for value in counters.values())
    rejects(lambda: native.cache_counts([]), "NATIVE_VISUAL_PREFILL_AND_DECODE_DECISIONS")
    rejects(lambda: native.cache_counts([dict(position="supplied_unchanged", drop_visual=True)]),
            "NATIVE_VISUAL_PREFILL_AND_DECODE_DECISIONS")

    observed.clear()
    rows.clear()
    env["engine"].infer = lambda *args, **kwargs: [SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", token_ids=[4, 9], message=SimpleNamespace(content="fixture"))])]
    rejects(lambda: exec(code, env), "NATIVE_VISUAL_PREFILL_AND_DECODE_DECISIONS")
    assert observed == ["installed", "restored"] and rows == [
        dict(arm="base", slot=0, outcome="error", text=None)]


def main():
    archive = (Path(__file__).resolve().parents[3] / "results/bw1000/2026-11-28" /
               "specialized-llava-clevr-val16-5203-native-base-count-cause-and-clean-post.json").read_bytes()
    assert sha(archive) == "3d18fb802f27471014c03406e8688d49c698db02ef386d4ce3637b97736da79a"
    evidence = json.loads(archive)
    assert evidence["actual_base_tensor_counts"]["parameter_tensors"] == 698
    assert evidence["actual_base_tensor_counts"]["lora_parameter_tensors"] == 0
    assert evidence["actual_accounting"]["gpu"][0] == "1103606.5203"
    assert evidence["actual_accounting"]["post"][0] == "1103606.5204" and evidence["final_cleanup_verified"]
    assert generation([4, 7, 9, 0], [9], 0, "stop", [4, 7, 9]) == "eos"
    assert generation([4, 7], [9], 0, "stop", [4, 7]) == "stop"
    assert generation([4] * CAP, [9], 0, "length", [4] * CAP) == "cap"
    rejects(lambda: generation([4, 9, 7], [9], 0, "stop", [4, 9, 7]), "NATIVE_EOS_BOUNDARY")
    rejects(lambda: generation([4, 7], [9], 0, "length", [4, 7]), "NATIVE_STOP_OR_CAP_UNPROVEN")
    rejects(lambda: generation([4, 9], [9], 0, "stop", [7, 9]), "DECODED_TOKEN_IDS_DIFFER")
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        exercise_base_gate(root)
        input_root = root / "input"
        image_root = input_root / "images"
        image_root.mkdir(parents=True)
        roster, prompts = [], []
        for slot in SLOTS:
            path = image_root / f"{slot:02d}.png"
            Image.new("RGB", (480, 320), (slot, 10, 5)).save(path)
            blob = path.read_bytes()
            roster.append(dict(slot=slot, name=path.name, bytes=len(blob), SHA256=sha(blob)))
            prompts.append(dict(slot=slot, images=[f"images/{path.name}"], messages=[dict(role="user", content=PROMPT)]))
        route, pins = input_root / "requests.jsonl", input_root / "IMAGES.json"
        requests_blob = b"".join(map(wire, prompts))
        put(route, requests_blob)
        put(pins, wire(roster))
        assert inputs(input_root, sha(requests_blob), sha(wire(roster))) == prompts
        rejects(lambda: inputs(input_root, "0" * 64, sha(wire(roster))), "MEMBER_HASH_DIFFERS")
        unknown = input_root / "answers.json"
        unknown.touch()
        rejects(lambda: inputs(input_root, sha(requests_blob), sha(wire(roster))), "EXACT_ANSWERLESS_CLOSURE")
        unknown.unlink()
        prompts[0]["messages"].append(dict(role="assistant", content="test"))
        route.write_bytes(b"".join(map(wire, prompts)))
        rejects(lambda: inputs(input_root, sha(route.read_bytes()), sha(wire(roster))), "NATIVE_CLEVR_PROMPT_OR_SLOT")
        route.write_bytes(requests_blob)
        roster[1]["SHA256"] = roster[0]["SHA256"]
        pins.write_bytes(wire(roster))
        rejects(lambda: inputs(input_root, sha(requests_blob), sha(wire(roster))), "IMAGE_PIN_SLOT_OR_DUPLICATE")

        output = root / "partial"
        output.mkdir()
        record(output, dict(arm="base", slot=0, outcome="eos", text="fixture-only"))
        record(output, dict(arm="base", slot=1, outcome="error", text=None))
        report = seal(output, "fixture", dict(model_loads=0, actual_generate_calls=0), {"phase": "fixture"})
        raw = (output / "predictions.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert [(row["arm"], row["slot"]) for row in rows] == [(arm, slot) for arm in ARMS for slot in SLOTS]
        assert report["predictions_SHA256"] == sha(raw) and len(rows) == report["planned"] == 32
        assert report["counts"]["base"]["missing"] == 14 and report["counts"]["cp25"]["missing"] == 16
        assert report["counts"]["base"]["error"] == 1 and report["passed"] is False
    source = Path(__file__).with_name("native.py").read_bytes()
    tree = ast.parse(source)
    compile(source, "exact-candidate-native.py", "exec")
    exercise_paired_cache_loop(tree)
    repo = Path(__file__).resolve().parents[3]
    original = repo / "code/bw1000/training-llava-onevision-cache-position-two-forward-CPU-v1/test_local.py"
    sys.path.insert(0, str(original.parent))
    try:
        runpy.run_path(str(original))["main"]()
    finally:
        sys.path.pop(0)
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "get_model_processor" for node in ast.walk(tree)) == 1
    print("exact .5203/.5204 evidence, 698/0 LoRA and finalizer fixtures, 32 sealed slots PASS, 0 real model or GPU calls")


if __name__ == "__main__":
    main()
