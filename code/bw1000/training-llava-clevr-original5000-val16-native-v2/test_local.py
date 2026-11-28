"""Check the real answerless contract and output reader with local fixtures."""

import ast
import json
from contextlib import redirect_stdout
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
    model.members = [(f"hidden_{i}.weight", expected) for i in range(1202)]
    processor = SimpleNamespace(model_meta=SimpleNamespace(template="llava_onevision1_5"))
    good = {}
    native.original_base_gate(model, processor, Model, backend, good)
    assert len(good["base_diagnostic"]["predicates"]) == 8 and all(good["base_diagnostic"]["predicates"].values())
    model.members[0] = ("secret_but_lora.lora_A.default.weight", unexpected)
    model.peft_config = {"secret_peft_key": object()}
    output = root / "diagnostic"
    opts = SimpleNamespace(request="synthetic-seal-fixture", output=output, requests_sha="d" * 64, images_sha="f" * 64)
    def installed_gate(args, owner, facts, current):
        current["phase"] = "load_base"
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
    assert len(diag["predicates"]) == 8 and {k for k, v in diag["predicates"].items() if not v} == {
        "peft_config_absent", "only_floating_bfloat16"}
    assert diag["lora_parameter_tensors"] == 1 and diag["peft_config"] == {"present": True, "type": "dict", "count": 1}
    assert {row["dtype"]: row["tensors"] for row in diag["dtype_device_counts"]} == {bf16: 1201, fp32: 1}
    assert b"secret" not in data and report["facts"]["model_loads"] == 1 and report["facts"]["actual_generate_calls"] == 0
    assert report["planned"] == 32 and all(arm["missing"] == 16 for arm in report["counts"].values())
    assert len((output / "predictions.jsonl").read_bytes().splitlines()) == 32


def main():
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
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "get_model_processor" for node in ast.walk(tree)) == 1
    print("contract fixtures PASS, 16 image rows, 32 sealed prediction rows, 0 model loads, 0 GPU calls")


if __name__ == "__main__":
    main()
