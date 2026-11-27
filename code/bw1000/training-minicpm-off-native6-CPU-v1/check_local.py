"""Exercise the actual builder and train readback without invoking a provider or Slurm."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile

import independent
import off_inputs
import prepare

FIXTURE_REQUEST = "95c7f467bed845eea922845a3594d53e"


def refused(call, code):
    try:
        call()
    except ValueError as error:
        assert str(error) == code
    else:
        raise AssertionError(code)


def main():
    origin = Path(sys.argv[1])
    source, train, scope = prepare.render(origin, FIXTURE_REQUEST, "99999999", "f0r0n0")
    assert scope["CPU_GO"] is False and scope["GPU_GO"] is False
    assert (source, train, scope) == prepare.render(origin, FIXTURE_REQUEST, "99999999", "f0r0n0")
    assert prepare.sha(train["INPUT_FILES.json"]) == "9a1338a008b91e2524873e12f49517889a23af4a9869232d5fa1c6103a0e60d4"
    assert len(train) == 6
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp).resolve()
        for name, raw in train.items():
            (folder / name).write_bytes(raw)
        actual = off_inputs.read_inputs(folder, scope["input_manifest_SHA256"])
        assert len(actual) == 6
        assert len({row["images"][0] for row in actual}) == 4
        row = dict(actual[0], images=["off-image-01.jpg"])
        alternate = [{"role": "user", "content": "<image>Question"}, {"role": "assistant", "content": "2 kg"}]
        refused(lambda: off_inputs.check_row(dict(row, messages=alternate), 1), "OFF_ANSWER_UNIT")
        refused(lambda: off_inputs.check_row(dict(row, images=["../extra.jpg"]), 1), "OFF_MEDIA_NAME")
        extra = folder / "dev.jsonl"
        extra.write_bytes(b"x")
        refused(lambda: off_inputs.read_inputs(folder, scope["input_manifest_SHA256"]), "OFF_TRAIN_SCOPE")
        extra.unlink()
        jpg = folder / "off-image-01.jpg"
        raw = jpg.read_bytes()
        jpg.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        refused(lambda: off_inputs.read_inputs(folder, scope["input_manifest_SHA256"]), "OFF_INPUT_MEMBER_SHA")
        refused(lambda: off_inputs.read_inputs(folder, "f" * 64), "OFF_INPUT_MANIFEST_SHA")
    refused(lambda: prepare.render(origin, FIXTURE_REQUEST, "99999999", "f0r0n0", "a" * 40),
            "PUBLICATION_AND_AUTHORIZATION_REQUIRED")
    authority = dict(request=FIXTURE_REQUEST, parent="99999999", node="f0r0n0", publication="a"*40,
                     input_manifest_SHA256=scope["input_manifest_SHA256"], CPU_GO=True, GPU_GO=False,
                     training_GO=False, Model_GO=False, data_on_target_authorized=True)
    adopted, _, bound = prepare.render(origin, FIXTURE_REQUEST, "99999999", "f0r0n0", "a"*40, authority)
    refused(lambda: prepare.render(origin, FIXTURE_REQUEST, "99999999", "f0r0n0", "a"*40,
                                  dict(authority, data_on_target_authorized=False)), "EXACT_ROOT_AUTHORIZATION")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp).resolve()
        for name, raw in adopted.items():
            (folder / name).write_bytes(raw)
        compile(prepare.make_payload(folder, 1795730000), "controller-payload", "exec")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(folder / "independent.py"), "--help"],
                                check=True, capture_output=True, text=True)
        assert "--manifest" in result.stdout
    original = (prepare.ORIGINAL / "worker.py").read_text(encoding="utf8")
    worker = adopted["worker.py"].decode()
    signatures = ("full.encode(row)", "limited.encode(row)", "limited.data_collator([b])",
                  "check_encoded(ids,labels,eos,suffix_ids)", "torch.cuda._lazy_init=no_device")
    for signature in signatures:
        assert original.count(signature) == worker.count(signature) == 1
    assert "training-domain-usgs-table-reading-v1" not in adopted["controller.py"].decode() + worker
    for name, raw in adopted.items():
        if name.endswith(".py"):
            ast.parse(raw, feature_version=(3, 10))
    label = bound["label"]
    record = ["99999999.1", label, "COMPLETED", "0:0", "1", "f0r0n0", "cpu=8,mem=12G,node=1",
              "2026-11-27T00:00:00", "2026-11-27T00:00:01"]
    assert independent.verify_row(record, "99999999", label, "f0r0n0")
    failed = record.copy()
    failed[2:4] = ["FAILED", "1:0"]
    assert not independent.verify_row(failed, "99999999", label, "f0r0n0")
    failed[6] += ",gres/dcu=1"
    refused(lambda: independent.verify_row(failed, "99999999", label, "f0r0n0"), "OFF_TERMINAL_TRES")
    print("PASS exact original 6 rows / 4 media, deterministic rendering, 9 refusals and terminal boundary. Native calls: 0.")


if __name__ == "__main__":
    main()
