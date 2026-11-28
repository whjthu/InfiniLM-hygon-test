"""Offline checks for four-row membership, native sample ledger and CP20 cursor."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import train4
from run import compare
from derive import helper_source


def reject(fn, expected):
    try:
        fn()
    except ValueError as error:
        assert str(error) == expected, str(error)
    else:
        raise AssertionError(expected)


def data_fixture(root, evidence):
    images = root / "imagefolder"
    images.mkdir()
    report = deepcopy(evidence)
    report["render"]["files"] = {}
    metadata = []
    for index, name in enumerate(train4.NAMES):
        image = Image.new("RGB", (256, 256), (index, 20, 40))
        image.save(images / name)
        caption = f"fixture café {index // 2}"
        metadata.append(dict(file_name=name, text=caption))
        report["target"]["rows"][index].update(
            caption_SHA256=sha256(caption.encode()).hexdigest(),
            decoded_RGB_SHA256=sha256(image.tobytes()).hexdigest(),
            original_file_SHA256=sha256((images / name).read_bytes()).hexdigest(),
        )
    (images / "metadata.jsonl").write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in metadata))
    for path in images.iterdir():
        raw = path.read_bytes()
        report["render"]["files"][path.name] = dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())
    proof = root / "fixture.json"
    proof.write_bytes(json.dumps(report).encode())
    with patch.object(train4, "ARCHIVE", sha256(proof.read_bytes()).hexdigest()):
        data = train4.bind_data(images, proof, decode=True)
        assert data["records"][0]["caption_sha256"] != report["target"]["rows"][0]["caption_SHA256"]
        assert data["records"][0]["caption_sha256"] == train4.json_sha(metadata[0]["text"])
        assert data["ordered_sha256"] == train4.json_sha(data["records"])
        extra = images / "holdout.png"
        extra.touch()
        reject(lambda: train4.verify(images, proof), "EXACT_FIVE_IMAGEFOLDER_MEMBERS")
        extra.unlink()
        with (images / "metadata.jsonl").open("ab") as stream:
            stream.write(b" ")
        reject(lambda: train4.verify(images, proof), "EXACT_2766_MEMBER_BYTES")
    return data


def ledger_fixture(records):
    def encode(rows):
        return b"".join(json.dumps(row).encode() + b"\n" for row in rows)

    rows = []
    for step in range(1, 26):
        epoch, batch = divmod(step - 1, 4)
        sample = (batch + epoch) % 4
        rows.append(dict(global_step=step, loss=0.1, grad_norm=1.0,
                         sample_witness=dict(epoch=epoch, batch_index=batch, sample_ids=[sample],
                                             samples=[records[sample]], tensors=dict.fromkeys(
                            ("pixel_values", "input_ids_one", "input_ids_two"), "a" * 64))))
    assert [(rows[step-1]["sample_witness"]["epoch"], rows[step-1]["sample_witness"]["batch_index"])
            for step in (19, 20, 21, 24, 25)] == [(4, 2), (4, 3), (5, 0), (5, 3), (6, 0)]
    assert train4.check_updates(encode(rows), 1, 25, records) == rows
    assert train4.check_updates(encode(rows[20:]), 21, 25, records) == rows[20:]
    bad = deepcopy(rows)
    bad[20]["sample_witness"].update(sample_ids=bad[21]["sample_witness"]["sample_ids"],
                                     samples=bad[21]["sample_witness"]["samples"])
    reject(lambda: train4.check_updates(encode(bad[20:]), 21, 25, records), "DUPLICATE_SAMPLE_IN_NATIVE_EPOCH")
    bad = deepcopy(rows)
    bad[19]["sample_witness"]["batch_index"] = 1
    reject(lambda: train4.check_updates(encode(bad), 1, 25, records), "NATIVE_UPDATE_OR_CURSOR_NOT_PROVEN")
    bad = deepcopy(rows)
    bad[0]["sample_witness"]["samples"][0]["caption_sha256"] = "e" * 64
    reject(lambda: train4.check_updates(encode(bad), 1, 25, records), "EXACT_NATIVE_SAMPLE_RECORD")
    reject(lambda: train4.check_updates(encode(rows[20:-1]), 21, 25, records), "COMPLETE_ACTUAL_UPDATE_LEDGER")
    witness = dict(completed_step=0, process_pid=10, process_startticks=100)
    restored = dict(completed_step=20, next_step=21, state_sha256="a" * 64,
                    restored=dict(state_sha256="a" * 64), restored_full_state_exact=True,
                    process_pid=20, process_startticks=200)
    source = dict(state_sha256="a" * 64)
    final = dict(state_sha256="b" * 64)
    assert compare(rows, rows[20:], source, final, final, witness, restored, 20)["passed"]
    drift = deepcopy(rows[20:])
    drift[1]["loss"] = 0.2
    assert compare(rows, drift, source, final, final, witness, restored, 20)["first_different_update"] == 22


def cursor_fixture(repo, root, data):
    package = repo / "code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz"
    with tarfile.open(package) as archive:
        src = archive.extractfile("framework/generation/src/genmlu/sdxl_training_resume_checkpoint.py").read()
    derived = helper_source(src)
    assert sha256(derived).hexdigest() == "7105c88a0f9a292396c37acec217e66e8cc60f7142903251a8be219544e223f0"
    namespace = {"__file__": "frozen_helper.py", "__name__": "fixture"}
    exec(compile(derived, "frozen_helper.py", "exec"), namespace)
    cursor = dict(epoch=4, consumed=4, order=[1, 3, 2, 0], generator="fixture")
    state = dict(data=cursor)
    contract = dict(data=data)
    payload = dict(step=20, state=state, contract=contract)
    document = dict(state_sha256=train4.json_sha(state), contract=contract)
    cp20 = root / "continuous/checkpoint-20"
    cp20.mkdir(parents=True)
    manifest_file = cp20 / namespace["MANIFEST"]
    manifest_file.write_bytes(b"fixture native checkpoint manifest")
    def read_state(path, *, numpy_rng):
        assert path == cp20 / namespace["STATE"] and numpy_rng
        return payload
    helper = SimpleNamespace(STATE=namespace["STATE"], MANIFEST=namespace["MANIFEST"],
                             validate_cursor=namespace["validate_cursor"], restricted_load=read_state,
                             state_sha=train4.json_sha, file_sha=namespace["file_sha"],
                             preflight=lambda path: document)
    result = train4.cp20_projection(helper, cp20, document)
    assert (result["step"], result["epoch"], result["consumed"], result["order"]) == (20, 4, 4, [1, 3, 2, 0])
    cursor["consumed"] = 3
    document["state_sha256"] = train4.json_sha(state)
    reject(lambda: train4.cp20_projection(helper, cp20, document), "EXACT_NEW_CP20_EPOCH4_CONSUMED4")
    cursor["consumed"] = 4
    payload["step"] = 10
    document["state_sha256"] = train4.json_sha(state)
    reject(lambda: train4.cp20_projection(helper, cp20, document), "SAME_NEW_CP20_MANIFEST_AND_STATE")


def main():
    repo = Path(__file__).resolve().parents[3]
    evidence = repo / "results/bw1000/2026-11-28/specialized-sdxl-2766-train-only-two-families-four-native-imagefolder-rows.json"
    _, report = train4.archive(evidence)
    original = repo / "code/bw1000/training-sdxl-cc0-native-full-resume-hip-v1/test_local.py"
    pins = runpy.run_path(str(original))["exercise_sources"](repo)
    with TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        data = data_fixture(root, report)
        ledger_fixture(data["records"])
        cursor_fixture(repo, root, data)
    print(json.dumps(dict(archive_2766=train4.ARCHIVE, exact_derived_provider_SHA256=pins,
                          offline_four_row_negative_tests="PASS", actual_GPU_calls=0), sort_keys=True))


if __name__ == "__main__":
    main()
