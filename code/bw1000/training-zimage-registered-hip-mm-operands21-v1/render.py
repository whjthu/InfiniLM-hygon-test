"""Rebind the verified native step-21 envelope to the operand observer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCOPE = HERE / "SCOPE.json"
OLD_LABEL = "training-zimage-coco20-registered-hip-profile21-v1"
OLD_REQUEST = "ab198e6bac9e4327836d9031d2b614e6"
OLD_TARGET = "results/training-common-zimage-profile21-v1"
OLD_MANIFEST = "bf050fa578b425a59aee8587f1dd306e5aa4eb5dfd992b329f56e486033b1b42"
OLD_ENTRY = "3b6f24140965daff93992313d770b01aef8c886e110ab6db27197ebf05caf806"


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def replace(raw, anchor, value):
    if raw.count(anchor) != 1:
        raise ValueError("Frozen envelope member changed")
    return raw.replace(anchor, value, 1)


def render(prior, out):
    scope = json.loads(SCOPE.read_bytes())
    framework = scope["framework"]
    index_raw = (prior / "SOURCE_FILES.json").read_bytes()
    entry = (prior / "entry.py").read_bytes()
    if digest(index_raw) != OLD_MANIFEST or digest(entry) != OLD_ENTRY:
        raise ValueError("Wrong previously qualified execution envelope")
    pinned = json.loads(index_raw)["files"]
    files = {name: (prior / name).read_bytes() for name in pinned}
    if any((len(raw), digest(raw)) != (pinned[name]["bytes"], pinned[name]["SHA256"])
           for name, raw in files.items()):
        raise ValueError("Original envelope changed")
    claim = json.loads(files["CLAIM.json"])
    if (claim["label"], claim["request"], claim["expected_checkpoint20_SHA256"]) != (
        OLD_LABEL, OLD_REQUEST, scope["checkpoint20_SHA256"]
    ):
        raise ValueError("Original checkpoint or label differs")
    claim.update(request=scope["request"], label=scope["label"], profile_source_step="1103606.1768",
                 scope="One resumed native training step with a metadata-only Kineto observer.",
                 standby_handoff=scope["standby_handoff"],
                 common_candidate={"candidate_commit": framework["candidate_commit"],
                                   "git_tree": framework["git_tree"],
                                   "manifest_sha256": framework["private_artifacts"]["candidate-manifest.json"]["SHA256"]})
    claim["framework_update_members"].update(
        {Path(name).name: pin for name, pin in framework["changed"].items()}
    )
    claim["framework_update_members"]["zimage_native_record.py"] = scope["inherited_recorder"]
    files["CLAIM.json"] = encode(claim)
    files["controller.py"] = replace(files["controller.py"], OLD_TARGET.encode(), scope["framework_target"].encode())
    before = b"        record.update(passed=True, completed_global_steps=expected_steps, actual_optimizer_updates=2,\n"
    files["worker.py"] = replace(files["worker.py"], before,
        b"        operands = result['profile_summary']['backward_mm_operands']\n"
        b"        binding = output / 'operator-profile-private' / 'backward-mm-operands.json'\n"
        b"        require(operands['file'] == binding.name and binding.stat().st_size == operands['bytes']\n"
        b"                and fingerprint(binding, 1 << 20) == operands['SHA256'], 'Operand receipt differs')\n"
        b"        record['actual_operand_observation'] = operands\n" + before)
    entry = replace(entry, OLD_LABEL.encode(), scope["label"].encode())
    entry = replace(entry, OLD_REQUEST.encode(), scope["request"].encode())
    entry = replace(entry, OLD_TARGET.encode(), scope["framework_target"].encode())
    entry = replace(entry, b'receipt["source_files"]==395', b'receipt["source_files"]==397')
    entry = replace(entry, b'receipt["profile21_option_verified"] is True',
                    b'receipt["profile21_option_verified"] is True and receipt["operand_observer_verified"] is True')
    files["SOURCE_FILES.json"] = encode({"files": {name: {"bytes": len(raw), "SHA256": digest(raw)}
                                                  for name, raw in files.items()}})
    files["entry.py"] = entry
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    out.mkdir()
    for name, raw in files.items():
        with (out / name).open("xb") as target:
            target.write(raw)
    return {"request": scope["request"], "label": scope["label"],
            "claim_SHA256": digest(files["CLAIM.json"]), "entry_SHA256": digest(entry),
            "manifest_SHA256": digest(files["SOURCE_FILES.json"]), "bundle_files": len(files) - 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.prior, args.destination), sort_keys=True))
