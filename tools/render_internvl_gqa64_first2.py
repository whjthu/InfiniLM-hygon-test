"""Bind the existing T03 eight-rank engineering controller to the GQA pilot."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "code/bw1000/training-internvl-usgs72-first2-current-hip-v2"
ROSTER = REPO / "code/bw1000/training-domain-pilot-v1/internvl_gqa_training_roster.py"
REQUEST = "962ea01bdb954b5592d38df1f8e106d9"
LABEL = "training-internvl-gqa64-first2-hip-v1"
CPU_REQUEST = "caac5aebc8e04fef9fd88f12ab69d952"
DATA_NAME = "training-internvl-gqa-native-64-cpu-cfea7ada75b741449ce786c20974b61b"
FINGERPRINT_NAME = "training-internvl-gqa-logit64-cpu-" + CPU_REQUEST
FINGERPRINT_SHA = "3b24a2b1f9f22408cc779129e335197de4eb513561369b297341608675a4e599"
STAGED_TRAIN_SHA = "78de19307549affa877be85756e7077bbb70d8d9feea09e1a8214157ee2fa000"
TRAIN_EXTENT = 19607
PREVIOUS = "39e05f9f8e66485fbeaf08d5a61d6f7e"
FENCE_REQUEST = "6063ab4872014f7dbbee8db7bd97abaa"
SOURCE_SHA = "f82764ec9271fd69fc346147331b4ea30d9515cb821620605bad6e901a61f753"
ROSTER_SHA = "3bd9a38cbd6c1aae99a6365cdb8899bfe1c73e436359ee14da13cc55323163f1"


def digest(value):
    return sha256(value).hexdigest()


def encode(doc):
    return (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def replace(data, old, new):
    if data.count(old) != 1:
        raise ValueError("Frozen T03 source boundary differs: " + old[:50])
    return data.replace(old, new)


def source_module():
    path = ORIGINAL / "render.py"
    raw = path.read_bytes()
    if digest(raw) != SOURCE_SHA:
        raise ValueError("Unexpected predecessor renderer")
    spec = importlib.util.spec_from_file_location("internvl_prior_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render():
    inherited = json.loads((ORIGINAL / "CLAIM.json").read_bytes())
    if inherited["request"] != PREVIOUS or inherited["planned_updates"] != 2:
        raise ValueError("Original first-two training contract differs")
    inherited.update(request=REQUEST, label=LABEL)
    source = source_module()
    files = source.render(inherited)
    roster = ROSTER.read_bytes()
    if digest(roster) != ROSTER_SHA:
        raise ValueError("Fingerprint roster source differs")
    files["internvl_gqa_training_roster.py"] = roster

    scope = files["scope.py"].decode()
    scope = replace(scope, 'DOMAIN_DATA = DOMAIN / "data"',
                    f'DOMAIN_DATA = RESULTS / {DATA_NAME!r} / "data"\n'
                    f'GQA_FINGERPRINT_RECEIPT = RESULTS / {FINGERPRINT_NAME!r} / "result.json"')
    scope = replace(scope, 'DATASET = DOMAIN_DATA / "train-swift.jsonl"', 'DATASET = DOMAIN_DATA / "train.jsonl"')
    files["scope.py"] = scope.encode()

    old_checks = files["domain_checks.py"].decode()
    tail = old_checks[old_checks.index("def before_sample("):]
    files["domain_checks.py"] = (
        '"""Validate the frozen GQA source at the ordinary T03 training boundary."""\n'
        "from scope import DATASET, GQA_FINGERPRINT_RECEIPT, need\n"
        "from internvl_gqa_training_roster import AFTER, BEFORE, load_roster\n\n"
        'PREPARE_SHA = "28fe9cb927390ad423add8b98411a90fb2d1c801ead124b354802bdd0779ecf6"\n\n\n'
        "def verified_domain():\n"
        f"    return load_roster(GQA_FINGERPRINT_RECEIPT, {FINGERPRINT_SHA!r}, DATASET, {TRAIN_EXTENT}, {CPU_REQUEST!r})\n\n\n"
        + tail
    ).encode()
    native = files["native.py"].decode()
    native = replace(native, 'expected="92e3bd193f395f6cd8108f62f86f2e6368619e6210cc3d14997953bd7ff0d538",extent=53958',
                     f'expected="{STAGED_TRAIN_SHA}",extent={TRAIN_EXTENT}')
    files["native.py"] = native.encode()
    callback = replace(files["callback.py"].decode(), "Runtime 72-sample roster differs", "Runtime GQA64 roster differs")
    files["callback.py"] = callback.encode()

    node = files["node.py"].decode()
    start = node.index("def command(")
    node = node[:start] + replace(node[start:], "    binds=[*", "    from scope import GQA_FINGERPRINT_RECEIPT\n\n    binds=[*")
    node = replace(node,
                   """f"{DOMAIN_DATA}:{DOMAIN_DATA}:ro", f"{DOMAIN/'source'}:{DOMAIN/'source'}:ro",
           f"{DOMAIN/'native-caller-rows.json'}:{DOMAIN/'native-caller-rows.json'}:ro",""",
                   """f"{DOMAIN_DATA}:{DOMAIN_DATA}:ro",
           f"{GQA_FINGERPRINT_RECEIPT}:{GQA_FINGERPRINT_RECEIPT}:ro",""")
    files["node.py"] = node.encode()

    inherited.pop("previous")
    inherited.update(schema="INTERNVL_GQA64_FIRST2_HIP_V1", case="T03.domain-gqa64-native-sft",
                     source_cpu_step="1103606.4370", dataset_sha256=STAGED_TRAIN_SHA,
                     terminal_fence_request=FENCE_REQUEST, original_legacy_request=PREVIOUS,
                     fresh_clear_node_and_physical_prepost_required=True,
                     handoff_requires_normal_worker_return_and_physical_cleanup=False)
    files["CLAIM.json"] = encode(inherited)
    files.pop("SOURCE_FILES.json")
    files["SOURCE_FILES.json"] = encode(
        {"request": REQUEST, "files": {key: {"bytes": len(raw), "SHA256": digest(raw)}
                                      for key, raw in files.items()}})
    for key, raw in files.items():
        if key.endswith(".py"):
            compile(raw, key, "exec")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = render()
    if args.output is not None:
        args.output.mkdir()
        for key, raw in files.items():
            with (args.output / key).open("xb") as stream:
                stream.write(raw)
    print(json.dumps({"request": REQUEST, "files": len(files), "manifest_SHA256": digest(files["SOURCE_FILES.json"]),
                      "target_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
