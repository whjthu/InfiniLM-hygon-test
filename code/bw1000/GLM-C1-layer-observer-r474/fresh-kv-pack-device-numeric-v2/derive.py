"""Render the single-scope GLM pack repair from exact reviewed v1 sources."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AREA = Path("code/bw1000/GLM-C1-layer-observer-r474")
OLD = AREA / "fresh-kv-pack-device-numeric-v1"
OLD_HOST = AREA / "fresh-kv-pack-device-host-v1" / "host_once.py"
NEW = AREA / "fresh-kv-pack-device-numeric-v2"
READER = AREA / "fresh-kv-pack-device-host-result-v2" / "inspect_result.py"
OLD_LABEL = "GLM474-fresh-kv-pack-numeric-GPU-v1"
LABEL = "GLM474-fresh-kv-pack-numeric-GPU-v2"
OLD_REQUEST = "5667064d92534d079d0edb1956750456"
REQUEST = "93148263aba346dda17b534d419d81c83"
HIP_CANONICAL = "/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26113.2065-6699a1f0"
HIP_BYTES = 12_349_016
METADATA_RECEIPT = "438db278e875376e11891787b3975bb248807152d61eb3d2ffec6732cb0ef46c"
PINS = {
    OLD / "probe.py": "0c3ad2473b1ee2319fa2966526fe3f5f77ad94dc2b1bed0050be469057b6496c",
    OLD / "device_inside.sh": "1358105d177ee661a24d3a9af40f922ec1a977d54219b43648a4fcaca9a497ec",
    OLD / "CLAIM.json": "6e4f4cf88810fa0016f82008dbb22bf0b036210e836849481001ec7ab06a0af2",
    OLD_HOST: "f5606c498fd72c6b2efe616b0dad679365289cb9bd2e8c451026679473b967cf",
    READER: "266449c1cabe928a33d4d6e2f46421e0c7e9969a49fd3efe33a3ad4dde2ced5c",
}


def digest(raw):
    return sha256(raw).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen replacement context differs: " + old)
    return text.replace(old, new, 1)


def scope(text):
    return once(once(text, OLD_LABEL, LABEL), OLD_REQUEST, REQUEST)


def render_probe(text):
    text = once(scope(text),
                'HIP_NAME = "libgalaxyhip.so.5.2.26113.2065-6699a1f0"',
                f'HIP_CANONICAL = Path("{HIP_CANONICAL}")\nHIP_BYTES = {HIP_BYTES}')
    old = '''def bindings(scope, expected, admitted_pci):
    native_raw = read_pin(NATIVE, NATIVE_SHA, NATIVE_BYTES)
    need(len(native_raw) == NATIVE_BYTES and NATIVE.resolve(strict=True) == NATIVE, "wrong .918 DSO identity")
    hip_path = Path(HIP_LINK).resolve(strict=True)
    need(hip_path == Path(HIP_LINK).with_name(HIP_NAME) and expected.get(HIP_LINK) == HIP_SHA,
         "unexpected canonical HIP provider")
    check = verify_providers(scope, expected)'''
    new = '''def hip_identity(expected):
    need(expected.get(HIP_LINK) == HIP_SHA, "unexpected original HIP provider SHA pin")
    hip_path = Path(HIP_LINK).resolve(strict=True)
    need(hip_path == HIP_CANONICAL, "unexpected canonical HIP provider path")
    need(hip_path.is_file() and hip_path.stat().st_size == HIP_BYTES, "unexpected canonical HIP provider size")
    return hip_path


def bindings(scope, expected, admitted_pci):
    native_raw = read_pin(NATIVE, NATIVE_SHA, NATIVE_BYTES)
    need(len(native_raw) == NATIVE_BYTES and NATIVE.resolve(strict=True) == NATIVE, "wrong .918 DSO identity")
    hip_path = hip_identity(expected)
    check = verify_providers(scope, expected)
    check.update(canonical_HIP=str(hip_path), canonical_HIP_bytes=HIP_BYTES)'''
    return once(text, old, new)


def render_host(text, files):
    text = once(once(scope(text), 'DEVICE = AREA / "fresh-kv-pack-device-numeric-v1"',
                     'DEVICE = AREA / "fresh-kv-pack-device-numeric-v2"'),
                'THIS = AREA / "fresh-kv-pack-device-host-v1/host_once.py"',
                'THIS = DEVICE / "host_once.py"')
    text = once(text, "from types import ModuleType", "from types import ModuleType, SimpleNamespace")
    text = once(text, "WAITER = SNAPSHOT", f'RESULT_READER = Path("{READER.as_posix()}")\nWAITER = SNAPSHOT')
    text = once(text, "PINS = {", f'PINS = {{\n    RESULT_READER: "{PINS[READER]}",')
    for filename, raw in files.items():
        text = once(text, PINS[OLD / filename], digest(raw))
    text = once(text, 'and physical["pci"] in before["pci"]',
                '''and bound["providers"]["canonical_HIP"] == str(probe.HIP_CANONICAL)
             and bound["providers"]["canonical_HIP_bytes"] == probe.HIP_BYTES
             and physical["pci"] in before["pci"]''')
    return once(text, 'observed = inspect_probe(run / "probe.jsonl", gpu["row"], intent_sha, before, probe)',
                '''reader = load(RESULT_READER, pin(checkout, RESULT_READER, PINS[RESULT_READER]))
        host = SimpleNamespace(inspect_probe=inspect_probe, bounded=bounded, own_identity=own_identity, need=need)
        observed = reader.inspect_probe(run / "probe.jsonl", gpu["row"], intent_sha, before, probe, host)''')


def render(repo):
    prior = {}
    for path, expected in PINS.items():
        raw = (repo / path).read_bytes()
        if digest(raw) != expected:
            raise ValueError("reviewed input changed: " + path.as_posix())
        prior[path] = raw
    files = {
        "probe.py": render_probe(prior[OLD / "probe.py"].decode()).encode(),
        "device_inside.sh": once(prior[OLD / "device_inside.sh"].decode(), OLD.name, NEW.name).encode(),
    }
    claim = json.loads(prior[OLD / "CLAIM.json"])
    claim.update(label=LABEL, request=REQUEST, new_files_SHA256={name: digest(raw) for name, raw in files.items()})
    claim["predecessor"] = {
        "request": OLD_REQUEST,
        "gpu_step": "1103606.932",
        "terminal": "FAILED 1:0",
        "passed_valid": 0,
        "planned_valid": 4,
        "passed_rejected": 0,
        "planned_rejected": 36,
        "failure_before_active_load": "unexpected canonical HIP provider",
        "replay_allowed": False,
    }
    claim["provider_path_repair"] = {
        "actual_zero_GRES_metadata_step": "1103606.935",
        "actual_metadata_receipt_SHA256": METADATA_RECEIPT,
        "alias_path": "/opt/dtk/lib/libgalaxyhip.so.5",
        "canonical_path": HIP_CANONICAL,
        "canonical_bytes": HIP_BYTES,
        "canonical_SHA256": claim["native"]["HIP_SHA256"],
        "pre_BOUND_failure_reader": READER.as_posix(),
        "pre_BOUND_failure_reader_SHA256": PINS[READER],
    }
    files["CLAIM.json"] = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode()
    files["host_once.py"] = render_host(prior[OLD_HOST].decode(), files).encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()
    files = render(REPO)
    for name, raw in files.items():
        path = HERE / name
        if args.write_generated:
            with path.open("xb") as output:
                output.write(raw)
        elif path.read_bytes() != raw:
            raise ValueError("staged byte divergence: " + name)
        print(digest(raw), name)


if __name__ == "__main__":
    main()
