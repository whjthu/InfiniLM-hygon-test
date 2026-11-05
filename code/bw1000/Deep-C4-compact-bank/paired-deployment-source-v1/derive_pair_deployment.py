"""Derive a CPU-gated, source-only deployment for the compact C4 paired entry."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

PINS = {
    "collect.py": "47adbbed309fd1e9124cf07f8009bfa8171d1e824c0ee4bce6e597f4b5ffb395",
    "controller.py": "81e670a110e9a8c2fe387cf99592c49a949c944235f65c7f3e8f05aef117f979",
    "idle.py": "e6efb5a8fdb0087c0023c1400ad85f8968f9c490cf11aa55e23f2d7e0acd1377",
    "inside.sh": "d64902041afffca353a31ce86312ca7cf33d6936abe3a21992ba9137418109b2",
    "launch_once.py": "ad3edd454d23e932e8e43efa0a54f75dedd9f9109276ef6053b69c1615f414a9",
    "owned_cleanup.py": "60c38ab7c09b807cf7704c113a51ff3169b811604653b02f29b9cb11db45d032",
    "resource_wave_host_alias_outer.sh": "b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d",
    "support.py": "967ef6b365c6e8a6d68dd05b4e70084ef7399bb4fc96d706c4ed6ec6aab4b377",
    "worker.py": "7564d8fa62eb5d231cdd96d02df927080b97c48e7b19b8e2d112caeaba5f84ba",
}
ENTRY = "f4cf34346fb1d44f3d01ccc8380fef6b83b43cf1b28e882e9c70be0e8515da68"
NATIVE = "7fcea960c5f218234ed8a4733fa7bd94b31055ff7396755c2afd6f5bba502f90"
CPU_RECEIPT = "710e6046deb71019f7e5e015206d08c630100b9999f9cc884307903e50e9bf38"
OLD = "Deep-C1-gap-enclosure-r467"
NEW = "Deep-C4-compact-bank-pair"


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def change(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError("missing or ambiguous deployment source anchor")
    return source.replace(before, after)


def derive(parent, entry, local):
    files = {}
    for name, expected in PINS.items():
        raw = (parent / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"original deployment differs: {name}")
        files[name] = raw.decode()
    raw = (entry / "SOURCE_MANIFEST.json").read_bytes()
    assert sha256(raw).hexdigest() == ENTRY
    manifest = json.loads(raw)
    assert not any(manifest[k] for k in ("CPU_GO", "GPU_GO", "Model_GO"))
    for name, expected in manifest["files_SHA256"].items():
        assert sha256((entry / name).read_bytes()).hexdigest() == expected
    files["config.json"] = (entry / "config.template.json").read_bytes().decode()
    support = files["support.py"]
    for before, after, count in (
        ("RUN='current-a314-profile-a';LABEL='Deep-r467-current-a314-profile'",
         "RUN='c4-compact-bank-pair-a';LABEL='Deep-C4-compact-bank-pair'", 1),
        ("27892f2a9c62bada471942fcdcd53b0be7e3843f53effaa434be165add1855e8", ENTRY, 1),
        ("R/'results/Deep-r467-current-profile'/RUN", "R/'results/Deep-C4-compact-bank-pair/current-v1'", 1),
        ("'/refactor/results/Deep-r467-current-profile/'+RUN", "'/refactor/results/Deep-C4-compact-bank-pair/current-v1'", 1),
        ("a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1", NATIVE, 2),
        ("c['requests']==5 and c['observer_schedule']==[False,False,False,True,False]and c['samples']==[8,87]",
         "c['requests']==6 and c['compact_schedule']==[False,True,False,True,True,False]", 1),
        ("assert q['actual_ELF_CPU750_50objects_SDK_profile_sources_verified']and q['actual_changed_entry_consumer']['passed']and q['actual_changed_entry_consumer']['actual_entry_main_execute']",
         f"assert q['entry_nonserve_executed']and q['Native_CPU_receipt_SHA256']=='{CPU_RECEIPT}'and q['deployment_manifest_SHA256']==sha(d/'SOURCE_MANIFEST.json')", 1),
    ):
        support = change(support, before, after, count)
    files["support.py"] = support
    files["controller.py"] = change(files["controller.py"], OLD + "/deployment-v1/inside.sh", NEW + "/deployment-v1/inside.sh")
    files["controller.py"] = change(files["controller.py"], "planned_requests=5", "planned_requests=6")
    files["controller.py"] = change(files["controller.py"], "FIVE_REQUESTS_NORMAL_PARTIAL_EVENT_PROFILE_RECORDED", "SIX_REQUESTS_NORMAL_COMPACT_BANK_PAIRED")
    files["launch_once.py"] = change(files["launch_once.py"],
        "requests=5,observer_schedule=[False,False,False,True,False],samples=[8,87]",
        "requests=6,compact_schedule=[False,True,False,True,True,False]")
    files["inside.sh"] = change(files["inside.sh"], OLD, NEW, 2)
    files["idle.py"] = change(files["idle.py"], f"b'/{OLD}/'in a", f"b'/{OLD}/'in a or b'/{NEW}/'in a")
    files["worker.py"] = change(files["worker.py"], "profile_coverage_complete=q['profile_coverage_complete']",
                               "same_state_pair_records=len(q['paired_decode'])")
    files["collect.py"] = (local / "collect_pair.py").read_bytes().decode()
    files["ENTRY_CPU_BINDING.json"] = '{"SHA256": null, "actual_CPU": null, "passed": false}\n'
    for name, text in files.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    data = {n: s.encode() for n, s in files.items()}
    data["SOURCE_MANIFEST.json"] = encode(dict(files_SHA256={n:sha256(b).hexdigest() for n,b in data.items()},
        entry_source_manifest_SHA256=ENTRY, CPU_GO=False, GPU_GO=False, Model_GO=False,
        execution_READY=False, inherited_GPU_seconds=2700, inherited_keeper_seconds=3000))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "entry", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    data = derive(a.parent, a.entry, Path(__file__).resolve().parent)
    a.output.mkdir()
    for name, raw in data.items():
        with (a.output / name).open("xb") as f:
            f.write(raw)


if __name__ == "__main__":
    main()
