"""Bind the C4 diagnostic entry and terminal checks to the bounded r467 controller."""
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
BASELINE = "a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1"
NATIVE = "1b729a2d29d39f084addc4978fbe3db7bd4b685360c384f7dfe2ae8908fc9e7f"
CPU_RECEIPT = "775a26846037e5925cf3e3343569c8c39140cd78d65fbee44bf8c72118d9ea28"
OLD = "Deep-C1-gap-enclosure-r467"
NEW = "Deep-C4-current-host-split"


def change(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"Missing or ambiguous deployment anchor: {before}")
    return text.replace(before, after)


def derive(parent, entry, local):
    files = {}
    for name, pin in PINS.items():
        raw = (parent / name).read_bytes()
        if sha256(raw).hexdigest() != pin:
            raise ValueError(f"Original deployment differs: {name}")
        files[name] = raw.decode()
    manifest = (entry / "SOURCE_MANIFEST.json").read_bytes()
    entry_pin = sha256(manifest).hexdigest()
    data = json.loads(manifest)
    assert not any(data[key] for key in ("CPU_GO", "GPU_GO", "Model_GO"))
    for name, pin in data["files_SHA256"].items():
        assert sha256((entry / name).read_bytes()).hexdigest() == pin
    files["config.json"] = (entry / "config.template.json").read_bytes().decode()
    for name in ("terminal_host_split.py", "consume.py", "check_deployment_CPU.py"):
        files[name] = (local / name).read_bytes().decode()

    support = files["support.py"]
    for before, after, count in (
        ("RUN='current-a314-profile-a';LABEL='Deep-r467-current-a314-profile'",
         "RUN='c4-current-host-split-a';LABEL='Deep-C4-current-host-split'", 1),
        ("27892f2a9c62bada471942fcdcd53b0be7e3843f53effaa434be165add1855e8", entry_pin, 1),
        ("R/'results/Deep-r467-current-profile'/RUN", "R/'results/Deep-C4-current-host-split/current-v1'", 1),
        ("'/refactor/results/Deep-r467-current-profile/'+RUN", "'/refactor/results/Deep-C4-current-host-split/current-v1'", 1),
        (BASELINE, NATIVE, 2),
        ("c['samples']==[8,87]", "c['samples']==[11,12]", 1),
        ("assert q['actual_ELF_CPU750_50objects_SDK_profile_sources_verified']and q['actual_changed_entry_consumer']['passed']and q['actual_changed_entry_consumer']['actual_entry_main_execute']",
         f"assert q['entry_nonserve_executed'] and q['Native_CPU_receipt_SHA256']=='{CPU_RECEIPT}' and q['deployment_manifest_SHA256']==sha(d/'SOURCE_MANIFEST.json')", 1),
    ):
        support = change(support, before, after, count)
    files["support.py"] = support
    files["launch_once.py"] = change(files["launch_once.py"], "samples=[8,87]", "samples=[11,12]")
    files["controller.py"] = change(files["controller.py"], OLD + "/deployment-v1/inside.sh", NEW + "/deployment-v1/inside.sh")
    files["inside.sh"] = change(files["inside.sh"], OLD, NEW, 2)
    files["idle.py"] = change(files["idle.py"], f"b'/{OLD}/'in a", f"b'/{OLD}/'in a or b'/{NEW}/'in a")

    collect = change(files["collect.py"], "from support import sha,NODE,actual", "from support import sha,NODE,actual\nfrom terminal_host_split import audit_window")
    collect = change(collect, "count=0;scalars=[]", "count=0;scalars=[];host_splits=[]")
    collect = change(collect, "  for w in r['profile_windows']:", "  for w in r['profile_windows']:\n   assert r['index']==3\n   host_splits.append(audit_window(output,w))")
    collect = change(collect, " for i in [8,87]:", " for i in [11,12]:")
    collect = change(collect, " return dict(entry_report_SHA256", " expected=[i for i in (11,12) if len(rs[3]['generated_token_ids'])>i]\n assert [w['sample'] for w in host_splits]==expected\n return dict(C4_host_splits=host_splits,entry_report_SHA256")
    files["collect.py"] = collect
    files["ENTRY_CPU_BINDING.json"] = '{"SHA256": null, "actual_CPU": null, "passed": false}\n'
    for name, text in files.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    pins = {name: sha256(text.encode()).hexdigest() for name, text in files.items()}
    files["SOURCE_MANIFEST.json"] = json.dumps(dict(files_SHA256=pins,
        entry_source_manifest_SHA256=entry_pin, CPU_GO=False, GPU_GO=False, Model_GO=False,
        execution_READY=False, inherited_GPU_seconds=2700, inherited_keeper_seconds=3000), indent=2) + "\n"
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "entry", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    files = derive(args.parent, args.entry, Path(__file__).resolve().parent)
    args.output.mkdir()
    for name, text in files.items():
        with (args.output / name).open("xb") as stream:
            stream.write(text.encode())
    print(sha256(files["SOURCE_MANIFEST.json"].encode()).hexdigest())


if __name__ == "__main__":
    main()
