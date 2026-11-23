"""Create the two-process registered COCO20 engineering execution package."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "training-zimage-registered-hip-scheduler-guard-v2"
LABEL = "training-zimage-coco20-registered-hip-recovery-v1"
REQUEST = "fc8bffa014714c9bbf26dbf64b48693"
TARGET = "results/training-common-zimage-native-recovery-v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False) + "\n").encode()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Verified envelope source changed")
    return text.replace(old,new,1)


def build(verified, destination):
    old = json.loads((PREVIOUS / "GPU_SCOPE.json").read_bytes())
    manifest = (verified / "SOURCE_FILES.json").read_bytes()
    entry = (verified / "entry.py").read_bytes()
    if sha(manifest) != old["manifest_sha256"] or sha(entry) != old["entry_sha256"]:
        raise ValueError("Wrong fixed-first execution envelope")
    files = {}
    for name,pin in json.loads(manifest)["files"].items():
        raw = (verified / name).read_bytes()
        if (len(raw),sha(raw)) != (pin["bytes"],pin["SHA256"]):
            raise ValueError("Changed fixed-first member " + name)
        files[name] = raw
    claim = json.loads(files["CLAIM.json"])
    proof = json.loads((HERE / "SOURCE_SCOPE.json").read_bytes())["source_proof"]
    claim.update(request=REQUEST,label=LABEL,whole_seconds=1080,worker_seconds=970,
                 standby_handoff={"step":"1103606.1755","name":"training80-dcu-standby-8740dfa0d280-f13r2n07",
                                  "node":claim["node"],"dcu":8},
                 planned_updates={"continuous":list(range(1,26)),"resume20":list(range(21,26))},
                 common_candidate={"candidate_commit":proof["candidate_commit"],"git_tree":proof["git_tree"],
                                   "manifest_sha256":proof["artifacts"]["candidate-manifest.json"]["SHA256"]})
    claim["registered_binding"]["cli_SHA256"] = proof["changed_members"]["tools/hygon_training_recipe.py"]["SHA256"]
    claim["framework_update_members"]["zimage_native_record.py"] = dict(
        proof["changed_members"]["generation/src/gencuda/zimage_native_record.py"],source="verified_integrated_candidate")
    files["CLAIM.json"] = encode(claim)
    files["worker.py"] = (HERE/"recovery_worker.py").read_bytes()
    old_target = "results/training-common-zimage-scheduler-guard-v2"
    controller = files["controller.py"].decode()
    controller = replace(controller, f'(BASE/"{old_target}/source-tree", "/common-framework", "ro"),',
                         f'(BASE/"{TARGET}/source-tree", "/common-framework", "ro"),')
    controller = replace(controller, '        with path.open("xb") as log:',
                         '        if gpu:\n            result.setdefault("attempted_gpu_phases",[]).append(phase)\n'
                         '        with path.open("xb") as log:')
    controller = replace(controller, '''        _, success = launch("device", entries, gpu=True)
        result["device_passed"] = success
        if (ROOT / "worker-result.json").is_file():
            result["worker_result_SHA256"] = sha256((ROOT / "worker-result.json").read_bytes()).hexdigest()
        result["optimizer_step_returned"] = (ROOT / "optimizer-step-returned.private.json").is_file()
        result["optimizer_step_intent"] = (ROOT / "optimizer-step-intent.private.json").is_file()''',
                         '''        _, continuous = launch("continuous", entries[:-1] + [env+" continuous"], gpu=True)
        result["continuous_passed"] = continuous
        need(continuous, "Registered continuous25 did not finish")
        middle = snapshot("between")
        need(pre["PCIs"]==middle["PCIs"] and pre["boot_id_SHA256"]==middle["boot_id_SHA256"],
             "Device topology changed before fresh-process recovery")
        _, restarted = launch("resume20", entries[:-1] + [env+" resume20"], gpu=True)
        result["resume20_passed"] = restarted
        result["device_passed"] = continuous and restarted
        for mode in ("continuous","resume20"):
            path = ROOT / (mode + "-worker-result.json")
            if path.is_file():
                result[mode+"_result_SHA256"] = sha256(path.read_bytes()).hexdigest()''')
    controller = replace(controller, '                need("device" in phases, "Actual terminal device accounting required before post")',
                         '                need(all(p in phases for p in result.get("attempted_gpu_phases",[])),\n'
                         '                     "Actual terminal accounting required before post")')
    files["controller.py"] = controller.encode()
    host = entry.decode()
    host = replace(host, old["label"], LABEL)
    host = replace(host, old["request"], REQUEST)
    host = replace(host, f'    common_root = R / "{old_target}"',
                   f'    common_root = R / "{TARGET}"\n    scheduler_root = R / "{old_target}"')
    host = replace(host, '    scheduler_raw = (common_root/"cpu-outcome.json").read_bytes()',
                   '    scheduler_raw = (scheduler_root/"cpu-outcome.json").read_bytes()')
    host = replace(host, 'sha(stage_raw)==claim["scheduler_CPU"]["stage_SHA256"]',
                   'receipt["registered_engineering_help_verified"] is True')
    host = replace(host, '        "--time=00:11:00","--kill-on-bad-exit=1",',
                   '        "--time=00:20:00","--kill-on-bad-exit=1",')
    files["SOURCE_FILES.json"] = encode({"files":{name:{"bytes":len(raw),"SHA256":sha(raw)} for name,raw in files.items()}})
    files["entry.py"] = host.encode()
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    destination.mkdir()
    for name,raw in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(raw)
    print(json.dumps(dict(label=LABEL,request=REQUEST,manifest_sha256=sha(files["SOURCE_FILES.json"]),
                          claim_sha256=sha(files["CLAIM.json"]),entry_sha256=sha(files["entry.py"]),
                          bundle_files=len(files)-1),sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified",type=Path)
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    build(args.verified,args.destination)
