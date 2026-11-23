"""Preserve the real HIP binding and dispatch one registered scheduler-fix run."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / "training-zimage-registered-hip-native-binding-v1"
REPORT = REPO / "results/bw1000/2026-11-23/training-zimage-native-1749-scheduler-preview-regression.json"
LABEL = "training-zimage-coco20-registered-hip-first-v2"
REQUEST = "fc575832065e44a0a20efa3e193ffae7"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2,sort_keys=True,allow_nan=False) + "\n").encode()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Source anchor differs")
    return text.replace(old,new,1)


def main(destination):
    previous = runpy.run_path(str(PREVIOUS / "render_registered_gpu.py"))
    original = json.loads((PREVIOUS / "GPU_SCOPE.json").read_bytes())
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        clone = Path(temporary) / "source"
        previous["main"](clone)
        files = {p.name:p.read_bytes() for p in clone.iterdir()}
    if sha(files["SOURCE_FILES.json"]) != original["manifest_sha256"] or sha(files["entry.py"]) != original["entry_sha256"]:
        raise ValueError("Verified original registered envelope differs")
    data = json.loads(REPORT.read_bytes())
    authority = data["independent_readback"]
    [terminal] = authority["accounting"]
    cpu = data["cpu"]["scheduler_regression"]
    if (terminal[0] != "1103606.1749" or terminal[2:4] != ["COMPLETED","0:0"]
            or "gres/" in terminal[6] or cpu["old_tensor_boolean_error_reproduced"] is not True
            or cpu["fixed_unchanged_scheduler_accepted"] is not True or cpu["changed_timestep_rejected"] is not True):
        raise ValueError("Missing independently verified native scheduler regression")
    scope = json.loads((HERE / "CPU_SCOPE.json").read_bytes())
    proof = scope["source_proof"]
    mounted = scope["target_directory"]
    binding = "results/training-common-zimage-registered-hip-v1"
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=LABEL,request=REQUEST,
                 standby_handoff={"step":"1103606.1748","name":"training80-dcu-standby-df9f9145e0c3-f13r2n07",
                                  "node":claim["node"],"dcu":8},
                 common_candidate={"candidate_commit":proof["candidate_commit"],"git_tree":proof["git_tree"],
                                   "manifest_sha256":proof["artifacts"]["candidate-manifest.json"]["SHA256"]},
                 scheduler_CPU={"receipt_SHA256":authority["CPU_outcome_SHA256"],"stage_SHA256":authority["stage_receipt_SHA256"],
                                "step":terminal[0]})
    for relative,pin in proof["changed_members"].items():
        claim["framework_update_members"][Path(relative).name] = dict(pin,source="verified_integrated_candidate")
    files["CLAIM.json"] = encode(claim)
    controller = files["controller.py"].decode()
    controller = replace(controller, f'(BASE/"{binding}/source-tree", "/common-framework", "ro"),',
                         f'(BASE/"{mounted}/source-tree", "/common-framework", "ro"),')
    files["controller.py"] = controller.encode()
    runner = files["worker.py"].decode()
    runner = replace(runner, '        write("optimizer-step-returned.private.json", {"actual_optimizer_updates": 1})',
                      '        captured = json.loads(progress.read_bytes())\n'
                      '        require(captured["phase"] == "FIXED_FIRST_UPDATE_RETURNED" and captured["failed"] is False\n'
                      '                and len(captured["files"]) == 12 and captured["completed_global_steps"] == 1,\n'
                      '                "The registered first-update archive did not close")\n'
                      '        write("optimizer-step-returned.private.json", {"actual_optimizer_updates": 1})')
    files["worker.py"] = runner.encode()
    host = files.pop("entry.py").decode()
    host = replace(host, previous["LABEL"], LABEL)
    host = replace(host, previous["REQUEST"], REQUEST)
    host = replace(host, f'    common_root = R / "{binding}"',
                   f'    common_root = R / "{mounted}"\n    binding_root = R / "{binding}"')
    host = replace(host, '    cpu_raw = (common_root/"cpu-outcome.json").read_bytes()',
                   '    cpu_raw = (binding_root/"cpu-outcome.json").read_bytes()\n'
                   '    scheduler_raw = (common_root/"cpu-outcome.json").read_bytes()\n'
                   '    need(sha(scheduler_raw)==claim["scheduler_CPU"]["receipt_SHA256"]\n'
                   '         and json.loads(scheduler_raw)["scheduler_regression"]["changed_timestep_rejected"] is True,\n'
                   '         "Original native scheduler regression receipt differs")')
    host = replace(host, 'sha(stage_raw)==target_cpu["stage_receipt_SHA256"]',
                   'sha(stage_raw)==claim["scheduler_CPU"]["stage_SHA256"]')
    host = replace(host, '(common_root/"native-binding.json").read_bytes()',
                   '(binding_root/"native-binding.json").read_bytes()')
    del files["SOURCE_FILES.json"]
    files["SOURCE_FILES.json"] = encode({"files":{name:{"bytes":len(raw),"SHA256":sha(raw)} for name,raw in files.items()}})
    files["entry.py"] = host.encode()
    for name,raw in files.items():
        if name.endswith(".py"):
            compile(raw,name,"exec")
    destination.mkdir()
    for name,raw in files.items():
        with (destination / name).open("xb") as out:
            out.write(raw)
    print(json.dumps(dict(label=LABEL,request=REQUEST,manifest_sha256=sha(files["SOURCE_FILES.json"]),
                          claim_sha256=sha(files["CLAIM.json"]),entry_sha256=sha(files["entry.py"]),
                          bundle_files=len(files)-1),sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    main(parser.parse_args().destination)
