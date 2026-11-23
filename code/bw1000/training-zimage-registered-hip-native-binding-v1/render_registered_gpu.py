"""Build one registered HIP first-step dispatch from the previously audited Slurm envelope."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIOR = HERE.parent / "training-zimage-coco20-real-image-reference-v3"
REPORT = REPO / "results/bw1000/2026-11-23/training-zimage-registered-native-hip-binding-1742.json"
LABEL = "training-zimage-coco20-registered-hip-first-v1"
REQUEST = "c0ad9752e843491a99a48a02a2aa6ff8"


def swap(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Verified Slurm source anchor differs")
    return text.replace(old, new, 1)


def encode(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(destination):
    previous = runpy.run_path(str(PRIOR / "render_gpu.py"))
    original = json.loads((PRIOR / "GPU_SCOPE.json").read_bytes())
    files, entry, _ = previous["build"]()
    if sha(files["SOURCE_FILES.json"]) != original["manifest_sha256"] or sha(entry) != original["entry_sha256"]:
        raise ValueError("Previously qualified Slurm envelope differs")
    scope = json.loads((HERE / "CPU_SCOPE.json").read_bytes())
    result = json.loads(REPORT.read_bytes())
    readback = result["independent_readback"]
    [accounting] = readback["accounting"]
    if accounting[:6] != ["1103606.1742", scope["job_name"], "COMPLETED", "0:0", "9", scope["node"]] or "gres/" in accounting[6]:
        raise ValueError("Original real-asset CPU binding did not complete")
    proof = scope["source_proof"]
    target = scope["target_directory"]
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=LABEL,request=REQUEST,
                 standby_handoff={"step":"1103606.1740","name":"training80-dcu-standby-4882f2eb36e7-f13r2n07",
                                  "node":claim["node"],"dcu":8},
                 common_candidate={"candidate_commit":proof["candidate_commit"],"git_tree":proof["git_tree"],
                                   "manifest_sha256":proof["artifacts"]["candidate-manifest.json"]["SHA256"]},
                 registered_binding={"SHA256":readback["binding_SHA256"],"cli_SHA256":readback["entry_cli_SHA256"],
                                     "CPU_step":accounting[0],"CPU_receipt_SHA256":readback["outcome_SHA256"],
                                     "stage_receipt_SHA256":readback["stage_receipt_SHA256"]})
    for relative,pin in proof["changed_members"].items():
        claim["framework_update_members"][Path(relative).name] = dict(pin, source="verified_integrated_candidate")
    files.pop("fixed_step.py")
    files["CLAIM.json"] = encode(claim)
    files["worker.py"] = (HERE / "registered_worker.py").read_bytes()
    old_stage = "results/training-common-zimage-image-identity-v3"
    controller = files["controller.py"].decode()
    controller = swap(controller, '(BASE/"'+old_stage+'/source-tree", "/common-framework", "ro"),',
                      '(BASE/"'+target+'/source-tree", "/common-framework", "ro"),\n'
                      '            (BASE/"'+target+'", "/binding-read", "ro"),\n'
                      '            (BASE/"results/training-zimage-coco20-fixed-full-hip-image-v2", "/prior-approved", "ro"),')
    files["controller.py"] = controller.encode()
    host = entry.decode()
    host = swap(host, previous["LABEL"], LABEL)
    host = swap(host, previous["REQUEST"], REQUEST)
    host = swap(host, old_stage, target)
    host = swap(host, '    cpu_raw = (common_root/"cpu-result.json").read_bytes()',
                '    cpu_raw = (common_root/"cpu-outcome.json").read_bytes()')
    host = swap(host, '    target_cpu = claim["corrected_image_CPU"]',
                '    target_cpu = claim["registered_binding"]')
    host = swap(host, 'target_cpu["receipt_SHA256"]', 'target_cpu["CPU_receipt_SHA256"]')
    host = swap(host, 'receipt["source_files"]==394', 'receipt["source_files"]==395')
    host = swap(host, 'and json.loads(cpu_raw)["verification"]["passed"] is True,',
                'and json.loads(cpu_raw)["rc"] == 0\n'
                '         and sha((common_root/"native-binding.json").read_bytes())==target_cpu["SHA256"]\n'
                '         and sha((common_root/"source-tree/tools/hygon_training_recipe.py").read_bytes())==target_cpu["cli_SHA256"],')
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
