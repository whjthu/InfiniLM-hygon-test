"""Freeze the one corrected HIP first-step attempt from the original approved package."""

import argparse
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = HERE.parent / "training-zimage-coco20-fixed-full-hip-v1"
ENTRY = ORIGINAL / "entry.py"
EVIDENCE = REPO / "results/bw1000/2026-11-23/training-zimage-original-image-native-CPU1734-terminal.json"
CANDIDATE = REPO / "results/bw1000/2026-11-23/training-zimage-fixed-full-hip-original1696-and-image-identity-candidate.json"
LABEL = "training-zimage-coco20-fixed-full-hip-image-v2"
REQUEST = "adb3d19584de4c7f90120247adb5847a"


def build():
    old = runpy.run_path(str(ORIGINAL / "render.py"))
    replace, sha, encode = (old[key] for key in ("replace", "sha", "encode"))
    files = old["render"]()
    base = json.loads((ORIGINAL / "SCOPE.json").read_bytes())
    entry = ENTRY.read_bytes()
    if (sha(entry) != base["entry_SHA256"]
            or sha(files["SOURCE_FILES.json"]) != base["rendered_source"]["manifest_sha256"]):
        raise ValueError("The original approved GPU package differs")
    evidence = json.loads(EVIDENCE.read_bytes())
    check = evidence["independent_target_readback"]
    [accounting] = check["accounting"]
    expected_row = ["1103606.1734", "training-zimage-original-image-cpu-b707f8b575a6", "COMPLETED", "0:0",
                    "15", "f13r2n08", "cpu=4,mem=8G,node=1"]
    if accounting != expected_row or check["original_image_check"] is not True:
        raise ValueError("Target-side validation of the native COCO20 image is missing")
    fixed = json.loads(CANDIDATE.read_bytes())["corrected_child"]
    target = json.loads((HERE / "TARGET.json").read_bytes())
    claim = json.loads(files["CLAIM.json"])
    claim.update(label=LABEL, request=REQUEST,
                 standby_handoff={"step": "1103606.1699", "name": "training80-dcu-standby-5f0aa3e687ad-f13r2n07",
                                  "node": claim["node"], "dcu": 8},
                 common_candidate={"candidate_commit": fixed["candidate_commit"], "git_tree": fixed["git_tree"],
                                   "manifest_sha256": fixed["artifacts"]["candidate-manifest.json"]["SHA256"]},
                 corrected_image_CPU={"step": accounting[0], "receipt_SHA256": check["receipt_SHA256"],
                                      "stage_receipt_SHA256": check["target_stage_receipt_SHA256"]})
    for path, pin in fixed["changed_members"].items():
        member = Path(path).name
        claim["framework_update_members"][member] = dict(pin, source="verified_integrated_candidate")
    if fixed["candidate_commit"] != target["corrected_candidate"]:
        raise ValueError("The target and local corrected framework commits differ")
    files.update({"CLAIM.json": encode(claim), "fixed_step.py": (HERE / "fixed_step.py").read_bytes()})
    old_mount = "results/training-common-dp2-zimage-alignment-git-v1"
    new_mount = target["target_directory"]
    files["controller.py"] = replace(files["controller.py"].decode(), old_mount, new_mount).encode()
    entry = replace(entry.decode(), old_mount, new_mount)
    entry = replace(entry, old["LABEL"], LABEL)
    entry = replace(entry, old["REQUEST"], REQUEST)
    entry = replace(entry, '''    receipt = json.loads((common_root/"source-tree-receipt.json").read_bytes())
    expected = claim["common_candidate"]
    need(all(receipt[k]==v for k,v in expected.items()) and receipt["source_checkout_verified"] is True,
         "Common candidate is not the verified 394-file target tree")''',
                    '''    stage_raw = (common_root/"stage-result.json").read_bytes()
    cpu_raw = (common_root/"cpu-result.json").read_bytes()
    target_cpu = claim["corrected_image_CPU"]
    expected = claim["common_candidate"]
    receipt = json.loads(stage_raw)
    need(sha(stage_raw)==target_cpu["stage_receipt_SHA256"] and sha(cpu_raw)==target_cpu["receipt_SHA256"]
         and receipt["candidate_commit"]==expected["candidate_commit"]
         and receipt["source_files"]==394 and receipt["target_source_rehash_verified"] is True
         and sha((common_root/"candidate-manifest.json").read_bytes())==expected["manifest_sha256"]
         and json.loads(cpu_raw)["verification"]["passed"] is True,
         "Corrected 394-file target tree or original native image evidence differs")''')
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    compile(entry, "entry.py", "exec")
    del files["SOURCE_FILES.json"]
    files["SOURCE_FILES.json"] = encode({"files": {name: {"bytes": len(raw), "SHA256": sha(raw)}
                                                  for name, raw in files.items()}})
    return files, entry.encode(), sha


def main(destination):
    files, entry, sha = build()
    destination.mkdir()
    for name, source in {**files, "entry.py": entry}.items():
        with (destination / name).open("xb") as output:
            output.write(source)
    print(json.dumps(dict(label=LABEL, request=REQUEST, files=len(files),
                          manifest_sha256=sha(files["SOURCE_FILES.json"]), entry_sha256=sha(entry),
                          claim_sha256=sha(files["CLAIM.json"])), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
