"""Render the original InternVL domain training package for the verified current node."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
ORIGINAL = HERE.parent / "training-internvl-usgs72-native-first2-hip-v1"
ORIGINAL_MANIFEST_SHA = "c5413b950444b3a6529cea630c3285bfd5ca9e714016aa36b757e4923e02389f"
OLD_REQUEST = "bc7cd25fcd6d4709a8804872fd236d8a"
OLD_LABEL = "training-internvl-usgs72-native-first2-hip-v1"
REPLAY = frozenset({"NATIVE_BINDING.json", "TARGET_MATH_SOURCES.json", "callback.py", "domain_checks.py", "native.py",
                    "original-launch.json", "owned_processes.py", "rebind.py", "slurm_accounting.py"})
ADAPTED = frozenset({"scope.py", "snapshot.py", "node.py", "control_once.py"})


def digest(value):
    return sha256(value).hexdigest()


def canonical(record):
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def replace(source, previous, updated):
    if source.count(previous) != 1:
        raise ValueError("expected a single source anchor: " + previous[:80])
    return source.replace(previous, updated)


NODE_ADAPTER = '''def actual_model_plan(exports_pin):
    # The original export authority is unchanged. Bind the new verified container
    # placement and the independent domain and initial trainable end-to-end check.
    exports = json.loads(pin(COMPUTE / "export-pins.json", expected=exports_pin))
    plan = json.loads(pin(MOUNT_RESULT / "localized-plan.json", expected=PLAN_SHA, cap=65536))
    proof = json.loads(pin(MOUNT_RESULT / "container.stdout", expected=CONTAINER_SHA))
    assets = json.loads(pin(ASSET_RESULT / "outcome.json", expected=ASSET_SHA))
    row, = assets["accounting"]
    facts = assets["asset_check"]
    need(proof["state"] == "VERIFIED_ALL17_READONLY_REGULAR_MOUNTS" and proof["members"] == 17
         and proof["source_weight_body_bytes"] == 0 and assets["passed"] is True
         and assets["runner_rc"] == 0 and row[0] == facts["step"] == "1103606.2561"
         and row[2:4] == ["COMPLETED", "0:0"] and row[5:7] == [NODE, "cpu=2,mem=4G,node=1"]
         and facts["parent"] == PARENT and facts["node"] == NODE and facts["training_rows_verified"] == 72
         and facts["initial_tensor_count"] == 504 and facts["actual_GPU_steps"] == facts["actual_optimizer_updates"] == 0,
         "current asset or model read-only evidence differs")
    need(exports["request"] == rebind.REQUEST and exports["plan_SHA256"] == rebind.PLAN_SHA
         and len(exports["files"]) == 20 and len(plan["binds"]) == 18, "expected all original shared export bindings")
    assembler = rebind.load_assembler()
    adjusted = rebind.localize(assembler, plan, exports["files"], Path("/proc/self/mountinfo").read_text())
    need(len(adjusted) == 20, "expected exact same-node local binding")
    return assembler, plan, adjusted


'''


def render_node(source, claim):
    start, end = source.index("PROOFS = {"), source.index("APPTAINER = ")
    guards = claim["target_original_receipts"]
    constants = ("MOUNT_RESULT = RESULTS / " + repr(guards["mount_path"]) + "\n"
                 "ASSET_RESULT = RESULTS / " + repr(guards["asset_path"]) + "\n"
                 "PLAN_SHA = " + repr(guards["plan_sha"]) + "\n"
                 "CONTAINER_SHA = " + repr(guards["container_stdout_sha"]) + "\n"
                 "ASSET_SHA = " + repr(guards["asset_outcome_sha"]) + "\n")
    source = source[:start] + constants + source[end:]
    start, end = source.index("def validate_compute_receipts("), source.index("def bound_source(")
    source = source[:start] + source[end:]
    start, end = source.index("def actual_model_plan("), source.index("def command(")
    source = source[:start] + NODE_ADAPTER + source[end:]
    return replace(source, "Published exact original .1004 export pins required", "Published exact current-node export pins required")


def render(launch_claim):
    raw = (ORIGINAL / "SOURCE_FILES.json").read_bytes()
    if digest(raw) != ORIGINAL_MANIFEST_SHA:
        raise ValueError("original package manifest differs")
    prior = json.loads(raw)["files"]
    files = {}
    for name in sorted(REPLAY | ADAPTED):
        data = (ORIGINAL / name).read_bytes()
        if len(data) != prior[name]["bytes"] or digest(data) != prior[name]["SHA256"]:
            raise ValueError("original package member differs: " + name)
        files[name] = data
    target = launch_claim["placement"]
    if (launch_claim["schema"], target["parent"], target["node"], target["dcu"], target["cpus"], target["mem_mib"], target["port"]) != (
            "INTERNVL_USGS72_FIRST2_CURRENT_HIP_V2", "1103606", "f13r1n14", 8, 64, 224000, 29967):
        raise ValueError("current target placement differs")
    for name in ("scope.py", "control_once.py"):
        text = files[name].decode()
        text = replace(text, OLD_REQUEST, launch_claim["request"])
        text = replace(text, OLD_LABEL, launch_claim["label"])
        text = replace(text, "1087656", target["parent"])
        text = replace(text, "a14r3n04", target["node"])
        files[name] = text.encode()
    snapshot = replace(files["snapshot.py"].decode(), "1087656", target["parent"])
    files["snapshot.py"] = replace(snapshot, "a01r4n16", target["node"]).encode()
    files["node.py"] = render_node(files["node.py"].decode(), launch_claim).encode()
    # Preserve the original termination/accounting logic and add a second queue
    # view to the clear-placement check before pre and GPU admission.
    control = files["control_once.py"].decode()
    anchor = "\n\ndef accounting(role):"
    second = '''
    for line in query(['squeue','--steps','-h','-j',PARENT,'-o','%i|%N']):
        step, nodes = line.split('|')
        if step in {PARENT+'.batch',PARENT+'.extern'}:
            continue
        need(re.fullmatch(PARENT+r'\\.\\d+',step) and nodes not in ('','(null)','None','Unknown','N/A'),
             'unknown numbered placement')
        members = [nodes] if re.fullmatch(r'f\\d+r\\d+n\\d+',nodes) else query(['scontrol','show','hostnames',nodes])
        need(members and NODE not in members,'queue reports another step on this T03 node')
'''
    files["control_once.py"] = replace(control, anchor, second + anchor).encode()
    for name in ADAPTED | (REPLAY - {"NATIVE_BINDING.json", "TARGET_MATH_SOURCES.json", "original-launch.json"}):
        compile(files[name], name, "exec")
    files["CLAIM.json"] = canonical(launch_claim)
    manifest = {"request": launch_claim["request"], "files": {name: {"bytes": len(data), "SHA256": digest(data)}
                 for name, data in sorted(files.items())}}
    files["SOURCE_FILES.json"] = canonical(manifest)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    claim = json.loads((HERE / "CLAIM.json").read_bytes())
    files = render(claim)
    args.output.mkdir()
    for name, data in sorted(files.items()):
        with (args.output / name).open("xb") as stream:
            stream.write(data)
    print(json.dumps({"request": claim["request"], "files": len(files), "native_bytes_reused": len(REPLAY),
                      "manifest_SHA256": digest(files["SOURCE_FILES.json"])}))


if __name__ == "__main__":
    main()
