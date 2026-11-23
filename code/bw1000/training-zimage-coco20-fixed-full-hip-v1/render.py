"""Build one full-HIP diagnostic from the original COCO20 loader and unified candidate."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = "code/bw1000/training-zimage-COCO20-native-first-step-hip-v2"
PARENT_MANIFEST = "12c375aa2eefd8621a59117c0e579a2260f028537db6e70eb0052c7fe6daf579"
CANDIDATE = "results/bw1000/2026-11-23/training-common-DP2-and-ZImage-394-file-target-candidate.json"
CANDIDATE_SHA = "1c19419de30f89bf793907a6b85390a93951403f09bd2fbfab441b6fa30d8144"
LABEL = "training-zimage-coco20-fixed-full-hip-v1"
REQUEST = "1e9507e0268f4d4da456d374e666a784"
TARGET = "f13r2n07"
CONTROL = "f13r2n08"
SCNET = "1103606"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Original HIP source anchor differs: " + before[:90])
    return text.replace(before, after, 1)


def render():
    original = (ROOT / PARENT / "SOURCE_FILES.json").read_bytes()
    candidate_raw = (ROOT / CANDIDATE).read_bytes()
    if sha(original) != PARENT_MANIFEST or sha(candidate_raw) != CANDIDATE_SHA:
        raise ValueError("Original COCO20 or latest common candidate evidence changed")
    evidence = json.loads(candidate_raw)
    files = {}
    for name, pin in json.loads(original)["files"].items():
        blob = (ROOT / pin["path"]).read_bytes()
        if (len(blob), sha(blob)) != (pin["bytes"], pin["SHA256"]):
            raise ValueError("Original HIP package member changed: " + name)
        files[name] = blob

    claim = json.loads(files["CLAIM.json"])
    pci_raw = (ROOT / "local-only/training80-replacement1194900-pci-pins-v1.json").read_bytes()
    if sha(pci_raw) != "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39":
        raise ValueError("Known 640-card PCI evidence changed")
    pcis = json.loads(pci_raw)[TARGET]
    if len(set(pcis)) != 8:
        raise ValueError("Known target PCI identity differs")
    target = evidence["target_materialized_source"]
    if evidence["target_root"] != "results/training-common-dp2-zimage-alignment-git-v1/source-tree":
        raise ValueError("Wrong shared framework candidate materialization")
    claim.update(label=LABEL,request=REQUEST,parent=SCNET,node=TARGET,control_node=CONTROL,
                 whole_seconds=650,worker_seconds=510,
                 standby_handoff={"step":"1103606.1671","name":"training80-dcu-standby-17d728a3e6d0-f13r2n07",
                                  "node":TARGET,"dcu":8},
                 expected_target_pci_sha256=sha(",".join(sorted(pcis)).encode()),
                 common_candidate={k:target[k] for k in ("candidate_commit","git_tree","manifest_sha256")},
                 fixed_hip_reference={
                     "original_step":"1087656.1153",
                     "progress":{"bytes":1873,"SHA256":"a1694d2ffc3491d5c89d896cd73260f9ded215bd83e6ea5eba78e5b14403d2d0"},
                 },
                 no_workload_or_dtype_change=True, actual_optimizer_updates_planned=1,
                 formal_cross_platform_numerics=False, stdlib_host_preflight=True)
    members = claim["framework_update_members"]
    for name in ("zimage_fixed_reference.py", "zimage_first_update.py", "zimage_native_record.py"):
        relative = "generation/src/gencuda/" + name
        members[name] = {k:v for k,v in evidence["independent_local_import"]["replaced_members"][relative].items()
                         if k in ("bytes","SHA256","git_blob","mode")}
        members[name]["source"] = "verified_integrated_candidate"
    files["CLAIM.json"] = encode(claim)
    files["fixed_step.py"] = (HERE / "fixed_step.py").read_bytes()

    worker = files["worker.py"].decode()
    worker = replace(worker, 'ROOT/"coco20_step.py"', 'ROOT/"fixed_step.py"')
    worker = replace(worker, "import gc\n", "import ctypes\nimport gc\n")
    worker = replace(worker, '        props = torch.cuda.get_device_properties(0)',
        '''        pci = ctypes.create_string_buffer(24)
        hip_runtime = ctypes.CDLL("libamdhip64.so")
        need(hip_runtime.hipDeviceGetPCIBusId(pci,len(pci),0) == 0,"Actual device PCI lookup failed")
        physical = pci.value.decode("ascii").lower()
        if len(physical)==7:
            physical = "0000:"+physical
        pre = json.loads((ROOT/"pre.json").read_bytes())
        need(pre["node"]==claim["node"] and physical in pre["PCIs"],"Assigned device escaped approved PCI set")
        result["physical_pci"] = physical
        props = torch.cuda.get_device_properties(0)''')
    files["worker.py"] = worker.encode()

    controller = files["controller.py"].decode()
    controller = replace(controller, "epoch <= time.time() < epoch + 10",
                         "epoch <= time.time() < epoch + 30")
    controller = replace(controller, "    def remaining(reserve=0):\n        available = deadline - time.time() - reserve",
                         "    monotonic_deadline = time.monotonic() + deadline - time.time()\n\n"
                         "    def remaining(reserve=0):\n"
                         "        available = min(deadline-time.time(),monotonic_deadline-time.monotonic())-reserve")
    controller = replace(controller, '    def accounting(name):\n        raw = query("sacct", "-n", "-P", "-j", parent,',
                         '    def accounting(name, accounting_parent=parent):\n'
                         '        raw = query("sacct", "-n", "-P", "-j", accounting_parent,')
    controller = replace(controller, 'and accounting(compat.name) == row',
                         'and accounting(compat.name, "1087656") == row')
    controller = replace(controller, '"--nodelist="+node, "--immediate=5",',
                         '"--nodelist="+node, "--immediate=5", "--threads=1",')
    controller = replace(controller, '"--time="+str(int(budget / 60) + 1), *command]',
                         '"--time="+time.strftime("%H:%M:%S", time.gmtime(int(budget))), *command]')
    controller = replace(controller, "import json\n", "import fcntl\nimport json\n")
    controller = replace(controller,
        "    def snapshot(phase):",
        '''    def owned_standby():
        matching = []
        allowed = claim["standby_handoff"]
        for line in query("scontrol", "show", "step", parent, "-o").splitlines():
            row = fields(line)
            if row["StepId"] in (parent+".batch", parent+".extern"):
                continue
            need(re.fullmatch(parent+r"\\.\\d+",row["StepId"])
                 and row.get("NodeList") not in (None,"","(null)"),"Unknown numbered step scope")
            expression = row["NodeList"]
            members = [expression] if re.fullmatch(r"f\\d+r\\d+n\\d+",expression) else query(
                "scontrol","show","hostnames",expression).splitlines()
            if node not in members or row.get("State") in {
                "COMPLETED","CANCELLED","FAILED","TIMEOUT","NODE_FAIL","PREEMPTED","DEADLINE","OUT_OF_MEMORY"
            }:
                continue
            need(members == [allowed["node"]] and row["StepId"] == allowed["step"]
                 and row["Name"] == allowed["name"] and row["State"] == "RUNNING"
                 and "gres/dcu=8" in row["TRES"].split(","), "Unrecognized live target node owner")
            matching.append(row)
        need(len(matching)<=1,"Duplicate target standby identity")
        return matching[0] if matching else None

    def snapshot(phase):''')
    original_start = controller.index('        for line in query("scontrol", "show", "step", parent, "-o").splitlines():', controller.index('    pre = None'))
    original_end = controller.index('        manifest_raw =',original_start)
    controller = controller[:original_start] + '        standby_before_static_checks = owned_standby()\n' + controller[original_end:]
    controller = replace(controller, "        pre = snapshot(\"pre\")",
        '''        standby = owned_standby()
        if standby is not None:
            need(standby_before_static_checks is not None
                 and standby["StepId"] == standby_before_static_checks["StepId"],
                 "Standby changed during bounded preparation")
            save("standby-handoff-intent.json",dict(node=node,step=standby["StepId"],name=standby["Name"]))
            result["standby_cancel_attempted"] = True
            query("scancel","--signal=TERM",standby["StepId"])
            row = wait_accounting(standby["Name"],cap=35)
            need(row is not None and row[0]==standby["StepId"] and row[5]==node,"Original standby terminal accounting missing")
            need(owned_standby() is None,"Original standby still appears live")
            result["standby_terminal"] = row
        pre = snapshot("pre")''')
    controller = replace(controller, "    pre = None\n    try:\n        check =",
        '''    pre = None
    handoff_lock = (BASE/"results/.training80-dcu-standby.lock").open("ab")
    try:
        fcntl.flock(handoff_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check =''')
    controller = replace(controller, '    save("outcome.json", result)\n',
                         '    handoff_lock.close()\n    save("outcome.json", result)\n')
    controller = replace(controller, "        pre = snapshot(\"pre\")",
                         '        pre = snapshot("pre")\n'
                         '        need(sha256(",".join(sorted(pre["PCIs"])).encode()).hexdigest()=='
                         'claim["expected_target_pci_sha256"],"Actual target PCI inventory differs")')
    controller = replace(controller,
                         '            (BASE/"results/training-qwen-a68952c-swift-initial-CPU-v1/framework", "/common-framework", "ro"),',
                         '            (BASE/"results/training-common-dp2-zimage-alignment-git-v1/source-tree", "/common-framework", "ro"),\n'
                         '            (BASE/"results/training-zimage-COCO20-native-first-step-hip-v2/first-update-private", "/original-first", "ro"),')
    files["controller.py"] = controller.encode()
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    files["SOURCE_FILES.json"] = encode({"files":{
        name: {"bytes":len(raw),"SHA256":sha(raw)} for name, raw in files.items()
    }})
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    out = parser.parse_args().destination
    files = render()
    out.mkdir()
    for name, data in files.items():
        with (out/name).open("xb") as stream:
            stream.write(data)
    print(json.dumps({"files":len(files),"manifest_sha256":sha(files["SOURCE_FILES.json"]),
                      "claim_sha256":sha(files["CLAIM.json"]),"label":LABEL,"request":REQUEST}))
