"""Freeze a single Hygon step21 profiler from the qualified registered envelopes."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "training-zimage-registered-hip-scheduler-guard-v2"
RECOVERY = HERE.parent / "training-zimage-registered-hip-native-recovery-v1"
LABEL = "training-zimage-coco20-registered-hip-profile21-v1"
REQUEST = "ab198e6bac9e4327836d9031d2b614e6"
TARGET = "results/training-common-zimage-profile21-v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False) + "\n").encode()


def change(text,before,after):
    if text.count(before) != 1:
        raise ValueError("Verified profile source anchor changed")
    return text.replace(before,after,1)


def build(prior,recovery,destination):
    old = json.loads((PREVIOUS/"GPU_SCOPE.json").read_bytes())
    original = (prior/"SOURCE_FILES.json").read_bytes()
    entry = (prior/"entry.py").read_bytes()
    if sha(original) != old["manifest_sha256"] or sha(entry) != old["entry_sha256"]:
        raise ValueError("Previously qualified single-card envelope differs")
    files = {}
    for name,pin in json.loads(original)["files"].items():
        value = (prior/name).read_bytes()
        if (len(value),sha(value)) != (pin["bytes"],pin["SHA256"]):
            raise ValueError("Changed envelope member")
        files[name]=value
    scope = json.loads((HERE/"SOURCE_SCOPE.json").read_bytes())
    proof = scope["source_proof"]
    claim = json.loads(files["CLAIM.json"])
    claim.update(request=REQUEST,label=LABEL,
                 standby_handoff={"step":"1103606.1765","name":"training80-dcu-standby-a7f726513b72-f13r2n07",
                                  "node":claim["node"],"dcu":8},
                 common_candidate={"candidate_commit":proof["candidate_commit"],"git_tree":proof["git_tree"],
                                   "manifest_sha256":proof["artifacts"]["candidate-manifest.json"]["SHA256"]},
                 profile_source_step="1103606.1759",expected_checkpoint20_SHA256=scope["checkpoint20_SHA256"],
                 actual_optimizer_updates_planned=2)
    recovery_source=json.loads((RECOVERY/"SOURCE_SCOPE.json").read_bytes())["source_proof"]
    if proof["parent_commit"]!=recovery_source["candidate_commit"]:
        raise ValueError("Profile must inherit the qualified recovery entry")
    claim["registered_binding"]["cli_SHA256"] = recovery_source["changed_members"]["tools/hygon_training_recipe.py"]["SHA256"]
    claim["framework_update_members"]["zimage_native_record.py"] = dict(
        proof["changed_members"]["generation/src/gencuda/zimage_native_record.py"],source="verified_integrated_candidate")
    files["CLAIM.json"]=encode(claim)
    recovery_scope=json.loads((RECOVERY/"GPU_SCOPE.json").read_bytes())
    mf=(recovery/"SOURCE_FILES.json").read_bytes()
    runner=(recovery/"worker.py").read_bytes()
    if sha(mf)!=recovery_scope["manifest_sha256"] or sha(runner)!=json.loads(mf)["files"]["worker.py"]["SHA256"]:
        raise ValueError("Changed engineering worker")
    worker=runner.decode()
    worker=change(worker,'PHASES = {"continuous": list(range(1, 26)), "resume20": list(range(21, 26))}',
                  'PHASES = {"resume20": [21]}\nRECOVERY = Path("/recovery-evidence")')
    worker=change(worker,'    output = ROOT / mode','    output = ROOT / "registered-output"')
    worker=change(worker,'ROOT / ("pre.json" if mode == "continuous" else "between.json")','ROOT / "pre.json"')
    worker=change(worker,'ROOT / "continuous-worker-result.json"','RECOVERY / "continuous-worker-result.json"')
    worker=change(worker,'ROOT / "continuous/checkpoint-20/state.private.pt"','RECOVERY / "continuous/checkpoint-20/state.private.pt"')
    worker=change(worker,'ROOT / "continuous/result.json"','RECOVERY / "continuous/result.json"')
    worker=change(worker,'len(pins20) == 1 and cp.stat().st_size == pins20[0]["bytes"]',
                  'len(pins20) == 1 and pins20[0]["SHA256"] == claim["expected_checkpoint20_SHA256"]\n'
                  '                    and cp.stat().st_size == pins20[0]["bytes"]')
    worker=change(worker,'extra = ["--checkpoint", str(cp)]','extra = ["--checkpoint", str(cp), "--profile-resumed21"]')
    worker=change(worker,'and result["actual_optimizer_updates"] == len(expected_steps)',
                  'and result["actual_optimizer_updates"] == 2 and result["profile_summary"]["passed"] is True')
    worker=change(worker,'        record.update(passed=True, completed_global_steps=expected_steps, actual_optimizer_updates=len(expected_steps),',
                  '        warmup = result["diagnostic_warmup"]\n'
                  '        require(warmup["actual_optimizer_calls"] == 1 and warmup["restored20_after_warmup"] is True,\n'
                  '                "Profiler warmup did not rewind to the original checkpoint")\n'
                  '        record.update(passed=True, completed_global_steps=expected_steps, actual_optimizer_updates=2,')
    worker=change(worker,'        write(mode + "-worker-result.json", record)','        write("worker-result.json", record)')
    worker=change(worker,'            record["completed_global_steps"] = json_read(output / "progress.json")["completed_global_steps"]\n'
                         '            record["actual_optimizer_updates"] = None',
                  '            progress = json_read(output / "progress.json")\n'
                  '            record["completed_global_steps"] = progress["completed_global_steps"]\n'
                  '            if (output / "diagnostic-warmup-returned.json").exists() or record["completed_global_steps"]:\n'
                  '                record["actual_optimizer_updates"] = None')
    worker=change(worker,'main(sys.argv[1])','main("resume20")')
    files["worker.py"]=worker.encode()
    old_target="results/training-common-zimage-scheduler-guard-v2"
    controller=change(files["controller.py"].decode(),f'(BASE/"{old_target}/source-tree", "/common-framework", "ro"),',
                      f'(BASE/"{TARGET}/source-tree", "/common-framework", "ro"),\n'
                      '            (BASE/"results/training-zimage-coco20-registered-hip-recovery-v1", "/recovery-evidence", "ro"),')
    files["controller.py"]=controller.encode()
    entry=change(entry.decode(),old["label"],LABEL)
    entry=change(entry,old["request"],REQUEST)
    entry=change(entry,f'    common_root = R / "{old_target}"',
                 f'    common_root = R / "{TARGET}"\n    scheduler_root = R / "{old_target}"')
    entry=change(entry,'(common_root/"cpu-outcome.json").read_bytes()','(scheduler_root/"cpu-outcome.json").read_bytes()')
    entry=change(entry,'sha(stage_raw)==claim["scheduler_CPU"]["stage_SHA256"]','receipt["profile21_option_verified"] is True')
    files["SOURCE_FILES.json"]=encode({"files":{n:{"bytes":len(v),"SHA256":sha(v)} for n,v in files.items()}})
    files["entry.py"]=entry.encode()
    for name,raw in files.items():
        if name.endswith(".py"):compile(raw,name,"exec")
    destination.mkdir()
    for name,raw in files.items():
        with (destination/name).open("xb") as out:out.write(raw)
    print(json.dumps(dict(request=REQUEST,label=LABEL,bundle_files=len(files)-1,
                          claim_sha256=sha(files["CLAIM.json"]),manifest_sha256=sha(files["SOURCE_FILES.json"]),
                          entry_sha256=sha(files["entry.py"])),sort_keys=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior",type=Path)
    parser.add_argument("recovery",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    build(args.prior,args.recovery,args.destination)
