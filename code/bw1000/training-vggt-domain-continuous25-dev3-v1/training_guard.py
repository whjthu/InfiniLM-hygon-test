"""Require the actual 321 terminal, clean post and original continuous25 head before dev scoring."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

TRAIN = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-vggt-tum-continuous25-recovery-v1")
REQUEST = "fd7afad15b594969b8e9273b939f1fd6"
NODE = "f11r4n19"
GPU = "1087656.321"
PRE = "1087656.320"
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
TERMINAL = {"COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","PREEMPTED","BOOT_FAIL","DEADLINE"}
SHA = re.compile(r"[0-9a-f]{64}")
FILES = ("outcome.json", "arms/continuous/receipt.json", "arms/continuous/native-result.json",
         "before.json", "after.json")
HEAD_FILE_SHA256 = "522483d98b126de0dcf00825cb32751d8bba7ca98a612d9d098a066b97c951e9"
HEAD_MODULE_SHA256 = "28d0b01564e01d59b3cbb2b20b43b4e5c5f69dbcaa042a1e0c5bf714f51c88ed"
HEAD_STAT = (864719552,10703273036,1794875418000000000)
CONTINUOUS_RESULT_SHA256 = "ee8ce9936e667173aa220b766fb10a601106d6e4f4e6d2bbd27d364b2675cc46"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, pin):
    if path.is_symlink() or not path.is_file() or not isinstance(pin,str) or not SHA.fullmatch(pin):
        raise ValueError("Missing pinned actual training evidence")
    raw=path.read_bytes()
    if sha(raw)!=pin:
        raise ValueError("Training evidence bytes changed")
    return json.loads(raw)


def check_binding(binding):
    keys={"schema","original_request","original_publication","gpu_step","post_step","files_SHA256",
          "head_file_SHA256","head_module_SHA256","head_stat","allow_recovery_failure"}
    if set(binding)!=keys or binding["schema"]!="vggt-continuous25-dev3-training-binding-v1":
        raise ValueError("Unexpected training binding schema")
    if (binding["original_request"]!=REQUEST or not re.fullmatch(r"218d6e6f[0-9a-f]{32}",binding["original_publication"])
            or binding["gpu_step"]!=GPU or not re.fullmatch(r"1087656\.[0-9]+",binding["post_step"])
            or binding["post_step"] in (GPU,PRE) or set(binding["files_SHA256"])!=set(FILES)):
        raise ValueError("Original training handles or provenance changed")
    if not all(SHA.fullmatch(x) for x in [*binding["files_SHA256"].values(),binding["head_file_SHA256"],binding["head_module_SHA256"]]):
        raise ValueError("Missing actual training/head SHA")
    if (not isinstance(binding["head_stat"],list) or len(binding["head_stat"])!=3
            or not all(type(v) is int and v>0 for v in binding["head_stat"])
            or type(binding["allow_recovery_failure"]) is not bool):
        raise ValueError("Invalid head stat or recovery decision")
    if (binding["head_file_SHA256"]!=HEAD_FILE_SHA256 or binding["head_module_SHA256"]!=HEAD_MODULE_SHA256
            or tuple(binding["head_stat"])!=HEAD_STAT
            or binding["files_SHA256"]["arms/continuous/native-result.json"]!=CONTINUOUS_RESULT_SHA256):
        raise ValueError("Expected the actual continuous25 head and result")


def accounting():
    output=subprocess.check_output(["sacct","-n","-P","-j","1087656","-o",COLS],timeout=12,text=True)
    rows={}
    for line in output.splitlines():
        values=line.split("|")
        if len(values)==7 and values[0] not in rows:
            rows[values[0]]=values
        elif len(values)==7:
            raise ValueError("Repeated Slurm accounting identity")
    return rows


def verify(binding, docs, current_rows):
    check_binding(binding)
    outcome=docs["outcome.json"]
    if outcome["request"]!=REQUEST or outcome["publication"]!=binding["original_publication"]:
        raise ValueError("Outcome lineage differs")
    stages=outcome["stages"]
    for phase,identity in (("pre",PRE),("gpu",GPU),("post",binding["post_step"])):
        row=stages[phase]["row"]
        if (not isinstance(row,list) or len(row)!=7 or row[0]!=identity or row[5]!=NODE
                or current_rows.get(identity)!=row or row[2].split()[0] not in TERMINAL
                or not stages[phase].get("terminal_visible")):
            raise ValueError("Original job terminal accounting is absent, stale or foreign")
    if (stages["pre"]["row"][2:4]!=["COMPLETED","0:0"] or stages["pre"].get("passed") is not True
            or stages["post"]["row"][2:4]!=["COMPLETED","0:0"] or stages["post"].get("passed") is not True):
        raise ValueError("Original independent before/post scope did not pass")
    before,after=docs["before.json"],docs["after.json"]
    if (before["pci"]!=after["pci"] or before["boot_id"]!=after["boot_id"]
            or after["kfd_holder_count"]!=0 or after["full_fd_inventory_performed"] is not True):
        raise ValueError("Original post failed device ownership closure")
    receipt=docs["arms/continuous/receipt.json"]
    result=docs["arms/continuous/native-result.json"]
    if receipt["result_SHA256"]!=binding["files_SHA256"]["arms/continuous/native-result.json"]:
        raise ValueError("Continuous result is not linked by its original receipt")
    metrics=result["metrics"]
    checks=result["checks"]
    if (receipt["arm"]!="continuous" or receipt["executed_optimizer_steps"]!=25
            or metrics["global_optimizer_step"]!=25 or metrics["executed_optimizer_steps"]!=25
            or metrics["completed_global_step_before_resume"]!=0 or metrics["measurement_steps"]!=20
            or metrics["sequence_count"]!=2 or metrics["cached_valid_batches"]!=2
            or metrics["camera_head_sha_after"]!=binding["head_module_SHA256"]):
        raise ValueError("Original completed continuous25 training proof absent")
    if not all(checks.get(k) is True for k in ("camera_head_weights_changed","frozen_aggregator_unchanged",
                                             "nonzero_finite_camera_head_gradients","main_forward_device")):
        raise ValueError("Continuous native model checks failed")
    head,=[a for a in result["artifacts"] if a["path"]=="/pilot/arms/continuous/train/camera_head.pt"]
    if head["sha256"]!=binding["head_file_SHA256"]:
        raise ValueError("Head file is not the original artifact")
    if receipt["final_state_SHA256"]!=metrics["final_state_sha256"] or receipt["checkpoint_identity"]!=metrics["checkpoint_identity"]:
        raise ValueError("Continuous final boundary identity differs")
    steps=receipt["steps"]
    if ([r["global_optimizer_step"] for r in steps]!=list(range(1,26))
            or any(r["batch_cursor"]!=(r["global_optimizer_step"]-1)%2 for r in steps)):
        raise ValueError("Continuous data cursor or step denominator differs")
    recovery_passed=outcome.get("passed") is True
    if not recovery_passed and not binding["allow_recovery_failure"]:
        raise ValueError("Training recovery failed or incomplete; independent dev decision required")
    return dict(original_training_step=GPU,original_post_step=binding["post_step"],
                original_recovery_passed=recovery_passed,
                recovery_failure_explicitly_accepted=binding["allow_recovery_failure"],
                original_model_training_updates=25,
                trained_head_file_SHA256=binding["head_file_SHA256"],
                trained_head_module_SHA256=binding["head_module_SHA256"])


def preflight(binding, rows=None):
    check_binding(binding)
    docs={name:read(TRAIN/name,binding["files_SHA256"][name]) for name in FILES}
    report=verify(binding,docs,accounting() if rows is None else rows)
    head=TRAIN/"arms/continuous/train/camera_head.pt"
    stat=head.stat()
    if head.is_symlink() or [stat.st_size,stat.st_ino,stat.st_mtime_ns]!=binding["head_stat"]:
        raise ValueError("Original head stat differs")
    digest=hashlib.sha256()
    with head.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            digest.update(part)
    if digest.hexdigest()!=binding["head_file_SHA256"]:
        raise ValueError("Original head bytes differ")
    return report
