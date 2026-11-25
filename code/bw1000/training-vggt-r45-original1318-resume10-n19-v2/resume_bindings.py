"""Bind the actual .1318 checkpoint, eight-pair dataset and comparison tape."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat


REQUEST="336ad92753d149f491c0cd447e9377fc"
DISPATCH="282db2c2e5fb41948dd33591f290a7e1"
LABEL="training-vggt-r45-original1318-resume10-n19-v2"
ORIGINAL_NODE='f13r4n09'
NODE='f13r1n19'
PLACEMENT_HISTORY_SHA='de8b3ae878e12ae097ab448c16ff1641207c33c26d04d6a59eaee46c8bdb7ea0'
PRIOR_N19_ROWS=(['1103606.3204', 'training80-g4-fresh-e016a011a6d1', 'COMPLETED', '0:0', '1', 'f13r1n19', 'cpu=1,mem=256M,node=1'], ['1103606.3205', 'training80-dcu-standby-e016a011a6d1-f13r1n19', 'COMPLETED', '0:0', '10642', 'f13r1n19', 'cpu=8,gres/dcu=8,mem=16G,node=1'])
N19_PCIS=('0000:09:00.0', '0000:36:00.0', '0000:55:00.0', '0000:77:00.0', '0000:85:00.0', '0000:b5:00.0', '0000:d5:00.0', '0000:f5:00.0')
PREVIOUS_FAILED_LABEL='training-vggt-r45-original1318-resume10-n19-v1'
PREVIOUS_FAILURE_SHA='86ab13825c2a84170e5a154b7fe1e0895831e026e1fd64741246a67e4fd009c4'
BASE_LABEL="training-vggt-tum-eight-pairs25-recovery-v2"
BASE_REQUEST="d7ff76e21b8248efbc8b0a9830c8344c"
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
OLD=ROOT/"results"/BASE_LABEL
FORENSIC="a99d4191ebfe0927712534268adb585bbbf635c267c0d994b6868fa76847fd0b"
CP_BYTES=2594191443
CP_SHA="cadc004d4055d62226baefa7aa01e1be1fed1a4f40b64a6774c18bf51f896dae"
CP_BOUNDARY="195fd7202d40fd0a32e1876ffb12451725f01cbee7641c9716575fc28cd37430"
OLD_RECEIPT="c4d7fcd30d16dc9d525d8bac8f5649352b0e0a12b7c85fca9bb5ff3a0f8f4205"
OLD_TAPE="10e2f03343ddaa99ae64faa4db37e8ec97b06ba589cc2dc25aa980ba3f08abe6"
OLD_FINAL="27e3e628dbb163cc96e8f0ca17d6978a1453dbd88c8f776d8e4973faa7302d6c"
BUNDLE="fdc6b68196db7403a00dada66bda274d08d737e83336cb4c3758362f7438b14c"
BACKEND="762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998"
SCHEDULER=dict(type="constant_learning_rate_in_optimizer_param_groups",mutable_scheduler=False,learning_rate=2e-5)
SCALARS=("loss","grad_norm","camera_head_grad_norm_before_clip")
CHECKPOINT=("model_SHA256","optimizer_SHA256","placements_SHA256","rng_SHA256","full_boundary_SHA256")


def need(ok,code):
    if not ok:
        raise ValueError(code)


def pinned(path,digest=None,size=None,cap=16384,*,stream=False):
    metadata=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(metadata.st_mode) and
         0<metadata.st_size<=cap and (size is None or size==metadata.st_size),"VGGT_EXACT_FILE_EXTENT")
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    h=sha256()
    consumed=0
    raw=bytearray()
    with path.open("rb") as f:
        opened=os.fstat(f.fileno())
        while block:=f.read(8<<20):
            consumed+=len(block)
            need(consumed<=cap,"VGGT_EXACT_READ_BOUND")
            h.update(block)
            if not stream:
                raw.extend(block)
        after=os.fstat(f.fileno())
    need(identity(metadata)==identity(opened)==identity(after)==identity(path.lstat())
         and consumed==metadata.st_size and (digest is None or h.hexdigest()==digest),"VGGT_STABLE_EXACT_FILE_HASH")
    return h.hexdigest() if stream else bytes(raw)


def reference(root):
    record=json.loads(pinned(root/"receipt.json",OLD_RECEIPT,17183,32768))
    tape=pinned(root/"train/steps.ndjson",OLD_TAPE,11379,16384)
    tape=[json.loads(line) for line in tape.splitlines()]
    need(tape==record["steps"] and record["arm"]=="continuous" and record["executed_optimizer_steps"]==25
         and record["resume_source_SHA256"] is None and record["final_state_SHA256"]==OLD_FINAL
         and record["bundle_SHA256"]==BUNDLE and record["backend_SHA256"]==BACKEND
         and record["checkpoints"]["10"]["file_SHA256"]==CP_SHA
         and record["checkpoints"]["10"]["full_boundary_SHA256"]==CP_BOUNDARY,"VGGT_EXACT_ORIGINAL_REFERENCE")
    verify_steps(record,0)
    return record


def verify_steps(receipt,start):
    rows=receipt["steps"]
    need([r["global_optimizer_step"] for r in rows]==list(range(start+1,26))
         and all(r["batch_cursor"]==(r["global_optimizer_step"]-1)%8 for r in rows)
         and all(type(r[k]) in (float,int) and math.isfinite(r[k]) for r in rows for k in SCALARS)
         and receipt["scheduler"]==SCHEDULER and receipt["gradient_accumulation"]==1,
         "VGGT_EXACT_FINITE_STEPS_AND_EIGHT_PAIR_CURSOR")
    return rows


def compare(old,current):
    previous,actual=verify_steps(old,0),verify_steps(current,10)
    need(current["arm"]=="resume10" and current["executed_optimizer_steps"]==15 and current["resume_source_SHA256"]==CP_SHA
         and current["process"]!=old["process"] and old["epoch"]<current["epoch"]<current["deadline"]
         and all(type(current["process"][k]) is int and current["process"][k]>0 for k in ("pid","startticks"))
         and all(old[k]==current[k] for k in ("bundle_SHA256","backend_SHA256","checkpoint_identity"))
         and set(current["checkpoints"])=={"20","25"} and current["formal_CO3D_eligible"] is False,
         "VGGT_NEW_EPOCH_AND_ORIGINAL_CHECKPOINT_IDENTITY")
    differences=[dict(step=before["global_optimizer_step"],cursor_equal=before["batch_cursor"]==after["batch_cursor"],
                      errors={k:abs(after[k]-before[k]) for k in SCALARS})
                 for before,after in zip(previous[10:],actual,strict=True)]
    digest_checks={}
    for key,row in current["checkpoints"].items():
        n=int(key)
        need((row["global_step"],row["next_global_step"],row["next_batch_cursor"])==(n,n+1,n%8)
             and row["optimizer_group_learning_rates"]==[2e-5],"VGGT_NEW_CHECKPOINT_CURSOR_AND_LR")
        digest_checks[key]={field:row[field]==old["checkpoints"][key][field] for field in CHECKPOINT}
    final=current["final_state_SHA256"]==old["final_state_SHA256"]
    ok=final and all(all(row.values()) for row in digest_checks.values()) and all(
        row["cursor_equal"] and max(row["errors"].values())<=1e-6 for row in differences)
    return dict(comparison_passed=ok,planned_new_updates=15,completed_new_updates=15,original_continuous_replayed=False,
                checkpoint_exact=digest_checks,final_state_equal=final,step_differences=differences,
                formal_CO3D_eligible=False,formal_performance_eligible=False)
