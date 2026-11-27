"""Bind the exact original CP3 and the native CPU step 4–25 occurrence plan."""
import json
import math
import os
from pathlib import Path
import stat

from cp3_contract import encode, need
from hashlib import sha256


REQUEST="f872143f007845ba88b5f4032e8d912d"
LABEL="training-internvl-gqa56-cp3-to-cp25-native-v1"
PORT=29995
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
CP3_ROOT=ROOT/"results/training-internvl-gqa56-cp2-to-cp3-native-v1"
CHECKPOINT=Path("/gqa-recovery-cp3")
CP3_NATIVE="932a8579bb727ae8f44f78ccccafbdb6b28aa35b18dfa54bd16c70b4e8d8b345"
CP3_FILES="957ef43aa6b70ae74742c95d29adaf57a34622adb3856ec892f34127b881511d"
CP3_ARCHIVE="179456c6d4d565d401ac458fc80ba666754499e18e92e691fbe73cb6475e6123"
CP3_FP="e264b9910582a278c1dfa998d1502b65798c8c9e7d10c250ad7188b54db2a0b0"
SOURCE_MANIFEST="a8e6744dc55ebbf02977a8acd335fe7d60a7353af5b3af9d2d5cf2eca07dbfce"
CPU_RESULT="670b572a335f4d6b5a084b662058a60e16eca3a2f2f92b4f6a08a49698d75086"
CPU_PLAN="04f7385ba2b79dd42dddedf03f2ec8f3f1848ae4ee13727f8eea0a4a01a4e005"
CPU_REQUEST="1ccc01ddf76049dca2e32d9f013be644"
CPU_ROOT=ROOT/f"incoming/internvl-gqa56-native-arrow-epochs1to12-{CPU_REQUEST}/output"
CPU_ROW=["1103606.4738","internvl-gqa56-arrow-series-1ccc01ddf760","COMPLETED","0:0","28","f13r1n14",
         "cpu=8,mem=24G,node=1","2026-11-27T19:23:15","2026-11-27T19:23:43"]
SAVE_STEPS=(10,20,25)
STEPS=tuple(range(4,26))
EPOCHS=tuple(range(1,13))


def lr(cursor):
    progress=(cursor-1)/(25-1)
    return 1e-4 * (0.5 * (1 + math.cos(math.pi * progress)))


def validate_cpu_result(result, *, bounded_projection=False):
    require=(result.get("request"),result["actual_step"],result["verified_continuation_slots"],
             result["verified_epochs"],result["verified_all_epochs_slots"],
             result["actual_loader_constructions_per_rank"],result["actual_set_epoch_calls_per_rank"],
             result["start_global_step"],result["nominal_max_steps"])
    need(require==(CPU_REQUEST,CPU_ROW[0],704,12,768,1,12,3,25) and
         (bounded_projection or result["start_epoch"]==1.5) and
         result["continuation_plan_SHA256"]==CPU_PLAN and result["continuation_plan_bytes"]==141110 and
         result["original_CP3_native_SHA256"]==CP3_NATIVE and result["epoch1_byte_matches_4726"] is True and
         result["actual_optimizer_updates"]==0 and result["actual_forward_order_verified"] is False and
         result["actual_template_collator_invoked"] is False and result["production_workers"]==4,
         "EXACT_ORIGINAL_4738_ZERO_GRES_PLAN_BOUNDARY")


def load_plan(root,pin):
    result=json.loads(pin(root/"result.json",expected=CPU_RESULT,extent=9579))
    validate_cpu_result(result)
    raw=pin(root/"steps4-25-plan.private.json",expected=CPU_PLAN,extent=141110,cap=150000)
    value=json.loads(raw)
    need(encode(value)==raw,"EXACT_ORIGINAL_4738_CANONICAL_PLAN_BYTES")
    return parse_plan(value)


def parse_plan(value):
    need((value["start_global_step"],value["start_epoch"],value["max_steps"])==(3,1.5,25) and
         [row["global_step"] for row in value["updates"]]==list(STEPS),"EXACT_CP3_START_CURSOR_AND_22_STEPS")
    ranks=[[] for _ in range(8)]
    for entry in value["updates"]:
        step=entry["global_step"]
        epoch,half=divmod(step-1,2)
        slots=entry["slots"]
        need(len(slots)==32 and
             [(slot["accumulation"],slot["rank"]) for slot in slots]==
             [(micro,rank) for micro in range(1,5) for rank in range(8)],"EXACT_STEP_32_ACCUMULATED_SLOTS")
        for slot in slots:
            need((slot["update"],slot["epoch_index"],slot["rank_offset0"])==
                 (step,epoch,4*half+slot["accumulation"]-1) and
                 all(type(slot[key]) is int and 0<=slot[key]<64 for key in ("logical","physical")) and
                 1<=slot["source_ordinal"]<=64,"EXACT_EPOCH_CURSOR_OF_SLOT")
            ranks[slot["rank"]].append(slot)
    need(all(len(rows)==88 for rows in ranks),"EXACT_88_FORWARDS_PER_RANK")
    return ranks


def verify_epochs_from_live(ranks,order,original_matrix):
    """Bind every expected logical occurrence to the verified live Arrow permutation."""
    need(sorted(order)==list(range(64)),"EXACT_LIVE_ROSTER_PERMUTATION")
    original={row["physical"]:(row["ordinal"],row["row_SHA256"]) for rank in original_matrix for row in rank}
    need(len(original)==64,"EXACT_BOUND_64_EPOCH1_PHYSICAL_IDENTITIES")
    for rank,rows in enumerate(ranks):
        seen=set()
        for row in rows:
            key=(row["epoch_index"],row["physical"])
            need(row["rank"]==rank and order[row["logical"]]==row["physical"] and key not in seen and
                 original[row["physical"]]==(row["source_ordinal"],row["row_SHA256"]),
                 "EXACT_EXPECTED_EPOCH_PHYSICAL_LINEAGE")
            seen.add(key)


def verify_checkpoint(source_directory,cp,pin,clock):
    need(cp.resolve(strict=True)==cp and os.statvfs(cp).f_flag & os.ST_RDONLY,"ORIGINAL_CP3_RO_MOUNT")
    native=json.loads(pin(source_directory/"native-outcome.json",expected=CP3_NATIVE,extent=9724))
    need(native["actual_step"]=="1103606.4728" and native["checkpoint3_sealed"] is True and
         native["native_returncode"]==0,"EXACT_4728_NATIVE_CP3_RESULT")
    inventory=json.loads(pin(source_directory/"checkpoint-3-files.json",expected=CP3_FILES,extent=1955))
    members=inventory["files"]
    need(inventory["step"]==3 and inventory["native_schedule_steps"]==25 and len(members)==17 and
         set(members)=={p.name for p in cp.iterdir()},"EXACT_17_SEALED_CP3_MEMBERS")
    for name, item in sorted(members.items()):
        path=cp/name
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,"rb") as stream:
            before=os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_size==item["bytes"],"ORIGINAL_CP3_FILE_EXTENT")
            hasher=sha256()
            while chunk:=stream.read(4*1024**2):
                clock()
                hasher.update(chunk)
            stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            need(stamp(before)==stamp(os.fstat(fd))==stamp(path.lstat()) and hasher.hexdigest()==item["SHA256"],
                 "ORIGINAL_CP3_SAME_FD_FILE_SHA")
    trainer=json.loads(pin(cp/"trainer_state.json",expected=members["trainer_state.json"]["SHA256"]))
    need((trainer["global_step"],trainer["epoch"],trainer["max_steps"])==(3,1.5,25),"ACTUAL_CP3_CURSOR")
    return len(members)
