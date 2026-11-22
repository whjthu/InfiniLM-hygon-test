"""The single fixed InternVL eight-DCU Slurm worker and read-only mount setup."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import rebind
import snapshot
from scope import (ASSETS, COMPUTE, CPU360, FRAMEWORK, HERE, NATIVE, NODE, PARENT, REFERENCE,
                   REQUEST, RESULTS, ROOT, RUN, SLURM, digest, encoded, need, pin, seconds_left, write_once)


PROOFS = {
    "outcome.json":"e6309c7fa0d9ec9af21dec4c6e2eff0954f3994ba77cb16b605f271203599b0d",
    "computed-proof.json":"fc83d91fcd3daec96087a870c341a266558e17505b7108d3709d8f7c116be266",
    "localized-plan.json":"16f8dd01a86207d22d039d20d66edf7dc738899aec98b4c72924b3dfdd6a80f4",
}
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
RUNTIME = RESULTS/"training-vlm-swift-offline-runtime-v1"
IMAGE = ROOT/"containers/dtk26-sandbox-r1"
SHELL = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
         "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; "
         "export PYTHONPATH=/swift-runtime/site; exec /usr/bin/python3 -B -u \"$@\"")


def validate_compute_receipts(outcome, proof):
    row=outcome["accounting"]
    need(outcome["state"]=="VERIFIED_ALL17_READONLY_ON_T03_COMPUTE_NODE"
         and row[0]=="1087656.1004" and row[2:4]==["COMPLETED","0:0"] and row[5]==NODE
         and "gres/" not in row[6] and outcome["step_client_rc"]==0
         and outcome["compute_proof_SHA256"]==PROOFS["computed-proof.json"]
         and proof["step"]==row[0] and proof["node"]==NODE
         and proof["read_only_members"]==17 and proof["localized_stat_views"]==20
         and proof["original_inodes_unchanged"] is True
         and proof["localized_plan_SHA256"]==PROOFS["localized-plan.json"],
         "Original .1004 readonly compute proof differs")


def bound_source(claim_pin,manifest_pin):
    claim=json.loads(pin(HERE/"CLAIM.json",expected=claim_pin))
    manifest=json.loads(pin(HERE/"SOURCE_FILES.json",expected=manifest_pin))
    need(claim["request"]==manifest["request"]==REQUEST and claim["GPU_GO"] is claim["root_GO"] is True
         and claim["training_effect_GO"] is False and claim["formal17_qualified"] is False,
         "Root adoption of the single engineering diagnostic is required")
    for name,entry in manifest["files"].items():
        need(Path(name).name==name,"Unexpected source manifest path")
        pin(HERE/name,expected=entry["SHA256"],extent=entry["bytes"])
    return claim


def actual_model_plan(exports_pin):
    values={name:json.loads(pin(COMPUTE/name,expected=sha)) for name,sha in PROOFS.items()}
    validate_compute_receipts(values["outcome.json"],values["computed-proof.json"])
    exports=json.loads(pin(COMPUTE/"export-pins.json",expected=exports_pin))
    plan=values["localized-plan.json"]
    need(exports["request"]==rebind.REQUEST and exports["plan_SHA256"]==rebind.PLAN_SHA
         and len(exports["files"])==20 and len(plan["binds"])==18,"Expected all original shared export bindings")
    assembler=rebind.load_assembler()
    adjusted=rebind.localize(assembler,plan,exports["files"],Path("/proc/self/mountinfo").read_text())
    need(len(adjusted)==20,"Expected exact same-node local binding")
    return assembler,plan,adjusted


def command(plan,epoch,plan_sha):
    binds=[*plan["binds"],f"{HERE}:{HERE}:ro",f"{RUN}:{RUN}:rw",f"{RUN / 'tmp'}:/tmp:rw",f"{FRAMEWORK}:{FRAMEWORK}:ro",
           f"{FRAMEWORK.parent/'source'}:{FRAMEWORK.parent/'source'}:ro",
           f"{CPU360}:{CPU360}:ro",f"{ASSETS}:{ASSETS}:ro",f"{REFERENCE}:{REFERENCE}:ro",
           f"{RUNTIME}:/swift-runtime:ro",f"{rebind.LIB}:/audit:ro","/usr/local/hyhal:/opt/hyhal:ro"]
    argv=[APPTAINER,"exec","--rocm","--contain","--no-home","--pwd","/"]
    for binding in binds:
        argv+=["--bind",binding]
    return [*argv,str(IMAGE),"/bin/bash","--noprofile","--norc","-c",SHELL,"t03-first2",
            str(HERE/"native.py"),str(epoch),plan_sha]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch",type=float,required=True)
    p.add_argument("--claim-sha256",required=True)
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--exports-sha256",required=True)
    p.add_argument("--before",type=Path,required=True)
    p.add_argument("--before-sha256",required=True)
    args=p.parse_args()
    need(os.name=="posix" and os.getuid()==154459 and os.environ.get("SLURM_JOB_ID")==PARENT
         and re.fullmatch(r"[0-9]+",os.environ.get("SLURM_STEP_ID",""))
         and socket.gethostname().split(".")[0]==NODE and len(os.sched_getaffinity(0))>=64,
         "Expected the single owned T03 step and actual 64-CPU affinity")
    seconds_left(args.epoch,NATIVE)
    claim=bound_source(args.claim_sha256,args.manifest_sha256)
    need(claim["exports_SHA256"]==args.exports_sha256,"Published exact original .1004 export pins required")
    before=json.loads(pin(args.before,expected=args.before_sha256))
    snapshot.NODE=NODE
    leased=snapshot.snapshot(True)
    need(before["full_fd_inventory_performed"] is True and leased["node"]==before["node"]==NODE
         and leased["pci"]==before["pci"] and leased["boot_id"]==before["boot_id"]
         and leased["kfd_holder_count"]==before["kfd_holder_count"]==0
         and 0<=leased["monotonic_ns"]-before["monotonic_ns"]<=45_000_000_000,
         "Same-step fresh eight-card PCI and KFD admission differs")
    assembler,plan,adjusted=actual_model_plan(args.exports_sha256)
    RUN.mkdir(mode=0o700)
    for name in ("home","cache","tmp","capture"):
        (RUN/name).mkdir(mode=0o700)
    plan_sha=write_once(RUN/"node-model-plan.json",plan)
    write_once(RUN/"intent.json",dict(request=REQUEST,epoch=args.epoch,deadline=args.epoch+NATIVE,
        node=NODE,step=PARENT+"."+os.environ["SLURM_STEP_ID"],leased=leased,
        plan_SHA256=plan_sha,live_export_paths=len(adjusted),before_SHA256=args.before_sha256,
        published_claim_SHA256=args.claim_sha256,published_manifest_SHA256=args.manifest_sha256,
        native_schedule_steps=25,planned_updates=2,planned_micro_forwards_by_rank=8))
    rc=None
    source_post=False
    try:
        with (RUN/"container.log").open("xb") as out:
            result=subprocess.run(command(plan,args.epoch,plan_sha),stdin=subprocess.DEVNULL,stdout=out,
                                  stderr=subprocess.STDOUT,timeout=seconds_left(args.epoch,SLURM)-20)
        rc=result.returncode
    finally:
        try:
            assembler.verify_originals(plan)
            source_post=True
        finally:
            write_once(RUN/"node-outcome.json",dict(request=REQUEST,container_rc=rc,source_post_unchanged=source_post))
    return int(rc!=0 or not source_post)


if __name__=="__main__":
    raise SystemExit(main())
