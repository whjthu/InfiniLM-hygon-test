"""Revalidate the unchanged Pack16 timing arithmetic and the new owner's terminal proof."""

import argparse
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from gate import HIP_SHA, NODE, OLD_LABEL, OLD_REQUEST, RUNTIME_SHA, digest, label, need, read, resources
from host import INTENT,RUNS,runtime_closure


READER_SHA="822942ad0bb4c7cb23893d4a48e960e15649e44ab20c3ebd7a6e8bd40e776216"
PRECHECK_SHA="2d6152e99c3401dbd05c2c63d64212807df855e98f01bfe030af08e6e51928a0"
DENOMINATOR={"routes":2,"pairs":3,"arms":6,"measured_mm_calls":384,"measured_copies":12}


def analyze(run,runtime,reader,post_raw,post_sha):
    runtime_closure(runtime)
    need(digest(post_raw)==post_sha,"N10_EXACT_INDEPENDENT_POST_RESULT_SHA256")
    post=json.loads(post_raw)
    prior=json.loads(read(run/INTENT,16384))
    owner=prior["request"]
    need(post["state"]=="TERMINAL_AND_INDEPENDENT_PHYSICAL_CLEANUP_VERIFIED"
         and post["refill_permitted"] is True and post["replay"] is False,"N10_INDEPENDENT_PHYSICAL_RELEASE_REQUIRED")
    row=post["row"]
    resources(row[6])
    need(row[1]==label(owner) and row[5]==NODE,"N10_NEW_OUTER_ACCOUNTING_IDENTITY")
    answer={"request":owner,"step":row[0],"independent_post_SHA256":post_sha,"planned":DENOMINATOR,
            "passed":False,"component_improvement_gate":False,"training_performance":False}
    if row[2:4]!=["COMPLETED","0:0"] or not (run/"compute/worker/terminal.json").is_file():
        return {**answer,"state":"NO_COMPLETE_WORKER_RESULT"}
    a=json.loads(read(run/"compute/ACTUAL_PCI_ADMISSION.json",16384))
    raw=read(run/"compute/PACK16_COMPONENT_SCOPE.json",16384,a["scope_SHA256"])
    scope=json.loads(raw)
    worker=json.loads(read(run/"compute/worker/terminal.json",262144))
    need((a["request"],a["step"],a["PCI"])==(owner,row[0],scope["PCI"])
         and a["hip_provider_SHA256"]==HIP_SHA and a["visible_devices"]==1 and a["actual_cgroup"] is True
         and scope["request"]==OLD_REQUEST and scope["label"]==OLD_LABEL and scope["outer_request"]==owner
         and scope["actual_outer_step"]==row[0] and scope["runtime_SHA256"]==RUNTIME_SHA
         and scope["claim_from_old_grant"] is False and worker["request"]==OLD_REQUEST
         and worker["scope_SHA256"]==a["scope_SHA256"] and worker["step"]==row[0] and worker["node"]==NODE
         and worker["PCI"]==a["PCI"] and worker["training_performance"] is False
         and worker["training_RHS_reuse_identity_verified"] is False
         and worker["same_rhs_reuse_per_burst"]==16 and worker["packed_RHS_payload_bytes"]==29491200
         and worker["planned_extra_pack_guard_bytes"]==1024 and worker["source_training_step"]=="1103606.1787",
         "N10_ORIGINAL_MEASUREMENT_AND_NEW_OUTER_IDENTITY")
    read(reader,65536,READER_SHA)
    read(reader.with_name("precheck.py"),65536,PRECHECK_SHA)
    original=runpy.run_path(str(reader))
    pairs=original["statistics_for"](worker["pairs"])
    routes=original["audit_routes"](runtime,run/"compute/worker",scope,worker)
    complete=(worker["complete"] is worker["passed"] is True and len(routes)==2 and pairs["verified_mm_calls"]==384
              and pairs["measured_copy_receipts"]==12)
    return {**answer,"state":"ORIGINAL_COMPONENT_PERFORMANCE_RECONCILED","passed":complete,
            "component_improvement_gate":complete and pairs["arithmetic_performance_gate"],
            "verified_calls":pairs["verified_mm_calls"],"verified_copies":pairs["measured_copy_receipts"],
            "verified_routes":len(routes),"pairs":pairs["wall_speedup"],"paired_medians":pairs["medians"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True)
    for name in ("runtime","reader","post"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--post-sha256",required=True)
    a=p.parse_args()
    label(a.request)
    post=read(a.post,16384,a.post_sha256)
    result=analyze(RUNS/a.request,a.runtime,a.reader,post,a.post_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
