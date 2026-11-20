"""Saved metadata only; no sacct refresh, input/media/PT body or model import."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

LABEL="training-qwen-usgs-exposure25-resume20-hip-v1"
REQUEST="f572fda6784140cbbdcae7bb91a916b9"
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")/LABEL


def consume(context):
    started=time.monotonic()
    total=0
    pins={}
    def need(ok):
        if not ok:
            raise ValueError("Original paired training metadata contract differs")
    def load(name,expected=None):
        nonlocal total
        p=ROOT/name
        need(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()))
        before=p.stat()
        need(0<before.st_size<=8*1024**2 and total+before.st_size<=64*1024**2)
        raw=p.read_bytes()
        after=p.stat()
        need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns))
        digest=hashlib.sha256(raw).hexdigest()
        need(expected is None or digest==expected)
        total+=len(raw)
        pins[name]=dict(bytes=len(raw),SHA256=digest)
        return json.loads(raw)
    intent=load("intent.json")
    need(intent["request_id"]==context.request==REQUEST and intent["publication"]==context.publication
         and intent["epoch"]==context.epoch and intent["deadline"]==context.epoch+2400
         and intent["source_manifest_SHA256"]==context.source_manifest_sha256
         and intent["claim_SHA256"]==context.claim_sha256)
    source=load("src/SOURCE_FILES.json",context.source_manifest_sha256)
    claim=load("src/CLAIM.json",context.claim_sha256)
    need(claim["request"]==REQUEST and claim["training_GO"] is True)
    # Source closure only: these files are small code/manifests, never model data.
    for name,pin in source["files"].items():
        p=ROOT/"src"/name
        raw=p.read_bytes()
        need(len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["SHA256"])
        total+=len(raw)
        need(total<=64*1024**2)
        pins["src/"+name]=dict(bytes=len(raw),SHA256=pin["SHA256"])
    receiver=load("receiver-outcome.json")
    controller=load("controller-outcome.json")
    need(receiver["request"]==REQUEST and receiver["controller_rc"]==0
         and receiver["normal_held_controller_reaped"] is True and receiver["passed"] is True
         and controller["passed"] is True and controller["failure"] is None)
    for phase in ("pre","gpu","post"):
        entry=controller["stages"][phase]
        row=entry["row"]
        need(entry["passed"] is True and entry["terminal_visible"] is True and entry["rc"]==0
             and row[1]==LABEL+"-"+phase and row[2:4]==["COMPLETED","0:0"] and row[5]=="f13r1n10")
        tres=dict(x.split("=",1) for x in row[6].split(","))
        need(tres.get("node")=="1" and {k:v for k,v in tres.items() if k.startswith("gres/")}==({"gres/dcu":"8"} if phase=="gpu" else {}))
    before,after=load("before.json"),load("after.json")
    need(before["pci"]==after["pci"] and len(before["pci"])==8 and before["boot_id"]==after["boot_id"]
         and after["kfd_holder_count"]==0 and after["full_fd_inventory_performed"] is True
         and after["owned_remaining"]==0 and after["port_available"] is True)
    view=load("view-receipt.json")
    ledger=load("view/exposure-ledger-private.json",view["ledger"]["SHA256"])
    order=[i%88 for i in range(800)]
    need(view["original_unique_rows"]==88 and view["physical_rows"]==800
         and ledger["original_index_by_physical_row"]==order and ledger["view_SHA256"]==view["view"]["SHA256"]
         and ledger["exposure_counts"]==[order.count(i) for i in range(88)] and ledger["dev_or_blind_read"] is False)
    max_gap=0.
    rank_summaries=[]
    supervised=0
    token_coverage=True
    for rank in range(8):
        def record(arm,name):
            return load(f"{arm}/train/treasury-witness/rank-{rank:02d}/{name}")
        c,r=record("continuous","result.json"),record("resumed","result.json")
        need(all(x["rank"]==rank and x["world_size"]==8 and x["completed_global_step"]==25 for x in (c,r)))
        need(len(c["inputs"])==len(c["outputs"])==100 and len(r["inputs"])==len(r["outputs"])==20
             and [x["global_step"] for x in c["steps"]]==list(range(1,26))
             and [x["global_step"] for x in r["steps"]]==list(range(21,26))
             and c["saved_checkpoints"]==[10,20,25] and r["saved_checkpoints"]==[25])
        need(all(x["optimizer_slot_count"]==504 and x["optimizer_step_min"]==x["optimizer_step_max"]==x["global_step"] for x in c["steps"]))
        need(all(type(x["adapter_changed"]) is bool for x in c["steps"]) and any(x["adapter_changed"] for x in c["steps"]))
        need(record("resumed","native-data-skip.json")==dict(rank=rank,before=100,skipped=80,after=20))
        restored=record("resumed","restored-before-RNG.json")
        saved_rng=record("continuous","checkpoint-20-rng.json")
        need(restored["restored_parameter_optimizer_scheduler_equal"] is True
             and restored["actual"]==c["steps"][19]["state"] and len(r["RNG_loads"])==1
             and r["RNG_loads"][0]["saved_equal"] is True and r["RNG_loads"][0]["actual"]==saved_rng["RNG"])
        need(record("continuous","checkpoint-25-rng.json")==record("resumed","checkpoint-25-rng.json"))
        for i,(a,b) in enumerate(zip(c["inputs"][80:],r["inputs"],strict=True)):
            need(a["ordinal"]==i+81 and b["ordinal"]==i+1 and a["global_step"]==b["global_step"]==20+i//4
                 and a["full_input"]==b["full_input"])
        for a,b in zip(c["outputs"][80:],r["outputs"],strict=True):
            gap=abs(a["loss_scalar"]-b["loss_scalar"])
            need(math.isfinite(gap) and gap==0 and a["loss"]==b["loss"])
            max_gap=max(max_gap,gap)
        need(all(a["state"]==b["state"] for a,b in zip(c["steps"][20:],r["steps"],strict=True)))
        for row in c["inputs"]:
            value=row.get("tokens",{}).get("shifted_supervised_tokens")
            if type(value) is int and value>=0:
                supervised+=value
            else:
                token_coverage=False
        rank_summaries.append(dict(rank=rank,continuous_updates=25,resumed_updates=5,micros=120,paired_tail_equal=True))
    for arm,steps in (("continuous",(10,20,25)),("resumed",(25,))):
        native=load(arm+"/native-outcome.json")
        need(native["passed"] is True and native["rc"]==0 and len(native["finished"])==8
             and native["cleanup"]["owned_processes_remaining"]==0 and not native["cleanup"]["termination_needed"])
        for entry in native["finished"]:
            relative=Path(entry["path"]).relative_to(ROOT).as_posix()
            need(pins[relative]==dict(bytes=entry["bytes"],SHA256=entry["SHA256"]))
        for n in steps:
            inventory=load(arm+f"-checkpoint-{n}-files.json")
            need(inventory["member_count"]==14 and len(inventory["files"])==14 and inventory["global_step"]==n)
    proof=load("paired-recovery.json")
    need(proof["exact_paired_recovery"] is True and proof["paired_remaining_microbatches"]==160)
    return dict(state="QWEN_USGS25_AND_RESUME20_INDEPENDENT_METADATA_PASS",passed=True,
         request=REQUEST,publication=context.publication,source_manifest_SHA256=context.source_manifest_sha256,
         actual_global_optimizer_invocations=30,ranks=rank_summaries,unique_train_examples=88,
         continuous_exposure_rows=800,paired_microbatches=160,maximum_absolute_loss_gap=max_gap,
         actual_continuous_shifted_supervised_tokens=supervised if token_coverage else None,actual_token_coverage_complete=token_coverage,
         actual_GPU_cards=8,actual_read_bytes=total,actual_metadata_files=len(pins),wall_seconds=time.monotonic()-started,
         metadata_manifest_SHA256=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
         new_model_or_processor_invocations=0,checkpoint_tensor_body_independently_read=False,
         dev_or_blind_read=False,formal_acceptance=False,effect_qualified=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--request",required=True);p.add_argument("--publication",required=True)
    p.add_argument("--epoch",type=float,required=True);p.add_argument("--source-manifest-sha256",required=True)
    p.add_argument("--claim-sha256",required=True)
    args=p.parse_args()
    def expired(*_):
        raise TimeoutError("Independent metadata-reader budget exhausted")
    signal.signal(signal.SIGALRM,expired);signal.alarm(45)
    result=consume(args)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
