"""Consume only the original failed dev3 v1 steps and independent post records."""
import hashlib,json
from pathlib import Path
import subprocess
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
R=ROOT/"training-qwen-treasury-explicit-dev3-gpu-v1"
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
SPEC={"base":("1087656","a01r4n14","331","333","335"),"old_domain":("1087656","a01r4n16","332","334","336"),"domain":("1103606","f13r1n10","164","165","166")}
pins={}
def read_raw(p,limit=1048576):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size<=limit
    b=p.read_bytes();pins[str(p)]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(read_raw(p))
intent=read(R/"intent.json");outcome=read(R/"outcome.json")
assert intent["request"]==outcome["request"]=="58edc167cb6b4de9b665d185f7173bd1"
assert intent["epoch"]==1794883544.702926 and intent["deadline"]==intent["epoch"]+1800
assert intent["publication"]=="4ab3b536eefb09f4ab02f8232a25ebdc946b5d96"
assert intent["all_arm_denominator"]==324 and len(set(intent["planned_ids"]))==108
assert outcome["passed"] is False
claim=read(R/"source/CLAIM.json");mf=read(R/"source/SOURCE_FILES.json")
assert pins[str(R/"source/CLAIM.json")]=="5b1d6b924ed634358f724e2a9a9766af6a75604d5dc517f12ad1cac8c5a20bb4"
assert pins[str(R/"source/SOURCE_FILES.json")]=="8918cc1b8b275a3ec63981adf5a60782dae1eed7d91f3a5fd65d0a8937d56a21"
assert len(mf["files"])==20
for n,pin in mf["files"].items():
    assert Path(n).name==n;b=read_raw(R/"source"/n)
    assert len(b)==pin["bytes"] and hashlib.sha256(b).hexdigest()==pin["SHA256"]
rows=[]
for parent in ("1087656","1103606"):
    text=subprocess.check_output(["sacct","-n","-P","-j",parent,"-o",COLS],text=True,timeout=9)
    rows += [x.split("|") for x in text.splitlines() if "training-qwen-treasury-explicit-dev3-gpu-v1-" in x]
assert len(rows)==9
arms={};leased=set()
assets=read(R/"source/MODEL_ASSETS.json")
for arm,(parent,node,pre,gpu,post) in SPEC.items():
    a=R/"arms"/arm;actual={}
    for phase,number,terminal in (("pre",pre,["COMPLETED","0:0"]),("gpu",gpu,["FAILED","1:0"]),("post",post,["COMPLETED","0:0"])):
        matched=[r for r in rows if r[0]==parent+"."+number]
        assert len(matched)==1 and matched[0][2:4]==terminal and matched[0][5]==node
        record=read(a/(phase+"-accounting.json"));assert record["row"]==matched[0] and record["terminal_visible"]
        assert record["passed"] is (phase!="gpu")
        actual[phase]=matched[0]
    before=read(a/"before.json");after=read(a/"after.json");assigned=read(a/"assigned.json")
    assert before==read(a/"pre.log") and after==read(a/"post.log")
    assert before["node"]==after["node"]==node and before["boot_id"]==after["boot_id"]==assigned["boot_id"]
    assert len(before["pci"])==len(set(before["pci"]))==8 and before["pci"]==after["pci"]
    assert after["kfd_holder_count"]==0 and after["full_fd_inventory_performed"] is True
    assert assigned["visible_pci"] in before["pci"] and type(assigned["worker_pid"]) is int and type(assigned["worker_startticks"]) is int
    leased.add((node,assigned["visible_pci"]))
    c=read(a/"controller-outcome.json");w=read(a/"worker-failure.json");provider=read(a/"provider.json")
    assert c["passed"] is False and c["generation_verified"] is False and w["error_class"]=="ValueError"
    assert w["planned_rows"]==108 and w["actual_training_updates"]==0
    log=read_raw(a/"gpu.log").decode()
    assert "ValueError: development input package differs" in log
    assert "bound_inputs" in log and "native_generation.py" in log
    assert not (a/"outputs").exists()
    arms[arm]=dict(actual_accounting=actual,planned_rows=108,generated_rows=0,missing_rows=108,
        per_topic_planned=36,error="development input package differs",independent_post_verified=True,
        all_PCI_and_boot_match=True,KFD_holders=0,original_post_source_checks_worker_PID_and_startticks=True,
        worker_PID=assigned["worker_pid"],worker_startticks=assigned["worker_startticks"],
        model_validation_seconds=provider["model_validation_seconds"],
        model_assets_hashed_bytes=sum(x["bytes"] for x in assets["model"].values()),
        model_construction_not_reached_by_source_trace=True,actual_training_updates=0)
assert len(leased)==3
print(json.dumps(dict(state="ORIGINAL_THREE_GPU_FAILED_INPUT_GATE_ZERO_PREDICTIONS_POSTS_VERIFIED",
    request=intent["request"],source_publication=intent["publication"],epoch=intent["epoch"],deadline=intent["deadline"],
    arms=arms,all_arm_denominator=324,missing_predictions=324,
    measured_Slurm_DCU_seconds=sum(int(a["actual_accounting"]["gpu"][4]) for a in arms.values()),
    formal_quality=False,formal_performance=False,effective_busy_qualified=False,
    source_inventory_verified=True,original_artifact_pins=pins,
    boundary="Source trace and absent output directories prove generation/model construction were not reached; worker HIP/Torch initialization and full model-asset hashing did execute. Failed collector statuses remain FAIL.")))
