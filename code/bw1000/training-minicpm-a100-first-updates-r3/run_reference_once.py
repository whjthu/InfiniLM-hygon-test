"""Run one exact eight-card original MiniCPM reference with an owned finite supervisor."""
import hashlib,importlib.metadata as metadata,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
RUNTIME_PREFIX=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
from deadline_contract import schedule,TERM_GRACE_SECONDS
from owned_processes import snapshot,refresh,cleanup
ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/"src"
def file_sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):h.update(data)
    return h.hexdigest()
def record(name,obj):
    with (ROOT/name).open("x") as f:json.dump(obj,f,indent=2)
def query(args):return subprocess.check_output(["nvidia-smi",*args],text=True,timeout=15).strip().splitlines()
def admission(binding):
    cards=query(["--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,driver_version","--format=csv,noheader,nounits"])
    rows=[[x.strip() for x in line.split(",")] for line in cards]
    apps=query(["--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader,nounits"])
    assert len(rows)==8 and {int(x[0]):x[1] for x in rows}=={int(k):v for k,v in binding["UUID_by_index"].items()}
    good=not apps and all(int(x[5])<=16 and int(x[6])==0 for x in rows)
    return {"cards":rows,"compute_apps":apps,"all_idle":good,"observed_epoch":time.time()}
def build(launch):
    argv=launch["command"].copy();env=launch["environment"].copy()
    assert argv[0]=="/home/whj/nvidia-baseline/env-vlm-fast/bin/swift" and argv[1]=="sft"
    assert env["NPROC_PER_NODE"]=="8" and env["CUDA_VISIBLE_DEVICES"]=="0,1,2,3,4,5,6,7"
    argv[0]="/home/whj/nvidia-baseline/env-vlm/bin/swift"
    i=argv.index("--output_dir")+1;argv[i]=str(ROOT/"train")
    i=argv.index("--external_plugins")+1
    while i<len(argv) and not argv[i].startswith("--"):i+=1
    argv.insert(i,str(SRC/"capture_first_updates.py"))
    i=argv.index("--callbacks")+1
    while i<len(argv) and not argv[i].startswith("--"):i+=1
    argv.insert(i,"original_first_updates")
    assert env["PATH"].startswith("/home/whj/nvidia-baseline/env-vlm-fast/bin:")
    env["PATH"]=env["PATH"].replace("/home/whj/nvidia-baseline/env-vlm-fast/bin:","/home/whj/nvidia-baseline/env-vlm/bin:",1)
    env["PATH"]=str(RUNTIME_PREFIX/"bin")+os.pathsep+env["PATH"]
    env["PYTHONPATH"]=str(RUNTIME_PREFIX/"triton-v1/site")+os.pathsep+str(RUNTIME_PREFIX/"site")
    env.update(OUTPUT_DIR=str(ROOT/"train"),MASTER_PORT="29873",FIRST_UPDATE_CAPTURE=str(ROOT/"capture"))
    return argv,env
def main():
    installed=json.loads((ROOT/"installed.json").read_bytes());claim=installed["claim_commit"]
    assert not (ROOT/"gpu-intent.json").exists(),"Reconcile the original run; never replay launch"
    for name,pin in installed["files"].items():assert file_sha(SRC/name)==pin,name
    raw=(SRC/"original-launch.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()=="015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789"
    launch=json.loads(raw);argv,overrides=build(launch)
    proof=json.loads((SRC/"SOURCE_PROOF.json").read_bytes());binding=json.loads((SRC/"GPU_BINDING.json").read_bytes())
    first=admission(binding)
    if not first["all_idle"]:
        result={"state":"PRELAUNCH_GPU_OCCUPIED","GPU_launched":False,"claim_commit":claim,"admission":first}
        record("prelaunch-result.json",result);print(json.dumps(result));return
    runtime=json.loads((SRC/"RUNTIME_BINDING.json").read_bytes())
    assert Path(runtime["prefix"])==RUNTIME_PREFIX
    assert file_sha(RUNTIME_PREFIX/"direct-v2/receipt.json")==runtime["metadata_overlay_receipt_SHA256"]
    assert file_sha(RUNTIME_PREFIX/"triton-v1/receipt.json")==runtime["triton_receipt_SHA256"]
    for filename,bound in runtime["plans"].items():
        local=SRC/filename
        assert file_sha(local)==bound["SHA256"]
        prefix=RUNTIME_PREFIX/bound["subdirectory"]
        assert file_sha(prefix/"PLAN.json")==bound["SHA256"]
        for name,item in json.loads(local.read_bytes())["files"].items():
            path=prefix/name
            assert path.is_file() and not path.is_symlink() and path.stat().st_size==item["bytes"] and file_sha(path)==item["SHA256"],name
    expected={**proof["pinned_public_Swift_sources"],"transformers/trainer.py":proof["pinned_public_Trainer_source_SHA256"]}
    for role,pin in expected.items():
        dist=metadata.distribution("transformers" if role.startswith("transformers/") else "ms-swift")
        assert file_sha(Path(dist.locate_file(role)))==pin,role
    versions={name:metadata.version(name) for name in ("torch","torchvision","transformers","ms-swift","peft","accelerate","Pillow","tokenizers")}
    assert versions["transformers"]=="5.12.1" and versions["ms-swift"]=="4.4.2"
    assert versions["peft"]=="0.19.1" and versions["accelerate"]=="1.14.0" and versions["tokenizers"]=="0.22.2"
    import torch
    assert str(torch.__version__)=="2.8.0a0+5228986c39.nv25.06" and torch.version.cuda=="12.9"
    assert not torch.cuda.is_initialized()
    files=json.loads((SRC/"MODEL_FILES.json").read_bytes())
    model=Path(files["model_root"])
    for name,row in files["files"].items():
        p=model/name
        assert p.is_file() and p.stat().st_size==row["bytes"] and file_sha(p)==row["SHA256"],name
    data=Path(launch["command"][launch["command"].index("--dataset")+1])
    assert file_sha(data)=="0e476394e3b0117d90cfad8f89ccfafe1800917692e86efed84eaf182c453811"
    baseline=Path("/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py")
    assert file_sha(baseline)=="d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8"
    assert shutil.disk_usage(ROOT).free>=80*1024**3
    assert not any(name in os.environ for name in ("RANK","LOCAL_RANK","WORLD_SIZE","GROUP_RANK"))
    with socket.socket() as check:check.bind(("127.0.0.1",29873))
    time.sleep(5);final=admission(binding)
    if not final["all_idle"]:
        result={"state":"PRELAUNCH_GPU_OCCUPIED","GPU_launched":False,"claim_commit":claim,"admission":final}
        record("prelaunch-result.json",result);print(json.dumps(result));return
    epoch=time.time();clock=schedule(epoch,epoch);mono=time.monotonic()
    environment=os.environ.copy();environment.update(overrides)
    environment["PYTHONPATH"]=str(SRC)+os.pathsep+environment.get("PYTHONPATH","")
    environment.update(FIRST_UPDATE_EPOCH=str(epoch),FIRST_UPDATE_DEADLINE=str(clock["native_deadline"]),FIRST_UPDATE_CAPTURE_CLAIM=claim,PYTHONDONTWRITEBYTECODE="1")
    record("gpu-intent.json",{"claim_commit":claim,**clock,"admission":final,"versions":versions,
            "command":argv,"GPU_cards":8,"original25_preserved":True,"formal_performance":False,"runtime_binding_SHA256":file_sha(SRC/"RUNTIME_BINDING.json")})
    registry={};timed_out=False;failure=None;child=None
    try:
        with (ROOT/"native.log").open("xb") as log:
            child=subprocess.Popen(["/home/whj/nvidia-baseline/env-vlm/bin/python",*argv],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row=snapshot().get(child.pid)
            if row:registry[child.pid]={k:row[k] for k in ("PID","PPID","PGID","startticks","state")}
            record("child-owner.json",{"PID":child.pid,"identity":registry.get(child.pid),"claim_commit":claim})
            while child.poll() is None:
                refresh(registry,ROOT)
                temp=ROOT/"owned-registry.tmp";temp.write_bytes(json.dumps(registry).encode());temp.replace(ROOT/"owned-registry.json")
                if time.monotonic()-mono>=850:timed_out=True;break
                try:child.wait(timeout=min(2,max(.1,850-(time.monotonic()-mono))))
                except subprocess.TimeoutExpired:pass
    except BaseException as error:failure=type(error).__name__
    finally:
        if child is not None and child.poll()==0:
            settle=min(mono+870,time.monotonic()+10)
            while refresh(registry,ROOT) and time.monotonic()<settle:time.sleep(.2)
        cleaned=cleanup(registry,ROOT,TERM_GRACE_SECONDS)
        if child is not None:
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:failure=failure or "OWNED_ROOT_REMAINS"
    try:post=admission(binding)
    except BaseException as error:post={"state":"POST_UNKNOWN","error_type":type(error).__name__,"compute_apps":None}
    actual={"claim_commit":claim,"GPU_launched":child is not None,"state":"NATIVE_RETURNED",
            **clock,"native_returncode":child.returncode if child else None,"timed_out":timed_out,"controller_error":failure,
            "wall_seconds":time.monotonic()-mono,"cleanup":cleaned,"post":post,"owners":registry,
            "native_log_SHA256":file_sha(ROOT/"native.log") if (ROOT/"native.log").is_file() else None,
            "native25_rank_records":sum((ROOT/"train/nvidia-baseline"/("rank-%02d.json"%i)).is_file() for i in range(8)),
            "capture_rank_records":sum((ROOT/"capture"/("rank-%02d"%i)/"receipt.json").is_file() for i in range(8))}
    actual["keeper_passed"]=bool(child and child.returncode==0 and not timed_out and failure is None and not cleaned["termination_needed"] and cleaned["owned_processes_remaining"]==0 and post["compute_apps"]==[])
    record("outcome.json",actual)
    print(json.dumps(actual))
if __name__=="__main__":main()
