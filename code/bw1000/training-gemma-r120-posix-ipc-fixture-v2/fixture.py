"""Run the Gemma IPC custody patch in one zero-GRES DTK step, without training."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


BASE=PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
IMAGE=BASE/"containers/dtk26-sandbox-r1"
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
ARROW="1fe17db3c76d1fac5472b305c538e26cc11e1e995f667bdd6868ac60596dcde1"
POOL="286b1899798e5c3dd01d131ddda151617a63e3b6e24d2470b73514f9dd259bc5"
ARCHIVE="e49db3405b64892b1042e62ab79ceb36e6aebcc593d7d5438e16ee48f393c5be"
CLEANUP="a23efda8ade028db002edeccb6b30159bca0247166b91f7e7bd306991d1ef7c2"
OFF="d131093686c9b1d44bae7d5d6db9f35147d6798dda2f64b9af97928931966786"
LABEL="gemma-r120-posix-ipc-fixture-v2"
SECONDS=120


def need(ok,why):
    if not ok:
        raise ValueError(why)


def encode(obj):
    return (json.dumps(obj,sort_keys=True,allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def read(path,cap):
    need(path.is_file() and not path.is_symlink(),"EXACT_REGULAR_INPUT")
    with path.open("rb") as f:
        raw=f.read(cap+1)
    need(0<len(raw)<=cap,"BOUNDED_READ")
    return raw


def write(path,obj):
    raw=encode(obj)
    with path.open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(bytes=len(raw),SHA256=digest(raw))


def source(stage,wanted):
    raw=read(stage/"SOURCE_FILES.json",8192)
    need(digest(raw)==wanted,"EXACT_STAGED_SOURCE_MANIFEST")
    files=json.loads(raw)["files"]
    need(set(files)=={"fixture.py","short_mount.py","short_lifecycle.py","native_worker_off.py","readback.py","cohort.py","provision.py","PROVIDER_BINDING.json"},
         "EXACT_FIXTURE_SOURCE_MEMBERS")
    members={name:read(stage/name,65536) for name in files}
    need(all(files[n]==dict(bytes=len(b),SHA256=digest(b)) for n,b in members.items())
         and digest(members["native_worker_off.py"])==OFF,"ORIGINAL_OFF_AND_IPC_PATCH_LINEAGE")
    off_tree=ast.parse(members["native_worker_off.py"])
    off=next(n for n in off_tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_plan")
    need(any(isinstance(n,ast.Tuple) and [x.value for x in n.elts if isinstance(x,ast.Constant)]==
             ["--use_logits_to_keep","false"] for n in ast.walk(off)),"OFF_WORKER_ARGV_STATIC_ONLY")
    return members


def mount_modules(members,root,alias,request):
    fake=ModuleType("contract")
    fake.REQUEST,fake.ROOT,fake.need=request,root,need
    fake.save=lambda name,value:write(root/name,value)
    modules={"contract":fake}
    for name in ("short_lifecycle","short_mount"):
        modules[name]=ModuleType(name)
    old = {k:sys.modules.get(k) for k in modules}
    try:
        sys.modules.update(modules)
        for name in ("short_lifecycle","short_mount"):
            exec(compile(members[name+".py"],name+".py","exec"),modules[name].__dict__)
    finally:
        for key,prior in old.items():
            if prior is None: del sys.modules[key]
            else:sys.modules[key]=prior
    # host_create_once imports this module dynamically in the frozen patch.
    sys.modules["short_lifecycle"]=modules["short_lifecycle"]
    modules["short_mount"].SHORT=alias
    return modules["short_mount"],modules["short_lifecycle"]


def scoped(a,request,now):
    need(a["request"]==request and re.fullmatch("[0-9a-f]{32}",request)
         and request not in {"886fb0896c094de9acc384dc0d13ebb2","b298f9ce6be74ce6b9035fc2f61d7f2e","35345a3f196b1d04ec1c4157e9d1c0ce"}
         and a["label"]==LABEL and a["zero_GRES_GO"] is a["root_authorized"] is True
         and a["GPU_GO"] is a["training_GO"] is a["weights_GO"] is False
         and a["parent"]=="1103606" and a["node"]=="f13r1n12"
         and type(a["epoch"]) in (int,float) and type(a["deadline"]) in (int,float)
         and a["epoch"]<=now<a["deadline"]<=a["epoch"]+SECONDS
         and a["original_4177_archive_SHA256"]==ARCHIVE,"EXACT_FRESH_POSIX_ZERO_GRES_PERMISSION")


def step(a,*,guest):
    env=os.environ
    number=env.get("SLURM_STEP_ID","")
    need(socket.gethostname().split(".")[0]==a["node"] and env.get("SLURM_JOB_ID")==a["parent"]
         and re.fullmatch("[0-9]+",number) and env.get("SLURM_CPUS_PER_TASK")=="4"
         and len(os.sched_getaffinity(0))>=4
         and env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and not re.search("dcu|gpu",env.get("SLURM_TRES_PER_TASK",""),re.I)
         and all(env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK")),
         "EXACT_ZERO_GRES_STEP_CONTEXT")
    if guest:
        need(all(env.get(k)=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")),
             "GUEST_GPU_VISIBILITY_MASKS")
    return a["parent"]+"."+number


def apptainer_args(root,alias,manifest,authority,auth_sha,request):
    stage=root/"src"
    results=BASE/"results"
    swift=results/"training-vlm-swift-offline-runtime-v1"
    transformers=results/"training-minicpm-hygon-processor-r2/cpu/runtime-site"
    env=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; unset PYTHONPYCACHEPREFIX; "
         "export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; "
         "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
         "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; "
         "export HOME="+shlex.quote(str(root/"home"))+" XDG_CACHE_HOME="+shlex.quote(str(root/"cache"))+
         " XDG_CONFIG_HOME="+shlex.quote(str(root/"config"))+" TMPDIR="+shlex.quote(str(alias))+
         " TMP="+shlex.quote(str(alias))+" TEMP="+shlex.quote(str(alias))+"; "
         "export PYTHONPATH=/minicpm-transformers:/swift-runtime/site; exec python3 -B -u \"$@\"")
    argv=[APPTAINER,"exec","--no-home","--no-mount","cwd","--pwd","/"]
    for src,dst,mode in ((root,root,"rw"),(root/"short-ipc",alias,"rw"),(stage,stage,"ro"),
                         (swift,PurePosixPath("/swift-runtime"),"ro"),(transformers,PurePosixPath("/minicpm-transformers"),"ro")):
        argv+=["--bind",f"{src}:{dst}:{mode}"]
    return argv+[str(IMAGE),"/bin/bash","--noprofile","--norc","-c",env,LABEL,str(stage/"fixture.py"),"guest",
                "--operation",str(root),"--alias",str(alias),"--request",request,"--manifest-sha256",manifest,
                "--authority",str(authority),"--authority-sha256",auth_sha]


def child_numbers():
    p=Path("/proc/self/task")/str(os.getpid())/"children"
    return {int(x) for x in p.read_text().split()}


def stamp(pid):
    return int((Path("/proc")/str(pid)/"stat").read_text().rsplit(") ",1)[1].split()[19])


def watched_boot():
    return digest(Path("/proc/sys/kernel/random/boot_id").read_bytes())


def fixture_transform(example):
    return {"converted":example["ordinal"]+10}


def guest(args,a,root,alias,members):
    identity=step(a,guest=True)
    mount,life=mount_modules(members,root,alias,args.request)
    info=dict(request=args.request,step=identity,GPU=0,actual_training_rows=0,weight_payloads=0,boot_SHA256=watched_boot(),
              planned_dataset_map_rows=2,actual_dataset_map_rows=0,manager_queue_roundtrip=False,
              original_4177_cleanup="REFUSED",original_4177_cleanup_SHA256=CLEANUP,passed=False)
    observer=life.Lifecycle(alias,os.getuid(),archive_path=root/"short-private-tmp-events.jsonl")
    retirement={}
    import cohort
    leases=cohort.Leases(alias,args.request)
    manager_before=set()
    completed=False
    try:
        mount.verify_inside(check_temp_environment=True)
        observer.sample("BEFORE_NATIVE")
        creation=json.loads(read(root/"short-private-tmp-creation.json",4096))
        info["mount"]=life.compare_mount(creation,life.mount_witness(root/"short-ipc",args.request),
                                        life.mount_witness(alias,args.request))
        for name,pin in (("arrow_dataset.py",ARROW),("utils/py_utils.py",POOL)):
            need(digest(read(Path("/swift-runtime/site/datasets")/name,512<<10))==pin,"EXACT_DEPLOYED_ARROW_SOURCE")
        def bound_audit(event,vals):
            if event in {"socket.getaddrinfo","subprocess.Popen","os.system"}:
                raise RuntimeError("NO_NETWORK_OR_EXTERNAL_CHILD")
            if event in {"socket.bind","socket.connect"}:
                s,address=vals
                need(s.family==socket.AF_UNIX and isinstance(address,str) and "\0" not in address
                     and Path(address).is_absolute() and Path(address).parent.resolve(strict=True).is_relative_to(alias),
                     "PRIVATE_FILESYSTEM_AF_UNIX_ONLY")
            if event=="open" and vals and isinstance(vals[0],(str,bytes)):
                p=Path(os.fsdecode(vals[0]))
                need(p.suffix not in {".pt",".bin",".safetensors",".pth"}
                     and not (p.is_relative_to(BASE) and {"dev","blind"}&set(p.parts)),
                     "NO_WEIGHTS_OR_HELDOUT_INPUTS")
        sys.addaudithook(bound_audit)
        import datasets.arrow_dataset as arrow
        import datasets.utils.py_utils as pool
        import multiprocess as provider
        need(Path(arrow.__file__)==Path("/swift-runtime/site/datasets/arrow_dataset.py")
             and Path(pool.__file__)==Path("/swift-runtime/site/datasets/utils/py_utils.py")
             and (getattr(arrow,"Pool",None) is provider.Pool or any(x is provider for x in vars(arrow).values())),
             "ACTUAL_ARROW_DATASET_MANAGER_PROVIDER")
        info["deployed_provider_sources"]=leases.attach(json.loads(members["PROVIDER_BINDING.json"]))
        info["registered_creation"]=leases.creation()
        write(root/"registered-ipc-creation.json",dict(request=args.request,creation=info["registered_creation"]))
        source=arrow.Dataset.from_dict({"ordinal":[0,1]})
        mapped=source.map(fixture_transform,num_proc=1,keep_in_memory=True,load_from_cache_file=False)
        need(list(mapped["converted"])==[10,11],"REAL_DATASET_MAP_TWO_ROWS")
        info["actual_dataset_map_rows"]=2
        del mapped,source
        manager_before=child_numbers()
        info["direct_children_before_manager"]=len(manager_before)
        info["managers"]={}
        for name in leases.order:
            cohort.manager(leases.items[name],observer,info["managers"],name,stamp)
        info["manager_queue_roundtrip"]=info["managers"]["multiprocess"]["queue_roundtrip"]
        completed=not manager_before and not child_numbers()
        need(completed,"ACTUAL_MANAGER_AND_CHILDREN_REAPED")
    except Exception as e:
        info["failure"]=safe_error(e)
    finally:
        observer.sample("PRE_RETIREMENT")
        try:
            leases.retire(children_reaped=completed)
            info["registered_cleanup_passed"]=True
        except Exception as e:
            info["registered_cleanup_failure"]=safe_error(e)
        finally:
            info["registered_cleanup"]=leases.results()
            try:
                leases.close()
            except Exception as e:
                info["registered_close_failure"]=safe_error(e)
        try:
            creation=json.loads(read(root/"short-private-tmp-creation.json",4096))
            life.retire_natural(root/"short-ipc",alias,creation,args.request,initially_marker_only=True,
                                children_reaped=completed,report=retirement)
            mount.verify_inside(check_temp_environment=False)
        except Exception as e:
            retirement["state"]="REFUSED"
            retirement["refusal"]=safe_error(e)
        observer.sample("POST_RETIREMENT")
        write(root/"short-private-tmp-cleanup.json",dict(request_id=args.request,retirement=retirement))
        status=observer.evidence()
        witness=write(root/"short-private-tmp-lifecycle.json",dict(request_id=args.request,**status))
        info["lifecycle"]=witness
        info["passed"]=(info.get("failure") is None and completed and retirement["state"]=="MARKER_ONLY"
                         and info.get("registered_cleanup_passed") is True and "registered_close_failure" not in info
                         and not status["errors"] and not status["truncated_events"])
        write(root/"guest.json",info)
    return int(not info["passed"])


def safe_error(error):
    label=str(error) if type(error) is ValueError else ""
    return dict(type=type(error).__name__,code=label if re.fullmatch("[A-Z0-9_]{1,90}",label) else None)


def host(args,a,root,alias,members):
    step_id=step(a,guest=False)
    mount,life=mount_modules(members,root,alias,args.request)
    output=dict(request=args.request,step=step_id,GPU=0,model_forwards=0,optimizer_updates=0,passed=False,
                carrier_attempted=False,authority_SHA256=args.authority_sha256,source_manifest_SHA256=args.manifest_sha256)
    boot=watched_boot()
    try:
        for name in ("home","cache","config"):
            (root/name).mkdir(mode=0o700)
        # Same patched production creation function, in the real compute-host process.
        mount.host_create_once()
        original=json.loads(read(root/"short-private-tmp-creation.json",4096))
        argv=apptainer_args(root,alias,args.manifest_sha256,args.authority,args.authority_sha256,args.request)
        write(root/"intent.json",dict(request=args.request,step=step_id,epoch=a["epoch"],deadline=a["deadline"],
                                      unique_no_retry=True,container_argv_SHA256=digest(encode(argv))))
        # Full raw interpreter stdout stays inside this private one-use operation.
        with (root/"container.stdout").open("xb") as stdout,(root/"container.stderr").open("xb") as stderr:
            output["carrier_attempted"]=True
            returned=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                    timeout=max(.1,a["deadline"]-time.time()),check=False).returncode
        output["container_rc"]=returned
        raw=read(root/"guest.json",16384)
        got=json.loads(raw)
        output["guest_SHA256"]=digest(raw)
        output["guest_passed"]=got["passed"]
        need(got["request"]==args.request and got["step"]==step_id and got["boot_SHA256"]==boot==watched_boot(),
             "EXACT_CONTAINER_SAME_STEP_BOOT")
        current=life.mount_witness(root/"short-ipc",args.request)
        life.compare_mount(original,current,current)
        output["host_marker_only_after"]=not life.residuals(root/"short-ipc",os.getuid())
        output["host_managers_absent"]={}
        for name,row in got.get("managers",{}).items():
            if "pid" not in row:
                continue
            try:
                live=stamp(row["pid"])==row["startticks"]
            except FileNotFoundError:
                live=False
            output["host_managers_absent"][name]=not live
        need(returned==0 and got["passed"] is True and output["host_marker_only_after"]
             and output["host_managers_absent"]=={"multiprocessing":True,"multiprocess":True}
             and not child_numbers(),"GUEST_AND_HOST_TERMINAL_CUSTODY")
        output["passed"]=True
    except Exception as e:
        output["failure"]=safe_error(e)
    write(root/"host.json",output)
    print(json.dumps(output,sort_keys=True))
    return int(not output["passed"])


def claim_host(root,args):
    need({x.name for x in root.iterdir()}=={"approval.json","src"},"SCOPE_HAD_PRIOR_MUTATION_OR_ATTEMPT")
    write(root/"owner-gate.json",dict(request=args.request,authority_SHA256=args.authority_sha256,
                                     source_manifest_SHA256=args.manifest_sha256,
                                     owner_pid=os.getpid(),owner_startticks=stamp(os.getpid())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("host","guest"))
    for name in ("operation","alias","authority"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("request","manifest-sha256","authority-sha256"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    root=args.operation
    alias=args.alias
    need(root.is_absolute() and root.resolve(strict=True)==root and root.name==args.request
         and root.stat().st_uid==os.getuid() and stat.S_IMODE(root.stat().st_mode)==0o700
         and re.fullmatch(r"/tmp/gr[0-9a-f]{8}",str(alias)) and args.authority==root/"approval.json",
         "EXCLUSIVE_SCOPE_ROOT_AND_SHORT_ALIAS")
    raw=read(args.authority,8192)
    need(digest(raw)==args.authority_sha256,"EXACT_ROOT_APPROVAL")
    a=json.loads(raw)
    scoped(a,args.request,time.time())
    need(a["source_manifest_SHA256"]==args.manifest_sha256 and a["short_alias"]==str(alias),"EXACT_SOURCE_AND_ALIAS_AUTHORITY")
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("IPC_DEADLINE")))
    signal.setitimer(signal.ITIMER_REAL,max(.001,a["deadline"]-time.time()))
    members=source(root/"src",args.manifest_sha256)
    if args.mode=="host":
        claim_host(root,args)
        return host(args,a,root,alias,members)
    return guest(args,a,root,alias,members)


if __name__=="__main__":
    raise SystemExit(main())
