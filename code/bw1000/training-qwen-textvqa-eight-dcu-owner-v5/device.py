"""Reuse the actual .220 HIP gate and .1231 physical snapshot before stock Swift."""

import argparse
import ctypes
from hashlib import sha256
from io import BytesIO
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from contract import (FRAMEWORK, IMPORT_CPUS, IMPORT_MODULE_NAMES, NATIVE, OLD_NATIVE_SHA, OLD_SCOPE_SHA, OLD_SNAPSHOT_SHA, REQUEST,
                      SHA64, STAGE, cache_prefix, check_compute_cache, compute_environment, digest, encode,
                      framework_observation, import_scope, need, original_evidence, save, small, stream_sha256)


def snapshot_module(root, intent):
    source=root/"src/snapshot1231.py"
    raw=small(source,OLD_SNAPSHOT_SHA)
    namespace={"__file__":str(source),"__name__":"_verified_snapshot1231"}
    exec(compile(raw,str(source),"exec"),namespace)
    namespace.update(NODE=intent["node"],STEP=re.compile(re.escape(intent["parent"])+r"\.[0-9]+"))
    return namespace["snapshot"]


def scope_module(root):
    source=root/"src/scope1231.py"
    namespace={"__file__":str(source),"__name__":"_verified_scope1231"}
    exec(compile(small(source,OLD_SCOPE_SHA),str(source),"exec"),namespace)
    return namespace


def hip_gate(root):
    source=root/"src/native220.py"
    raw=small(source,OLD_NATIVE_SHA)
    first=b" assert socket.gethostname().split('.')[0]==NODE"
    last=b" framework=FRAMEWORK"
    need(raw.count(first)==raw.count(last)==1,"EXACT_ORIGINAL_220_GPU_GATE_ANCHORS")
    body=raw[raw.index(first):raw.index(last)]
    need(b"hip.hipGetDeviceCount" in body and b"hip.hipDeviceGetPCIBusId" in body,
         "EXACT_ORIGINAL_HIP_ASSIGNMENT_CALLS")
    program=b"def gate(ROOT,NODE,PARENT,PORT,device_snapshot):\n"+body+b" return leased,cards\n"
    namespace={name:globals()[name] for name in ("ctypes","json","os","re","socket","time")}
    exec(compile(program,"<frozen .220 physical assignment>","exec"),namespace)
    return namespace["gate"]


def port_clean(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(("127.0.0.1",port))


def scoped_processes(root):
    callback=b"qwen_street_scene_first_two"
    target=os.fsencode(root/"train")
    direct={}
    marker=root/"child-owner.json"
    if marker.is_file():
        item=json.loads(small(marker,limit=4096))
        identity=item["identity"]
        if identity is not None:
            direct[int(item["PID"])]=identity["startticks"]
    visible=[]
    manager=root/"src/owned220.py"
    namespace={"__file__":str(manager),"__name__":"_verified_owned220"}
    exec(compile(small(manager,"63995f2a2a3a6edaa29013fe1942114e0279d9cf48e7a9051515d5158b8552aa"),
                 str(manager),"exec"),namespace)
    for pid, row in namespace["snapshot"]().items():
        argv=row["args"]
        if row["state"]=="Z":
            continue
        if direct.get(pid)==row["startticks"]:
            visible.append(pid)
        elif b"--output_dir" in argv and callback in argv:
            at=argv.index(b"--output_dir")+1
            if at<len(argv) and argv[at]==target:
                visible.append(pid)
    return sorted(set(visible))


def observe(root,intent,phase,snapshot=None,check_port=port_clean,processes=scoped_processes):
    sampler=snapshot or snapshot_module(root,intent)
    value=sampler(False)
    check_port(intent["port"])
    value.update(master_port=intent["port"],master_port_available=True,phase=phase,request=intent["request"])
    if phase=="post":
        value["owned_processes_remaining"]=processes(root)
        value["independent_owned_process_scan"]=True
    return value


def runtime(root):
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, RandomSampler
    import importlib.metadata as meta
    import qwen_vl_utils
    need(str(torch.__version__)=="2.10.0" and torch.version.hip=="6.3.26113"
         and Path(torch.__file__).resolve()==Path("/usr/local/lib/python3.10/dist-packages/torch/__init__.py")
         and meta.version("ms-swift")=="4.4.2" and meta.version("transformers")=="5.12.1"
         and meta.version("peft")=="0.19.1" and meta.version("qwen-vl-utils")=="0.0.14"
         and Path(qwen_vl_utils.__file__).resolve()==Path("/vision/qwen_vl_utils/__init__.py"),
         "EXACT_ORIGINAL_220_PASSIVE_RUNTIME")
    source={}
    for cls in (BatchSamplerShard,BatchSampler,RandomSampler):
        source[cls.__module__+"."+cls.__name__]=sha256(small(Path(inspect.getfile(cls)),limit=1024**2)).hexdigest()
    need(source==original_evidence(root/"src"),"EXACT_1252_LOADED_SAMPLER_PROVIDERS")
    return {"torch":str(torch.__version__),"hip":torch.version.hip,"sampler_source_SHA256":source}


def trainer_argv(root,intent,assignment_sha):
    src=root/"src"
    return [str(src/"trainer/native_entry.py"),"--request",intent["request"],"--parent",intent["parent"],
            "--node",intent["node"],"--output",str(root),"--assignment",str(root/"assigned.json"),
            "--assignment-sha",assignment_sha,"--framework",str(FRAMEWORK),
            "--original",str(STAGE/"original-launch.json"),"--binding",str(STAGE/"QWEN_TARGET_NATIVE_BINDING.json"),
            "--recipe",str(STAGE/"recipe.py"),"--owned",str(src/"owned220.py"),
            "--public-native",str(src/"native1251.json"),"--terminal220",str(src/"terminal220.json"),
            "--assets",str(src/"assets220.json"),"--port",str(intent["port"]),"--epoch",str(intent["epoch"]),
            "--deadline",str(intent["epoch"]+NATIVE),"--maximum",str(NATIVE)]


def exec_trainer(root,intent,assignment_sha,execute=os.execve):
    argv=trainer_argv(root,intent,assignment_sha)
    execute(sys.executable,[sys.executable,"-B","-u",*argv],compute_environment(os.environ))


def passive_imports(root):
    src=root/"src"
    sys.path.insert(0,str(src/"trainer"))
    torch_root=Path("/usr/local/lib/python3.10/dist-packages/torch")
    paths={
        "torch":torch_root,
        "torch.distributed.rpc":torch_root,
        "torch.distributed.nn.jit.instantiator":torch_root,
        "swift.callbacks":Path("/swift-runtime/site/swift"),
        "contract":src,"device":src,"evidence":src/"trainer","prepare":src/"trainer",
        "native_entry":src/"trainer","street_updates":src/"trainer",
        "tools.hygon_vlm_recipe":FRAMEWORK/"tools",
        "vlm.runtime.first_update_capture":FRAMEWORK/"vlm/runtime",
        "vlm.runtime.initial_trainables":FRAMEWORK/"vlm/runtime",
        "vlm.runtime.native_recipe_contract":FRAMEWORK/"vlm/runtime",
        "vlm.runtime.checkpointing_lifecycle":FRAMEWORK/"vlm/runtime",
    }
    need(paths.keys()==IMPORT_MODULE_NAMES,"EXACT_PASSIVE_IMPORT_MODULE_SET")
    actual={}
    for name,expected in paths.items():
        module=importlib.import_module(name)
        path=Path(module.__file__).resolve(strict=True)
        need(path.is_relative_to(expected),"EXACT_IMPORT_MODULE_PATH")
        source=small(path,limit=2*1024**2)
        actual[name]={"path":str(path),"bytes":len(source),"SHA256":digest(source)}
    import torch
    from importlib.util import cache_from_source
    need(str(torch.__version__)=="2.10.0" and torch.version.hip=="6.3.26113"
         and not torch.cuda.is_initialized(),"EXACT_ORIGINAL_TORCH_WITH_CUDA_UNINITIALIZED")
    need(cache_from_source("rpc_module.py").startswith("__pycache__/")
         and stream_sha256(BytesIO(b"qwen Python310"))==digest(b"qwen Python310"),"EXACT_CLEAN_IMPORT_HELPERS")
    return actual


def zero_gres_import(root,probe,parent,node,source_sha,*,observe_framework=framework_observation,observe_imports=passive_imports):
    owner_request=import_scope(root,probe,parent,node)
    check_compute_cache()
    step=os.environ.get("SLURM_STEP_ID","")
    need(sys.version_info[:2]==(3,10) and socket.gethostname().split(".")[0]==node
         and os.environ.get("SLURM_JOB_ID")==parent and step.isdecimal()
         and os.environ.get("SLURM_CPUS_PER_TASK")==str(IMPORT_CPUS) and len(os.sched_getaffinity(0))>=IMPORT_CPUS
         and os.environ.get("SLURM_STEP_GPUS") in (None,"","(null)")
         and os.environ.get("SLURM_GPUS_ON_NODE") in (None,"","(null)","0")
         and all(os.environ.get(name)=="" for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
         and SHA64.fullmatch(source_sha),"EXACT_NUMERIC_ZERO_GRES_FOUR_CPU_IMPORT_RUNTIME")
    small(root/"src/PACKET.json",source_sha)
    report=dict(request=probe,owner_request=owner_request,numeric_step=parent+"."+step,node=node,
                source_manifest_SHA256=source_sha,python="3.10",GPU=0,optimizer_updates=0,
                no_gpu_approval=True,replay_allowed=False)
    save(root/("py310-import-"+probe+"-intent.json"),report)
    before=None
    try:
        before=observe_framework()
        report["framework_before"]=before
        report["module_sources"]=observe_imports(root)
        report["cache_prefix_is_none"]=True
        report["bytecode_writes_disabled"]=True
        report["cuda_initialized"]=False
    except (ValueError,RuntimeError,ImportError,OSError,IndexError,KeyError,TypeError,AttributeError) as err:
        report["import_error"]={"type":type(err).__name__,"detail_SHA256":digest(str(err).encode())}
    try:
        report["framework_after"]=observe_framework()
        report["framework_unchanged"]=before is not None and report["framework_after"]==before
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as err:
        report["framework_after_error"]={"type":type(err).__name__,"detail_SHA256":digest(str(err).encode())}
    passed=bool(report.get("framework_unchanged") and "module_sources" in report and "import_error" not in report)
    report["state"]="EXACT_PYTHON310_ZERO_GRES_IMPORT_PASS" if passed else "PYTHON310_IMPORT_NOT_QUALIFIED"
    data=encode(report)
    need(len(data)<=262144,"BOUNDED_PRIVATE_IMPORT_EVIDENCE")
    save(root/("py310-import-"+probe+".json"),report)
    return {"request":probe,"state":report["state"],"GPU":0,"framework_files":393 if passed else 0,
            "module_count":len(report.get("module_sources",{})),"receipt_SHA256":digest(data),
            "actual_step":report["numeric_step"],"no_gpu_approval":True},int(not passed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("phase",choices=("pre","gpu","post","probe-imports"))
    parser.add_argument("--probe-request")
    parser.add_argument("--parent")
    parser.add_argument("--node")
    parser.add_argument("--source-sha")
    args=parser.parse_args()
    root=args.root
    if args.phase=="probe-imports":
        need(root.resolve(strict=True)==root and REQUEST.fullmatch(args.probe_request or ""),
             "EXACT_TARGET_IMPORT_SOURCE_NAMESPACE")
        result,rc=zero_gres_import(root,args.probe_request,args.parent,args.node,args.source_sha)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
        return rc
    intent=json.loads(small(root/"intent.json"))
    need(root.resolve(strict=True)==root and root.name=="qwen-textvqa-first-two-"+intent["request"],
         "EXACT_NUMERIC_STEP_OWNER_ROOT")
    if args.phase!="gpu":
        need(sys.dont_write_bytecode and sys.pycache_prefix==str(cache_prefix(root)),
             "EXACT_ISOLATED_HOST_NUMERIC_WORKER_PYTHON_CACHE")
        print(json.dumps(observe(root,intent,args.phase),sort_keys=True,allow_nan=False),flush=True)
        return
    check_compute_cache()
    scope=scope_module(root)
    scope["remaining"](intent["epoch"],NATIVE)
    leased,cards=hip_gate(root)(root,intent["node"],intent["parent"],intent["port"],snapshot_module(root,intent))
    step=leased["step"]
    fields=Path("/proc/self/stat").read_text().split(") ",1)[1].split()
    assigned=dict(node=intent["node"],step=step,PID=os.getpid(),startticks=int(fields[19]),
                  PCI_by_index=cards,leased_snapshot=leased,CPU_affinity=sorted(os.sched_getaffinity(0)),
                  SLURM_CPUS_PER_TASK=64)
    assignment=root/"assigned.json"
    save(assignment,assigned)
    save(root/"runtime-provider.json",runtime(root))
    exec_trainer(root,intent,digest(small(assignment,limit=32768)))


if __name__=="__main__":
    raise SystemExit(main())
