"""Diagnose the exact failed Swift 800-row read, with a real SFT data-path boundary."""

import argparse
import ast
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from copy import copy
import errno
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sys
import tempfile
import time

import probe as original
import r30_cache_gate as repair
import r31_modelscope_cache as modelscope_cache
import ipc_witness


SRC=Path("/r34-src")
OUT=Path("/r34-out")
SCRATCH=Path("/r34-scratch")
FRAME=Path("/input-framework")
ENTRY=FRAME/"vlm/runtime/swift_entry.py"
PROCESSOR=Path("/r29-processor")
PROCESSOR_PIN=(2107,"eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b")
ORIGINAL_FAILURE="1103606.3368"
OLD_MANIFEST="45c52a9ea4841097f050c27887a6be8e9b838bda1812c5cb168cc9c0b36ae749"
PACKAGE={'probe.py', 'probe_r34.py', 'r30_cache_gate.py', 'r31_modelscope_cache.py', 'MODEL_PROCESSOR_FILES.json', 'ipc_witness.py'}
need,read,digest,canonical=original.need,original.read,original.digest,original.canonical


def path_class(raw):
    if not isinstance(raw,(str,bytes,os.PathLike)):
        return None
    name=os.fsdecode(raw)
    for prefix,category in (
            (str(original.VIEW),"ORIGINAL_TRAIN800_JSONL"),(str(SCRATCH),"REQUEST_SCRATCH_CACHE"),
            (str(OUT),"REQUEST_RESULT"),(str(SRC),"REQUEST_SOURCE"),(str(ENTRY),"ORIGINAL_SWIFT_ENTRY"),
            (str(FRAME),"ORIGINAL_FRAMEWORK"),(str(PROCESSOR),"PROCESSOR_ONLY_METADATA"),
            (original.TRAIN_MEDIA,"ORIGINAL_TRAIN_MEDIA"),("/r28-original/","OTHER_PINNED_ORIGINAL_ASSET"),
            ("/swift-runtime/","SWIFT_RUNTIME"),("/minicpm-transformers/","TRANSFORMERS_RUNTIME"),
            ("/dev/shm","SYSTEM_SHM"),("/tmp","SYSTEM_TMP"),("/public/","SHARED_PATH_OTHER"),
            ("/usr/","SYSTEM_RUNTIME")):
        if name==prefix or name.startswith(prefix if prefix.endswith("/") else prefix+"/"):
            return category
    return "OTHER_ABSOLUTE_PATH" if os.path.isabs(name) else "RELATIVE_OR_NON_PATH"


def diagnostic(error):
    errors=[]
    pending=error
    visited=set()
    frames=[]
    while pending is not None and id(pending) not in visited and len(errors)<3:
        visited.add(id(pending))
        code=getattr(pending,"errno",None)
        entry={"type":type(pending).__name__,"errno":code if type(code) is int else None,
               "errno_symbol":errno.errorcode.get(code) if type(code) is int else None,
               "path_classes":[category for category in (path_class(getattr(pending,"filename",None)),
                    path_class(getattr(pending,"filename2",None))) if category]}
        if isinstance(pending,original.Reject):
            text=str(pending)
            if re.fullmatch(r"[A-Z0-9_]{1,88}",text):
                entry["gate"]=text
        errors.append(entry)
        stack=[]
        tb=pending.__traceback__
        while tb is not None:
            frame=tb.tb_frame
            name=frame.f_globals.get("__name__","")
            function=frame.f_code.co_name
            if isinstance(name,str) and re.fullmatch(r"[a-zA-Z0-9_.]{1,100}",name):
                if name.split(".")[0] not in {"swift","datasets","pyarrow","tempfile","pathlib","os",
                                               "torch","accelerate","transformers","probe","__main__","probe_r29"}:
                    name="other"
                stack.append({"module":name,"function":function if re.fullmatch(r"[A-Za-z0-9_<>]{1,80}",function)
                              else "other","line":tb.tb_lineno})
            tb=tb.tb_next
        frames.extend(stack[-(6-len(frames)): ] if len(frames)<6 else [])
        pending=pending.__cause__ or (None if pending.__suppress_context__ else pending.__context__)
    return {"errors":errors,"frames":frames[:6],"raw_text_included":False}


def mount_info(paths):
    found={}
    with Path("/proc/self/mountinfo").open("r",encoding="utf-8") as stream:
        text=stream.read(1048577)
    need(len(text)<=1048576,"BOUNDED_MOUNTINFO")
    for path in paths:
        exact=str(path)
        options=[row.split()[5].split(",") for row in text.splitlines()
                 if len(row.split())>=6 and row.split()[4]==exact]
        need(len(options)==1,"EXACT_REQUEST_BIND_MOUNT")
        found[path_class(path)]="ro" if "ro" in options[0] else "rw" if "rw" in options[0] else "unknown"
    return found


def write(name,value):
    raw=canonical(value)
    need(len(raw)<=65536,"BOUNDED_PRIVATE_OUTPUT")
    with (OUT/name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes":len(raw),"SHA256":digest(raw)}


@contextmanager
def quiet_case():
    previous=logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull,"w",encoding="utf-8") as sink,redirect_stdout(sink),redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous)


def scratch_path(report):
    need(SCRATCH.is_dir() and SCRATCH.resolve(strict=True)==SCRATCH and not any(SCRATCH.iterdir()),
         "EXCLUSIVE_EMPTY_REQUEST_SCRATCH")
    path=Path(tempfile.mkdtemp(prefix="native-",dir=SCRATCH))
    report["scratch"]["created"]=True
    first=path/".self-test"
    second=path/".self-test-renamed"
    with first.open("xb") as stream:
        stream.write(b"r29\n")
        stream.flush()
        os.fsync(stream.fileno())
    first.rename(second)
    need(second.read_bytes()==b"r29\n","SCRATCH_ROUNDTRIP")
    second.unlink()
    report["scratch"]["write_rename_unlink"]=True
    return path


def configure_cache(temp,report):
    for token in ("HF_HOME","HF_HUB_CACHE","HF_DATASETS_CACHE","XDG_CACHE_HOME","TMPDIR","TEMP","TMP"):
        path=temp/token.lower()
        path.mkdir()
        os.environ[token]=str(path)
    tempfile.tempdir=os.environ["TMPDIR"]
    report["scratch"]["python_temp_class"]=path_class(tempfile.gettempdir())


def cli_inputs(plan,temp):
    command=plan["command"]
    need(command[1]=="sft" and command[0].endswith("/vlm/runtime/swift_entry.py"),"ORIGINAL_SWIFT_ENTRY_CLI_SHAPE")
    args=command[2:].copy()
    for key,value in (("--dataset",str(original.VIEW)),("--output_dir",str(temp/"unused-output")),
                      ("--model",str(PROCESSOR)),
                      ("--external_plugins",str(FRAME/"vlm/plugins/gemma_domain_continuous.py"))):
        need(args.count(key)==1,"EXACT_ORIGINAL_CLI_BIND")
        args[args.index(key)+1]=value
    return args


def entry_evidence(manifest):
    record=manifest["files"]["vlm/runtime/swift_entry.py"]
    raw=read(ENTRY,record["bytes"],record["SHA256"])
    tree=ast.parse(raw)
    imports=sorted({(item.module or "").split(".")[0] for item in ast.walk(tree) if isinstance(item,ast.ImportFrom)})
    return {"SHA256":digest(raw),"source_bytes":len(raw),"references_swift":("swift" in imports or b"swift." in raw)}


def original_cli(SwiftSft,plan,temp,report):
    # The true Swift parser determines operative argument defaults. Never instantiate
    # the normal SFT constructor, which prepares the model before the dataset.
    instance=object.__new__(SwiftSft)
    parser=instance._parse_args
    get_data=getattr(SwiftSft,"_get_dataset",None)
    need(callable(get_data) and len(inspect.signature(parser).parameters)==1
         and len(inspect.signature(get_data).parameters)==1,"ORIGINAL_SWIFT_SFT_DATA_ONLY_API")
    report["swift_parse_provider"]=original.provider(SwiftSft._parse_args)
    report["swift_data_provider"]=original.provider(get_data)
    parsed=parser(cli_inputs(plan,temp))
    need(getattr(parsed,"dataset_shuffle",None) is True and getattr(parsed,"split_dataset_ratio",None)==0
         and tuple(getattr(parsed,"dataset",()))==(str(original.VIEW),)
         and getattr(parsed,"seed",None)==42 and getattr(parsed,"data_seed",None)==42
         and getattr(parsed,"packing",None) is False,"ACTUAL_ORIGINAL_SWIFT_CLI_DATA_ARGUMENTS")
    def load(shuffle):
        args=copy(parsed)
        args.dataset_shuffle=shuffle
        instance.args=args
        value=get_data(instance)
        need(isinstance(value,tuple) and len(value)==2 and value[1] is None,"NO_DEV_OR_BLIND_DATASET")
        return value[0]
    report["actual_sft_cli_parse_passed"]=True
    return load


def cpu_only(event,args):
    original.offline(event,args)
    if event=="open" and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if re.search(r"\.(?:safetensors|pt|pth|ckpt|bin)$",path,re.I) and path.startswith(("/public/","/r29-processor/")):
            raise original.Reject("MODEL_WEIGHT_OPEN_FORBIDDEN")


def package(expected):
    closed=json.loads(read(SRC/"SOURCE_FILES.json",expected=expected))
    need(set(closed["files"])==PACKAGE and closed["prior_compute_source_manifest_SHA256"]==OLD_MANIFEST,
         "EXACT_R30_SOURCE_CLOSURE")
    for name,meta in closed["files"].items():
        read(SRC/name,meta["bytes"],meta["SHA256"])


def run(args):
    step=original.placement(args.parent,args.node,args.epoch,time.time(),os.environ,
                           socket.gethostname().split(".")[0],len(os.sched_getaffinity(0)))
    need(re.fullmatch(r"[0-9a-f]{32}",args.request) and args.request not in {"1f07ca7b65a647d0862a978dfd0c51d4","aecf42f2d9c543c4b80b05b7d52436ef","89c915db062a4574a9b0a1b2c0c6f7c7","f51e026158944d79ace5cf3ce54d8241","9b0e8c00e71d40d8bb11dc60ab320001","95fc35d6951b48c8b4b965da3428dcdd"}
         and re.fullmatch(r"[0-9a-f]{64}",args.manifest) and OUT.is_dir() and not any(OUT.iterdir())
         and os.environ.get("PYTHONPYCACHEPREFIX") is None and sys.pycache_prefix is None and sys.dont_write_bytecode,
         "FRESH_DISTINCT_R29_CPU_OWNER")
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    remaining=args.epoch+120-time.time()
    end=time.monotonic()+remaining
    def check():
        need(time.monotonic()<end and time.time()<args.epoch+120,"R29_CPU_DEADLINE")
    signal.signal(signal.SIGALRM,original.expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
    report={"schema":"R34_NATIVE_PRIVATE_UNIX_LOADER_PHASES_V1","request":args.request,"step":step,"node":args.node,
            "epoch":args.epoch,"deadline":args.epoch+120,"source_manifest_SHA256":args.manifest,
            "prior_step":ORIGINAL_FAILURE,"prior_replay_allowed":False,
            "actual_gpu":0,"actual_optimizer_updates":0,"actual_training_ranks":0,"training_ready":False,
            "planned_rows":800,"planned_rank_projections":8,"production_adoption":False,"phases":[],
            "builder_cache_evidence":{"calls":[],"recovery_calls":0,"swift_namespace_references":[]},
            "modelscope_cache":{"swift_tmp_returns":{}},
            "native_loader_completed":{"swift_unshuffled_call":False,"swift_shuffled_call":False},
            "arrow_reached":False,"sampler_reached":False,
            "scratch":{"created":False,"write_rename_unlink":False,"removed":False}}
    write("INTENT.json",report)
    mark="source_preflight"
    temp=None
    witness=None
    tracking=None
    try:
        package(args.manifest)
        report["mount_access"]=mount_info((original.VIEW,SCRATCH,OUT,SRC,FRAME,PROCESSOR))
        need(report["mount_access"]=={"ORIGINAL_TRAIN800_JSONL":"ro","REQUEST_SCRATCH_CACHE":"rw",
             "REQUEST_RESULT":"rw","REQUEST_SOURCE":"ro","ORIGINAL_FRAMEWORK":"ro","PROCESSOR_ONLY_METADATA":"ro"},
             "EXACT_READONLY_AND_WRITABLE_BIND_MODES")
        mark="input_open"
        raw=read(original.VIEW,*original.PINS[original.VIEW])
        report["original_input"]={"SHA256":digest(raw),"bytes":len(raw),"physical_rows":len(raw.splitlines())}
        need(report["original_input"]["physical_rows"]==800,"ORIGINAL_800_INPUT_ROWS")
        report["phases"].append(mark)
        plan_raw=read(original.ORIGINAL_PLAN,*original.PINS[original.ORIGINAL_PLAN])
        plan=json.loads(plan_raw)
        workers=original.recipe(plan_raw)
        version=json.loads(read(original.RUNTIME,*original.PINS[original.RUNTIME]))
        framework=json.loads(read(original.FRAMEWORK_MANIFEST,*original.PINS[original.FRAMEWORK_MANIFEST]))
        need(version["versions"]==original.VERSIONS and len(framework["files"])==396
             and {name:importlib.metadata.version(name) for name in original.VERSIONS}==original.VERSIONS,
             "EXACT_ORIGINAL_845_VERSIONS")
        report["framework_swift_entry"]=entry_evidence(framework)
        processor=json.loads(read(SRC/"MODEL_PROCESSOR_FILES.json",*PROCESSOR_PIN))["files"]
        need(len(processor)==14 and {member.name for member in PROCESSOR.iterdir()}==set(processor),"EXACT_NO_WEIGHT_PROCESSOR_ROOT")
        for name,value in processor.items():
            read(PROCESSOR/name,value["bytes"],value["SHA256"],64*1024**2)
        mark="request_scratch_create_and_cache"
        temp=scratch_path(report)
        configure_cache(temp,report)
        modelscope_root=modelscope_cache.setup(temp,report["modelscope_cache"])
        report["phases"].append(mark)
        witness_root=ipc_witness.scratch_root(SCRATCH,temp,os.environ.get("TMPDIR"),os.getuid())
        witness=ipc_witness.Admission(OUT/ipc_witness.FILENAME,args.request,step,witness_root,phase=lambda:mark,deadline=check)
        tracking=witness.track_local_listeners()
        tracking.__enter__()
        sys.addaudithook(lambda event,values:witness.guard(event,values,cpu_only))
        mark="native_provider_import"
        with quiet_case():
            import torch
            import safetensors
            import safetensors.torch
            from accelerate.data_loader import BatchSamplerShard,SeedableRandomSampler,prepare_data_loader
            from datasets import Dataset,config as dsconfig,disable_caching
            from swift.dataset import load_dataset
            from transformers import Seq2SeqTrainingArguments,Trainer
            from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
            need(not torch.cuda.is_initialized(),"NO_GPU_CONTEXT")
            def deny(*_a,**_kw):
                raise original.Reject("NO_GPU_OR_WEIGHT_DESERIALIZER")
            torch.cuda._lazy_init=deny
            torch.load=deny
            safetensors.safe_open=deny
            safetensors.torch.safe_open=deny
            safetensors.torch.load_file=deny
            torch.set_num_threads(2)
            disable_caching()
        trainer=original.provider(Trainer)
        need(trainer["source_SHA256"]==version["trainer"]["SHA256"] and trainer["source_bytes"]==version["trainer"]["bytes"],
             "EXACT_RUNTIME_TRAINER_SOURCE")
        report["torch_accelerate_provider_verified"]=True
        report["scratch"]["actual_datasets_cache_class"]=path_class(dsconfig.HF_DATASETS_CACHE)
        need(report["scratch"]["actual_datasets_cache_class"]=="REQUEST_SCRATCH_CACHE"
             and path_class(tempfile.gettempdir())=="REQUEST_SCRATCH_CACHE","ACTUAL_CACHE_AND_TMP_BOUND_TO_REQUEST")
        report["phases"].append(mark)
        mark="actual_modelscope_getter"
        with quiet_case():
            import swift.dataset.utils as swift_utils
            import modelscope.hub.utils.utils as model_utils
            swift_temp=modelscope_cache.verify(swift_utils,model_utils,modelscope_root,temp,path_class,read,report["modelscope_cache"])
        report["phases"].append(mark)
        mark="original_sft_cli_parse_and_data_binding"
        try:
            with quiet_case():
                from swift.pipelines.train.sft import SwiftSft
                route=original_cli(SwiftSft,plan,temp,report)
            report["route"]="ACTUAL_ORIGINAL_SWIFT_SFT_CLI_DATA_METHOD"
            report["phases"].append(mark)
        except (OSError,ValueError,KeyError,TypeError,AttributeError,ImportError,RuntimeError) as error:
            report["original_sft_binding_diagnosis"]=diagnostic(error)
            report["original_sft_argument_identity"]=repair.sft_value_error(error)
            if not args.allow_component_fallback:
                raise
            route=lambda shuffled:original.swift_load(load_dataset,shuffled)
            report["route"]="DIRECT_SWIFT_LOAD_DATASET_COMPONENT_ONLY"
        with repair.native_binding(temp,path_class,report["builder_cache_evidence"]) as observed:
            mark="swift_unshuffled_call"
            observed.stage=mark
            check()
            with quiet_case(),modelscope_cache.observe(swift_temp,temp,modelscope_root,path_class,report["modelscope_cache"],mark):
                with witness.loader_phase(mark):
                    unshuffled=route(False)
            report["native_loader_completed"][mark]=True
            report["phases"].append(mark)
            mark="swift_shuffled_call"
            observed.stage=mark
            check()
            with quiet_case(),modelscope_cache.observe(swift_temp,temp,modelscope_root,path_class,report["modelscope_cache"],mark):
                with witness.loader_phase(mark):
                    shuffled=route(True)
            report["native_loader_completed"][mark]=True
            report["phases"].append(mark)
        mark="arrow_physical_row_lineage"
        report["arrow_reached"]=True
        lineage=original.physical_lineage(raw,unshuffled,shuffled,Dataset)
        report["phases"].append(mark)
        report["actual_swift_physical_rows"]=len(lineage)
        mark="native_stock_sampler"
        report["sampler_reached"]=True
        with quiet_case():
            ranks,providers=original.stock_projections(shuffled,lineage,torch,Seq2SeqTrainer,Seq2SeqTrainingArguments,
                prepare_data_loader,BatchSamplerShard,SeedableRandomSampler,torch.utils.data.BatchSampler,workers,check)
        need(not torch.cuda.is_initialized(),"NO_GPU_CONTEXT")
        ipc_summary=witness.summary()
        metrics=ipc_summary["phase_metrics"]
        need(ipc_summary["phases_closed"]==2 and ipc_summary["state_failure"] is None
             and ipc_summary["owner_diagnostic_failures"]==0 and sum(p["native_returned"] for p in metrics.values())>0
             and all(p["permits"]==p["native_returned"] and p["native_failed"]==p["pending"]==0 for p in metrics.values()),
             "R34_NATIVE_IPC_CLOSED_PHASES_WITH_ACCOUNTED_RETURNS")
        first_two={rank:[order[:4],order[4:8]] for rank,order in ranks.items()}
        check_slots=[ordinal for updates in first_two.values() for boundary in updates for ordinal in boundary]
        need(len(first_two)==8 and len(check_slots)==len(set(check_slots))==64,"R33_EIGHT_RANKS_TWO_BOUNDARIES")
        report["native_first_two_rank_slots"]={"rank_projections":8,"boundaries":2,"slots":64,
            "distinct_original_ordinals":64,"private_sha256":digest((json.dumps(first_two,sort_keys=True,separators=(",",":"))+"\n").encode())}
        report["phases"].append(mark)
        payload={"schema":"R29_NATIVE_CPU_PHYSICAL_PROJECTION_V1","step":step,"request":args.request,
                 "route":report["route"],"providers":providers,"ranks":ranks,"raw_ordinals_by_dataset_index":lineage,
                 "baseline_view_SHA256":original.PINS[original.VIEW][1],"two_optimizer_boundaries_projection_only":first_two,"production_adoption":False}
        report.update(state="NATIVE_CPU_DIAGNOSTIC_PROJECTION_COMPLETE",passed=True,
                      actual_rank_projections=8,actual_physical_slots=800,private_projection=write("PRIVATE_ORDER.json",payload))
    except (OSError,ValueError,KeyError,TypeError,AttributeError,ImportError,RuntimeError) as error:
        report.update(state="NATIVE_CPU_DIAGNOSIS_STOPPED",passed=False,diagnostic=diagnostic(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if tracking is not None:
            tracking.__exit__(None,None,None)
        if witness is not None:
            try:
                report["ipc_boundary"]=witness.summary()
            except (OSError,ValueError,KeyError,TypeError):
                report["ipc_boundary"]={"policy":"REQUEST_PRIVATE_UNIX_EXACT_PHASE_ENDPOINT",
                                         "summary":"UNAVAILABLE","raw_addresses_forwarded":False}
        if temp is None and report["scratch"]["created"]:
            leaves=list(SCRATCH.iterdir())
            if len(leaves)==1 and leaves[0].name.startswith("native-"):
                temp=leaves[0]
        if temp is not None:
            try:
                shutil.rmtree(temp)
                report["scratch"]["removed"]=not temp.exists() and not any(SCRATCH.iterdir())
            except OSError as error:
                report["scratch"]["cleanup_diagnostic"]=diagnostic(error)
        if report["scratch"]["created"] and not report["scratch"]["removed"]:
            report.update(state="NATIVE_CPU_DIAGNOSIS_SCRATCH_CLEANUP_FAILED",passed=False)
        report["current_phase"]=mark
        write("RESULT.json",report)
    return int(not report["passed"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","node","request","manifest"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--epoch",required=True,type=float)
    parser.add_argument("--allow-component-fallback",action="store_true")
    return run(parser.parse_args())


if __name__=="__main__":
    sys.exit(main())
