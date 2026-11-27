"""Observe the actual Swift sampler on the frozen 64-row geometry, without a model."""
import argparse
import ast
from hashlib import sha256
import importlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import stat
import time
from datetime import timedelta


SWIFT = Path("/swift-runtime/site/swift")
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
SPLIT = ROOT/"internvl-gqa64-train56-internal8-aa810b24efcc4a5ab04a3cdd1c7825a4/output"
SPLIT_PIN = "28c6a29c501bbb0ef066014aadfc15eaa43fc5f31952d9960917208ca6c669d1"
DATA_PIN = "28643ee29fc0a2d59ccf26883a66bd870dbb89be1f7f710c23dc9aca0727d174"
MIXIN_PIN = "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9"
METHOD_AST_PIN = "44fe8efc2f2fd5ba4548730a14707fb52b955c53807e590ae467af73cf95a0bc"
METHOD_LEXICAL_PIN = "5d2aeed447b350581013b8356e4e9586207e72e3c895c9c705dbfbbf79151d43"
REPRESENTATION_RECEIPT_PIN = "a71b56437625a64257d78f23a841ddabb3f51e36230753dff5448e52a642c83f"
PREDECESSOR_REQUESTS = {"6866e939973d4743bc44dfe9c308e135","f9acc59f42f947af8c41e5f3276cc010"}
PREVIOUS_ARCHIVE_SHA = "e7b8eb1549e0ce496f6c735f1a53a8ddcaaf310ff7b61c9ce4d9221463cb1ab0"
ARG_PIN = "e3fd302d795e2ab09ef07124ad6c92991b9c0be486c8c12c0ef6e648eca38683"
SHARD_PIN = "ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c"
SIZE, WORLD, BATCH, ACCUMULATION = 64, 8, 1, 4
MASKS = ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")
FIELDS = {"total_samples":SIZE,"world_size":WORLD,"batch_size":BATCH,"tp_size":1,"shuffle":True,
          "drop_last":False,"base_seed":42,"curr_seed":42,"group_by_length":False}
PHASES = ("start","provider","init","draw","destroy")
ERROR_CODE = re.compile(r"[A-Z_0-9]{1,120}")


def need(ok,code):
    if not ok:
        raise ValueError(code)


def raw_hash(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path,size,digest,*,exact=True):
    need(path.resolve(strict=True)==path,"CANONICAL_FROZEN_INPUT")
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    with os.fdopen(os.open(path,flags),"rb") as stream:
        before=os.fstat(stream.fileno())
        raw=stream.read(size+1)
        after=os.fstat(stream.fileno())
    signature=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
    need(stat.S_ISREG(before.st_mode) and signature(before)==signature(after)==signature(path.lstat()) and
         0<len(raw)<=size and (not exact or len(raw)==size) and raw_hash(raw)==digest,"FROZEN_INPUT_IDENTITY")
    return raw


def verify_split():
    summary=json.loads(pin(SPLIT/"SPLIT.json",8192,SPLIT_PIN,exact=False))
    need(summary["target_original_binding_verified"] is True and summary["optimizer_updates"]==0 and
         summary["files"]["proposed-exposures64.jsonl"]==dict(bytes=19599,SHA256=DATA_PIN),
         "EXACT_FROZEN_SPLIT_GEOMETRY")
    raw=pin(SPLIT/"proposed-exposures64.jsonl",19599,DATA_PIN)
    need(len(raw.splitlines())==SIZE,"EXACT_64_PHYSICAL_JSONL_ROWS")


def load_source(relative, size, digest, role):
    try:
        return pin(SWIFT/relative,size,digest)
    except (OSError, ValueError):
        raise ValueError("EXACT_SWIFT_"+role+"_FILE") from None


def load_module(name, relative, role):
    try:
        module=importlib.import_module(name)
    except (AttributeError, ImportError, OSError, RuntimeError):
        raise ValueError("EXACT_SWIFT_"+role+"_IMPORT") from None
    need(getattr(module,"__file__",None)==str(SWIFT/relative),"EXACT_SWIFT_"+role+"_ORIGIN")
    return module


def method_segment(raw):
    text=raw.decode("utf-8")
    owners=[node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name=="DataLoaderMixin"]
    need(len(owners)==1,"EXACT_STATIC_LOADER_OWNER")
    matches=[node for node in owners[0].body if isinstance(node,ast.FunctionDef) and node.name=="get_train_dataloader"]
    need(len(matches)==1,"EXACT_STATIC_LOADER_METHOD")
    node=matches[0]
    need(node.lineno==1276,"EXACT_STATIC_LOADER_FIRST_LINE")
    segment=ast.get_source_segment(text,node)
    need(segment is not None,"EXACT_STATIC_LOADER_SEGMENT_PRESENT")
    encoded=segment.encode("utf-8")
    need(len(encoded)==2763,"EXACT_STATIC_LOADER_SEGMENT_EXTENT")
    need(raw_hash(encoded)==METHOD_AST_PIN,"EXACT_STATIC_LOADER_SEGMENT_SHA")


def callable_identity(mod, impl):
    method=getattr(getattr(mod,"DataLoaderMixin",None),"get_train_dataloader",None)
    need(inspect.isfunction(method) and method.__qualname__=="DataLoaderMixin.get_train_dataloader" and
         method.__module__==mod.__name__,"EXACT_LIVE_LOADER_METHOD_OWNER")
    try:
        location=inspect.getsourcefile(method)
        lines,first=inspect.getsourcelines(method)
    except (OSError,TypeError):
        raise ValueError("EXACT_LIVE_LOADER_INSPECTION_AVAILABLE") from None
    need(location==str(SWIFT/"trainers/mixin.py"),"EXACT_LIVE_LOADER_METHOD_PATH")
    need(first==1276,"EXACT_LIVE_LOADER_FIRST_LINE")
    lexical="".join(lines).encode("utf-8")
    need(len(lexical)==2768,"EXACT_LIVE_LOADER_LEXICAL_EXTENT")
    need(raw_hash(lexical)==METHOD_LEXICAL_PIN,"EXACT_LIVE_LOADER_LEXICAL_SHA")
    for name in ("BatchSamplerShard","DataLoaderShard"):
        expected=getattr(impl,name,None)
        code="EXACT_LIVE_"+("BATCH_SAMPLER_CLASS" if name=="BatchSamplerShard" else "DATA_LOADER_CLASS")
        need(inspect.isclass(expected) and expected.__module__==impl.__name__ and
             method.__globals__.get(name) is expected,code)


def check_provider():
    mixin=load_source("trainers/mixin.py",64818,MIXIN_PIN,"MIXIN")
    load_source("trainers/arguments.py",16906,ARG_PIN,"ARGUMENTS")
    load_source("dataloader/shard.py",3553,SHARD_PIN,"SHARD")
    method_segment(mixin)
    mod=load_module("swift.trainers.mixin","trainers/mixin.py","MIXIN")
    arg=load_module("swift.trainers.arguments","trainers/arguments.py","ARGUMENTS")
    impl=load_module("swift.dataloader.shard","dataloader/shard.py","SHARD")
    callable_identity(mod,impl)
    # Look up the deployed field on every class declaring it, without constructing SFT arguments.
    defaults=[field.default for cls in vars(arg).values() if isinstance(cls,type) and cls.__module__==arg.__name__
              for field in getattr(cls,"__dataclass_fields__",{}).values() if field.name=="train_dataloader_shuffle"]
    need(defaults and all(value is True for value in defaults),"EXACT_NATIVE_DATALOADER_SHUFFLE_DEFAULT")
    return impl.BatchSamplerShard


def rank_draw(torch, shard_type, rank):
    dist=torch.distributed
    need((dist.get_rank(),dist.get_world_size())==(rank,WORLD),"OWN_CPU_PROCESS_GROUP_GEOMETRY")
    shard=shard_type(total_samples=SIZE,batch_size=BATCH,shuffle=True,drop_last=False,data_seed=42,
                     tp_size=1,group_by_length=False,lengths=None)
    config={**FIELDS,"rank":rank}
    need(type(shard) is shard_type and
         all(type(getattr(shard,key)) is type(expected) and getattr(shard,key)==expected for key,expected in config.items())
         and len(shard)==SIZE//WORLD,"EXACT_ACTUAL_SWIFT_SHARD_FIELDS")
    before=torch.get_rng_state()
    indexes=[]
    for batch in shard:
        need(len(indexes)<8 and type(batch) is list and len(batch)==1 and type(batch[0]) is int and
             0<=batch[0]<SIZE,"EXACT_BOUNDED_NATIVE_BATCH")
        indexes.append(batch[0])
    need(torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized(),"UNCHANGED_CPU_RNG_AND_GPU_LAZY_STATE")
    need(len(indexes)==len(set(indexes))==8,"EXACT_EIGHT_LOGICAL_INDEXES")
    return indexes


def combine(rows):
    need(len(rows)==WORLD and all(len(item)==8 and len(set(item))==8 for item in rows),"EXACT_EIGHT_RANKS")
    combined=[item for row in rows for item in row]
    need(set(combined)==set(range(SIZE)) and len(combined)==SIZE,"EXACT_FULL_LOGICAL_PERMUTATION")
    return dict(native_logical_permutation_verified=True,rank_slots=[len(row) for row in rows],
                native_order_SHA256=raw_hash(encode(rows)),rank_logical_indices=rows,
                matches_unshuffled_interleave=all(row==list(range(rank,SIZE,WORLD)) for rank,row in enumerate(rows)),
                swift_dataloader_method_AST_segment_SHA256=METHOD_AST_PIN,
                swift_dataloader_method_lexical_sourcelines_SHA256=METHOD_LEXICAL_PIN,
                swift_batch_sampler_source_SHA256=SHARD_PIN,
                actual_loader_invoked=False,actual_arrow_mapping_verified=False,physical_forward_order_verified=False,
                actual_model_loaded=False,actual_optimizer_updates=0,training_ready=False)


def write(path,data):
    with path.open("xb") as stream:
        stream.write(encode(data))
        stream.flush()
        os.fsync(stream.fileno())


def safe_symbol(value):
    return value if re.fullmatch(r"[A-Za-z0-9_.<>-]{1,100}",value) else None


def failure(error):
    frames=[]
    tb=error.__traceback__
    while tb is not None:
        code=tb.tb_frame.f_code
        frames.append(dict(basename=safe_symbol(Path(code.co_filename).name),
                           function=safe_symbol(code.co_name),line=tb.tb_lineno))
        tb=tb.tb_next
    message=str(error)
    errno=getattr(error,"errno",None)
    return dict(error_type=safe_symbol(type(error).__name__),errno=errno if type(errno) is int else None,
                message_SHA256=raw_hash(message.encode("utf-8","replace")),
                error_code=message if isinstance(error,ValueError) and
                ERROR_CODE.fullmatch(message) and message.startswith(("EXACT_","OWN_","UNCHANGED_")) else None,
                stack=frames[-6:])


def provider():
    import torch
    need(str(torch.__version__)=="2.10.0" and torch.version.hip=="6.3.26113" and torch.version.cuda is None
         and not torch.cuda.is_initialized(),"EXACT_MASKED_HIP_TORCH")
    torch.set_num_threads(1)
    shard=check_provider()
    return torch,shard


def mark(here,rank,phase,status,**details):
    write(here/f"rank-{rank}-{phase}-{status}.json",
          dict(rank=rank,pid=os.getpid(),phase=phase,status=status,monotonic_ns=time.monotonic_ns(),**details))


def worker(here,rank,work_deadline,*,load_provider=provider):
    groups=None
    phase="start"
    failed=False
    indexes=None
    try:
        mark(here,rank,phase,"entered")
        need(time.monotonic()<work_deadline,"EXACT_WORK_DEADLINE_BEFORE_START")
        mark(here,rank,phase,"returned")
        phase="provider"
        mark(here,rank,phase,"entered")
        torch,shard=load_provider()
        groups=torch.distributed
        mark(here,rank,phase,"returned")
        phase="init"
        mark(here,rank,phase,"entered")
        remaining=work_deadline-time.monotonic()
        need(remaining>0,"EXACT_WORK_DEADLINE_BEFORE_GLOO")
        need(groups.is_gloo_available() and not groups.is_initialized(),"OWN_NEW_CPU_GLOO_GROUP_REQUIRED")
        groups.init_process_group("gloo",init_method=(here/"group-store").as_uri(),rank=rank,
                                  world_size=WORLD,timeout=timedelta(seconds=min(50,remaining)))
        need(groups.is_initialized(),"EXACT_GLOO_GROUP_INITIALIZED")
        mark(here,rank,phase,"returned")
        phase="draw"
        mark(here,rank,phase,"entered")
        indexes=rank_draw(torch,shard,rank)
        mark(here,rank,phase,"returned",indexes=indexes)
    except Exception as error:
        failed=True
        mark(here,rank,phase,"error",**failure(error))
    phase="destroy"
    try:
        mark(here,rank,phase,"entered")
        performed=groups is not None and groups.is_initialized()
        if performed:
            groups.destroy_process_group()
            need(not groups.is_initialized(),"EXACT_GLOO_GROUP_DESTROYED")
        mark(here,rank,phase,"returned",performed=performed)
        need(failed or (performed and time.monotonic()<work_deadline),"EXACT_NORMAL_RANK_TEARDOWN_DEADLINE")
    except Exception as error:
        failed=True
        mark(here,rank,phase,"error",**failure(error))
    if failed:
        raise SystemExit(2)
    mark(here,rank,"terminal","normal",slots=len(indexes))


def child_worker(here,rank,deadline):
    # The staged records carry the diagnostic. Native stderr may contain raw paths.
    with open(os.devnull,"wb") as sink:
        os.dup2(sink.fileno(),1)
        os.dup2(sink.fileno(),2)
        try:
            worker(here,rank,deadline)
        except Exception:
            # A failed receipt write may leave a gap visible to the supervisor.
            raise SystemExit(2) from None


def wait_all(children,deadline):
    alive=[child for _,child in children if child.is_alive()]
    while alive and time.monotonic()<deadline:
        time.sleep(min(0.025,max(0,deadline-time.monotonic())))
        alive=[child for child in alive if child.is_alive()]
    for _,child in children:
        child.join(0)
    return alive


def stage_error(row):
    stack=row["stack"]
    need(type(stack) is list and len(stack)<=6 and
         all(isinstance(frame,dict) and set(frame)=={"basename","function","line"} and
             type(frame["line"]) is int and frame["line"]>=0 and
             all(frame[key] is None or (isinstance(frame[key],str) and safe_symbol(frame[key])==frame[key])
                 for key in ("basename","function")) for frame in stack),"EXACT_REDACTED_STAGE_STACK")
    name,errno,message,code=(row[key] for key in ("error_type","errno","message_SHA256","error_code"))
    need((name is None or (isinstance(name,str) and safe_symbol(name)==name)) and
         (errno is None or type(errno) is int) and isinstance(message,str) and
         re.fullmatch(r"[0-9a-f]{64}",message) and
         (code is None or (isinstance(code,str) and ERROR_CODE.fullmatch(code) and
                           code.startswith(("EXACT_","OWN_","UNCHANGED_")))),"EXACT_REDACTED_STAGE_ERROR")
    return dict(rank=row["rank"],phase=row["phase"],error_type=name,errno=errno,message_SHA256=message,
                error_code=code,stack=stack)


def rank_evidence(here,rank,pid,exit_code):
    entered=[]
    returned=[]
    errors=[]
    indexed=None
    destroyed=False
    normal=False
    last_tick=0
    phases=[(phase,status) for phase in PHASES for status in ("entered","returned","error")]
    for phase,status in [*phases,("terminal","normal")]:
        path=here/f"rank-{rank}-{phase}-{status}.json"
        if not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                raw=stream.read(8193)
            need(len(raw)<=8192,"EXACT_BOUNDED_STAGE_RECEIPT")
            row=json.loads(raw)
            tick=row["monotonic_ns"]
            need((row["rank"],row["pid"],row["phase"],row["status"])==(rank,pid,phase,status) and
                 type(tick) is int and tick>=last_tick,"EXACT_STAGE_RECEIPT_IDENTITY")
            last_tick=tick
            projected_error=stage_error(row) if status=="error" else None
        except (OSError,KeyError,TypeError,ValueError) as error:
            errors.append(dict(rank=rank,phase=phase,**failure(error)))
            continue
        if status=="entered":
            entered.append(phase)
        elif status=="error":
            errors.append(projected_error)
        elif status=="returned":
            returned.append(phase)
            if phase=="draw":
                indexed=row.get("indexes")
            if phase=="destroy":
                destroyed=row.get("performed") is True
        else:
            normal=row.get("slots")==8
    report=dict(rank=rank,pid=pid,exit_code=exit_code,entered=entered,returned=returned,
                destroy_performed=destroyed,normal_terminal=normal,errors=errors)
    passed=(exit_code==0 and normal and destroyed and entered==returned==list(PHASES) and not errors and
            isinstance(indexed,list) and len(indexed)==8 and all(type(item) is int for item in indexed))
    return report,indexed if passed else None


def run(here,cap,*,target=child_worker):
    need(cap>1,"EXACT_POSITIVE_COMMON_WORK_DEADLINE")
    cleanup_reserve=min(5,cap/3)
    hard=time.monotonic()+cap
    work=hard-cleanup_reserve
    context=mp.get_context("spawn")
    started=[]
    launch_errors=[]
    cleanup=[]
    try:
        for rank in range(WORLD):
            need(time.monotonic()<work,"EXACT_COMMON_LAUNCH_DEADLINE")
            child=context.Process(target=target,args=(here,rank,work))
            try:
                child.start()
            except OSError as error:
                launch_errors.append(dict(rank=rank,phase="spawn",**failure(error)))
                continue
            started.append((rank,child))
        wait_all(started,work)
    except (OSError, ValueError) as error:
        launch_errors.append(dict(phase="spawn_or_wait",**failure(error)))
    finally:
        for rank,child in started:
            if child.is_alive():
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass
                cleanup.append(dict(rank=rank,action="terminate"))
        wait_all(started,hard-cleanup_reserve/2)
        for rank,child in started:
            if child.is_alive():
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
                cleanup.append(dict(rank=rank,action="kill"))
        wait_all(started,hard)
    by_rank=dict(started)
    survivor_ranks=[rank for rank,child in started if child.is_alive()]
    reports=[]
    slots=[]
    for rank in range(WORLD):
        child=by_rank.get(rank)
        if rank in survivor_ranks:
            report,indexes=dict(rank=rank,pid=child.pid,exit_code=None,evidence_read=False),None
        else:
            report,indexes=rank_evidence(here,rank,child.pid if child else None,child.exitcode if child else None)
        reports.append(report)
        slots.append(indexes)
    status=dict(planned_ranks=WORLD,started_ranks=len(started),reaped_ranks=sum(r["exit_code"] is not None for r in reports),
                after_cleanup_rank_reports=reports,survivor_ranks=survivor_ranks,cleanup=cleanup,launch_errors=launch_errors,
                normal_ranks=sum(x is not None for x in slots),actual_model_loaded=False,actual_optimizer_updates=0,
                native_logical_permutation_verified=False,training_ready=False)
    if all(x is not None for x in slots) and not (cleanup or launch_errors or survivor_ranks):
        try:
            status.update(combine(slots))
        except ValueError as error:
            status["aggregate_error"]=failure(error)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--request",required=True)
    args=parser.parse_args()
    need(re.fullmatch("[0-9a-f]{32}",args.request) and args.request not in
         {"aa810b24efcc4a5ab04a3cdd1c7825a4","d955fae7a2644ff8aececf23b543b386"} | PREDECESSOR_REQUESTS,
         "NEW_SAMPLER_ONLY_REQUEST")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent,
         "CANONICAL_NEW_PRIVATE_RESULT")
    need(all(os.environ.get(key)=="-1" for key in MASKS) and
         os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1","MASKED_OFFLINE_RUNTIME")
    step=os.environ.get("SLURM_STEP_ID","")
    parent=os.environ.get("SLURM_JOB_ID","")
    need(re.fullmatch("[0-9]+",step) and re.fullmatch("[0-9]+",parent) and
         not os.environ.get("SLURM_STEP_GPUS") and os.environ.get("SLURM_GPUS_ON_NODE") in (None,"","0") and
         any({"job_"+parent,"step_"+step} <= set(line.split(":")[-1].split("/"))
             for line in Path("/proc/self/cgroup").read_text().splitlines()),"EXACT_NUMBERED_CPU_SLURM_CGROUP")
    verify_split()
    args.output.mkdir(mode=0o700)
    os.umask(0o077)
    os.environ.update(GLOO_SOCKET_IFNAME="lo",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    receipt=run(args.output,90)
    receipt.update(request=args.request,actual_step=parent+"."+step,independent_zero_GRES_terminal_required=True,
                   original_failed_step="1103606.4638",original_failure_archive_SHA256=PREVIOUS_ARCHIVE_SHA,
                   representation_receipt_SHA256=REPRESENTATION_RECEIPT_PIN)
    write(args.output/"result.json",receipt)
    visible={key:value for key,value in receipt.items() if key!="rank_logical_indices"}
    print(encode(visible).decode(),end="",flush=True)
    return 0 if receipt["native_logical_permutation_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())
