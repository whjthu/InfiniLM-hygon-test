"""Observe the actual Swift sampler on the frozen 64-row geometry, without a model."""
import argparse
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
METHOD_PIN = "44fe8efc2f2fd5ba4548730a14707fb52b955c53807e590ae467af73cf95a0bc"
ARG_PIN = "e3fd302d795e2ab09ef07124ad6c92991b9c0be486c8c12c0ef6e648eca38683"
SHARD_PIN = "ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c"
SIZE, WORLD, BATCH, ACCUMULATION = 64, 8, 1, 4
MASKS = ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")
FIELDS = {"total_samples":SIZE,"world_size":WORLD,"batch_size":BATCH,"tp_size":1,"shuffle":True,
          "drop_last":False,"base_seed":42,"curr_seed":42,"group_by_length":False}


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


def check_provider():
    for rel,size,digest in (("trainers/mixin.py",64818,MIXIN_PIN),("trainers/arguments.py",16906,ARG_PIN),
                            ("dataloader/shard.py",3553,SHARD_PIN)):
        pin(SWIFT/rel,size,digest)
    mod=importlib.import_module("swift.trainers.mixin")
    arg=importlib.import_module("swift.trainers.arguments")
    impl=importlib.import_module("swift.dataloader.shard")
    need(Path(mod.__file__)==SWIFT/"trainers/mixin.py" and Path(arg.__file__)==SWIFT/"trainers/arguments.py" and
         Path(impl.__file__)==SWIFT/"dataloader/shard.py","EXACT_SWIFT_PROVIDER_ORIGINS")
    method=mod.DataLoaderMixin.get_train_dataloader
    lines,first=inspect.getsourcelines(method)
    need(first==1276 and raw_hash("".join(lines).encode())==METHOD_PIN and
         method.__globals__["BatchSamplerShard"] is impl.BatchSamplerShard and
         method.__globals__["DataLoaderShard"] is impl.DataLoaderShard,"EXACT_REAL_SWIFT_LOADER_BINDING")
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
                swift_dataloader_method_body_SHA256=METHOD_PIN,swift_batch_sampler_source_SHA256=SHARD_PIN,
                actual_loader_invoked=False,actual_arrow_mapping_verified=False,physical_forward_order_verified=False,
                actual_model_loaded=False,actual_optimizer_updates=0,training_ready=False)


def write(path,data):
    with path.open("xb") as stream:
        stream.write(encode(data))
        stream.flush()
        os.fsync(stream.fileno())


def worker(here,rank):
    groups=None
    try:
        import torch
        need(str(torch.__version__)=="2.10.0" and torch.version.hip=="6.3.26113" and torch.version.cuda is None
             and not torch.cuda.is_initialized(),"EXACT_MASKED_HIP_TORCH")
        torch.set_num_threads(1)
        shard=check_provider()
        groups=torch.distributed
        need(groups.is_gloo_available() and not groups.is_initialized(),"OWN_NEW_CPU_GLOO_GROUP_REQUIRED")
        groups.init_process_group("gloo",init_method=(here/"group-store").as_uri(),rank=rank,
                                  world_size=WORLD,timeout=timedelta(seconds=50))
        indexes=rank_draw(torch,shard,rank)
        write(here/f"rank-{rank}.json",dict(rank=rank,indexes=indexes))
    except (ValueError,AttributeError,OSError,RuntimeError,ImportError,TypeError,KeyError) as error:
        msg=str(error)
        write(here/f"rank-{rank}.json",dict(rank=rank,error_type=type(error).__name__,
              error_code=msg if re.fullmatch("[A-Z_0-9]{1,120}",msg) else None))
    finally:
        if groups is not None and groups.is_initialized():
            groups.destroy_process_group()


def run(here,cap):
    context=mp.get_context("spawn")
    children=[context.Process(target=worker,args=(here,rank)) for rank in range(WORLD)]
    deadline=time.monotonic()+cap
    started=[]
    try:
        for child in children:
            child.start()
            started.append(child)
        for child in started:
            child.join(max(0,deadline-time.monotonic()))
        records=[json.loads(path.read_bytes()) for rank in range(WORLD)
                 if (path:=here/f"rank-{rank}.json").is_file()]
        errors=[{key:value for key,value in row.items() if key!="indexes"} for row in records if "error_type" in row]
        if (len(records)==WORLD and all(r.get("rank")==i and "indexes" in r for i,r in enumerate(records)) and
                all(child.exitcode==0 for child in children)):
            try:
                return combine([row["indexes"] for row in records])
            except ValueError as error:
                errors.append(dict(error_type="ValueError",error_code=str(error)))
        return dict(native_logical_permutation_verified=False,planned_ranks=WORLD,retired_rank_receipts=len(records),
                    errors=errors,
                    actual_model_loaded=False,actual_optimizer_updates=0,training_ready=False)
    finally:
        for child in started:
            if child.is_alive():
                child.terminate()
        for child in started:
            child.join(1)
            if child.is_alive():
                child.kill()
                child.join(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--request",required=True)
    args=parser.parse_args()
    need(re.fullmatch("[0-9a-f]{32}",args.request) and args.request not in
         {"aa810b24efcc4a5ab04a3cdd1c7825a4","d955fae7a2644ff8aececf23b543b386"},"NEW_SAMPLER_ONLY_REQUEST")
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
    receipt.update(request=args.request,actual_step=parent+"."+step,independent_zero_GRES_terminal_required=True)
    write(args.output/"result.json",receipt)
    visible={key:value for key,value in receipt.items() if key!="rank_logical_indices"}
    print(encode(visible).decode(),end="",flush=True)
    return 0 if receipt["native_logical_permutation_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())
