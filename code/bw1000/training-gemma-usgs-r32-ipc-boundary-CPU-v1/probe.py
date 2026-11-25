"""Zero-GRES native Swift/Arrow/Trainer sampler witness for the frozen Gemma 800 view."""

import argparse
from hashlib import sha256
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import tempfile
import time


ROWS,WORLD,RANK_BATCHES,SEED,ACCUM,WORK=800,8,100,42,4,120
VIEW=Path("/r28-original/train800-swift.jsonl")
ORIGINAL_PLAN=Path("/r28-original/continuous-plan.json")
RUNTIME=Path("/r28-original/runtime-binding.json")
FRAMEWORK_MANIFEST=Path("/r28-original/fresh-framework-files.json")
OUT=Path("/r28-out")
RUN_SOURCE=Path("/r28-src")
PINS={
    VIEW:(602794,"74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"),
    ORIGINAL_PLAN:(None,"12d67618d386b04ac973e59e4580f32e8cb938e77d7322fb2bd5e4c31bf64de7"),
    RUNTIME:(350,"2bf81432130fa7c69920753f1666ce5cafedaecefe941327a268ad86d6412b10"),
    FRAMEWORK_MANIFEST:(55787,"40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329"),
}
VERSIONS={"torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580","ms-swift":"4.4.2",
          "transformers":"5.12.1","tokenizers":"0.22.2","Pillow":"11.3.0"}
TRAIN_MEDIA=("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"
             "training-gemma-usgs-render72-native-CPU-v1/data/train/media/")


class Reject(ValueError):
    """An allowlisted diagnostic. No input values leave the worker."""


def need(ok,code):
    if not ok:
        raise Reject(code)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def read(path,size=None,expected=None,cap=4*1024**2):
    first=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(first.st_mode) and 0<first.st_size<=cap
         and (size is None or first.st_size==size),"PINNED_REGULAR_FILE")
    with path.open("rb") as stream:
        raw=stream.read(cap+1)
        opened=os.fstat(stream.fileno())
    last=path.lstat()
    key=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(len(raw)==first.st_size and key(first)==key(opened)==key(last)
         and (expected is None or digest(raw)==expected),"STABLE_SOURCE_PIN")
    return raw


def save(name,value):
    raw=canonical(value)
    need(len(raw)<=65536,"PRIVATE_WITNESS_SIZE_BOUND")
    with (OUT/name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes":len(raw),"SHA256":digest(raw)}


def placement(parent,node,epoch,now,env,hostname,affinity):
    step=env.get("SLURM_STEP_ID","")
    need(re.fullmatch(r"[0-9]+",parent) and parent==env.get("SLURM_JOB_ID") and re.fullmatch(r"[0-9]+",step)
         and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node) and node==hostname
         and env.get("SLURM_CPUS_PER_TASK")=="4" and affinity>=4,"EXACT_NUMERIC_4CPU_STEP")
    need(env.get("SLURM_STEP_GPUS","") in ("","0","(null)","none","N/A")
         and all(env.get(name,"") in ("","0","(null)") for name in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not re.search("gpu|dcu",env.get("SLURM_TRES_PER_TASK",""),re.I)
         and all(env.get(name)=="-1" for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
         and all(env.get(name)=="1" for name in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE")),
         "NO_STEP_GRES_AND_NO_VISIBLE_GPU")
    need(math.isfinite(epoch) and 0<epoch<=now<epoch+WORK,"ORIGINAL_CPU_EPOCH")
    return parent+"."+step


def recipe(raw):
    obj=json.loads(raw)
    args=obj["command"]
    expected={"--dataset_shuffle":"true","--seed":"42","--data_seed":"42","--packing":"false",
              "--per_device_train_batch_size":"1","--gradient_accumulation_steps":"4","--dataloader_drop_last":"false",
              "--max_steps":"25","--split_dataset_ratio":"0","--max_length":"512"}
    need(all(args.count(flag)==1 and args[args.index(flag)+1]==value for flag,value in expected.items())
         and obj["environment_overrides"]["NPROC_PER_NODE"]=="8"
         and obj["required_assets"]["dataset"].endswith("/view/train800-swift.jsonl")
         and obj["resume_from_checkpoint"] is None,"EXACT_FROZEN_ORIGINAL_RECIPE")
    return int(args[args.index("--dataloader_num_workers")+1])


def projection(row):
    messages=row["messages"]
    images=row["images"]
    need(isinstance(messages,list) and [item.get("role") for item in messages]==["user","assistant"]
         and len(images)==1,"ORIGINAL_TRAIN_ONLY_IMAGE_AND_TWO_MESSAGES")
    image=images[0]
    if isinstance(image,dict):
        need(not image.get("bytes"),"ARROW_IMAGE_MUST_REMAIN_PATH_ONLY")
        image=image["path"]
    need(isinstance(image,str) and image.startswith(TRAIN_MEDIA)
         and re.fullmatch(r"[0-9]{3}\.png",image.removeprefix(TRAIN_MEDIA)),"ORIGINAL_TRAIN_ONLY_MEDIA_ALIAS")
    text=[(message["role"],message["content"]) for message in messages]
    need(all(isinstance(content,str) for _,content in text),"ORIGINAL_TRAIN_MESSAGE_SCHEMA")
    return text,image


def provider(callable_object):
    path=Path(inspect.getfile(callable_object)).resolve(strict=True)
    raw=read(path)
    return {"module":callable_object.__module__,"source_SHA256":digest(raw),"source_bytes":len(raw)}


def swift_load(loader,shuffle):
    signature=inspect.signature(loader)
    kwargs={"split_dataset_ratio":0.0,"shuffle":shuffle,"streaming":False,"seed":SEED,"num_proc":1}
    # A dynamic **kwargs-only fallback is insufficient evidence of operative loader semantics.
    need(set(kwargs)<=set(signature.parameters),"SWIFT_DATASET_EXPLICIT_OFFLINE_SHUFFLE_API_REQUIRED")
    try:
        signature.bind([str(VIEW)],**kwargs)
    except TypeError as error:
        raise Reject("SWIFT_DATASET_CALL_SIGNATURE_UNAVAILABLE") from error
    result=loader([str(VIEW)],**kwargs)
    need(isinstance(result,tuple) and len(result)==2 and result[1] is None,"SWIFT_NO_VALIDATION_OR_OTHER_SPLIT")
    return result[0]


def physical_lineage(raw,unshuffled,shuffled,dataset_type):
    need(type(unshuffled) is type(shuffled) is dataset_type and len(unshuffled)==len(shuffled)==ROWS,
         "NATIVE_SWIFT_ARROW_TRAIN_800_ROWS")
    first,second=unshuffled._data.table,shuffled._data.table
    need(len(first)==len(second)==ROWS and first.equals(second,check_metadata=True)
         and {"messages","images"}<=set(first.column_names),"SWIFT_SHUFFLE_PRESERVES_PHYSICAL_ARROW_TABLE")
    source=[json.loads(line) for line in raw.splitlines()]
    need(len(source)==ROWS and unshuffled._indices is None,"UNSHUFFLED_HAS_EXACT_SOURCE_ORDER")
    table=first.select(["messages","images"])
    need(all(projection(source_row)==projection(arrow_row) for source_row,arrow_row in
             zip(source,table.to_pylist(),strict=True)),"EXACT_ORIGINAL_JSONL_TO_ARROW_PAYLOAD_ORDER")
    need(shuffled._indices is not None,"NATIVE_SHUFFLE_PHYSICAL_LINEAGE_UNAVAILABLE")
    indices=shuffled._indices.table
    need(indices.column_names==["indices"],"ONE_NATIVE_ARROW_PHYSICAL_INDEX_COLUMN")
    order=indices.column("indices").to_pylist()
    need(len(order)==ROWS and all(type(i) is int for i in order) and sorted(order)==list(range(ROWS))
         and order!=list(range(ROWS)),"NATIVE_SWIFT_SHUFFLED_ARROW_PERMUTATION")
    return [index+1 for index in order]


def stock_projections(dataset,lineage,torch,trainer_type,args_type,prepare_data_loader,shard_type,
                      seedable_type,stock_batch,workers,check):
    # The same installed trainer method chooses the sampler. No synthetic order or
    # training step is used, and actual distributed GPU loader execution is not asserted.
    args=args_type(output_dir="/dev/null",use_cpu=True,seed=SEED,data_seed=SEED,max_steps=25,
                   per_device_train_batch_size=1,gradient_accumulation_steps=ACCUM,
                   dataloader_num_workers=workers,dataloader_drop_last=False)
    accelerator=args.accelerator_config.to_dict()
    need(accelerator.get("use_seedable_sampler") is True and accelerator.get("even_batches") is True
         and accelerator.get("split_batches") is False,"ORIGINAL_DEFAULT_STOCK_SEEDABLE_ACCELERATE_POLICY_UNPROVEN")
    shell=object.__new__(trainer_type)
    shell.args=args
    shell.train_dataset=dataset
    chooser=shell._get_train_sampler
    signature=inspect.signature(chooser)
    need(len(signature.parameters) in (0,1),"NATIVE_TRAINER_SAMPLER_SELECTION_API")
    native={}
    saw=None
    for rank in range(WORLD):
        check()
        choice=chooser(dataset) if signature.parameters else chooser()
        need(isinstance(choice,torch.utils.data.RandomSampler) and choice.replacement is False and
             choice.num_samples==ROWS,"ORIGINAL_STOCK_NO_REPLACEMENT_TRAIN_SAMPLER")
        dataloader=torch.utils.data.DataLoader(dataset=dataset,batch_sampler=stock_batch(choice,1,False),num_workers=0)
        retained=prepare_data_loader(dataloader,device=torch.device("cpu"),num_processes=WORLD,process_index=rank,
            split_batches=False,put_on_device=False,even_batches=True,use_seedable_sampler=True,data_seed=SEED)
        shard=retained.batch_sampler
        need(type(shard) is shard_type and type(shard.batch_sampler) is stock_batch
             and type(shard.batch_sampler.sampler) is seedable_type and shard.process_index==rank
             and shard.num_processes==WORLD and shard.batch_sampler.sampler.initial_seed==SEED,
             "EXACT_NATIVE_ACCELERATE_RANK_SHARD_AND_SEED")
        retained.set_epoch(0)
        slots=list(shard)
        need(len(slots)==RANK_BATCHES and all(len(slot)==1 and type(slot[0]) is int and 0<=slot[0]<ROWS
                 for slot in slots),"ACTUAL_STOCK_RANK_100_BATCHES")
        native[str(rank)]=[lineage[slot[0]] for slot in slots]
        saw=type(shard.batch_sampler.sampler)
    need(sorted(slot for order in native.values() for slot in order)==list(range(1,ROWS+1)),
         "ACTUAL_NATIVE_ALL_EIGHT_PROJECTIONS_NO_PADDING")
    return native,{"transformers_trainer":provider(type(shell)._get_train_sampler),"accelerate_shard":provider(shard_type),
                   "torch_sampler":provider(saw)}


def offline(event,args):
    if event in ("socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"):
        raise Reject("OFFLINE_NATIVE_COMPONENT_ONLY")
    if event=="open" and args and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if any(part in {"dev","blind"} for part in Path(path).parts[1:]) and path.startswith("/public/"):
            raise Reject("NO_OLD_HELDOUT_BODY_PATH")


def expire(*_):
    raise TimeoutError("NATIVE_SAMPLER_CPU_DEADLINE")


def run(args):
    step=placement(args.parent,args.node,args.epoch,time.time(),os.environ,socket.gethostname().split(".")[0],
                   len(os.sched_getaffinity(0)))
    need(re.fullmatch(r"[0-9a-f]{32}",args.request) and re.fullmatch(r"[0-9a-f]{64}",args.manifest)
         and os.environ.get("PYTHONPYCACHEPREFIX") is None and sys.pycache_prefix is None
         and sys.dont_write_bytecode and "torch" not in sys.modules,"FRESH_BOUND_CPU_PYTHON")
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    need(len(os.sched_getaffinity(0))==4 and OUT.is_dir() and not any(OUT.iterdir()),"EXCLUSIVE_CPU_OUTPUT_AND_FOUR_AFFINITY")
    remaining=args.epoch+WORK-time.time()
    mono_end=time.monotonic()+remaining
    def check():
        need(time.monotonic()<mono_end and time.time()<args.epoch+WORK,"NATIVE_SAMPLER_CPU_DEADLINE")
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
    base={"schema":"R28_NATIVE_CPU_COMPONENT_SAMPLER_RESULT_V1","request":args.request,"step":step,"node":args.node,
          "epoch":args.epoch,"deadline":args.epoch+WORK,"planned_physical_rows":800,"planned_rank_projections":8,
          "actual_optimizer_updates":0,"actual_live_training_ranks":0,"training_ready":False,"actual_gpu_steps":0,
          "source_manifest_SHA256":args.manifest}
    save("INTENT.json",base)
    stage="source_preflight"
    try:
        manifest=json.loads(read(RUN_SOURCE/"SOURCE_FILES.json",expected=args.manifest))
        need(set(manifest["files"])=={"probe.py"},"EXACT_COMPUTE_SOURCE_MANIFEST")
        pp=manifest["files"]["probe.py"]
        read(RUN_SOURCE/"probe.py",pp["bytes"],pp["SHA256"])
        raw=read(VIEW,*PINS[VIEW])
        workers=recipe(read(ORIGINAL_PLAN,*PINS[ORIGINAL_PLAN]))
        version=json.loads(read(RUNTIME,*PINS[RUNTIME]))
        frame=json.loads(read(FRAMEWORK_MANIFEST,*PINS[FRAMEWORK_MANIFEST]))
        need(version["versions"]==VERSIONS and len(frame["files"])==396
             and {name:importlib.metadata.version(name) for name in VERSIONS}==VERSIONS,"EXACT_845_RUNTIME_IDENTITY")
        sys.addaudithook(offline)
        with tempfile.TemporaryDirectory(dir="/dev/shm",prefix="r28-swift-") as volatile:
            for name in ("HF_HOME","HF_HUB_CACHE","HF_DATASETS_CACHE","XDG_CACHE_HOME","TMPDIR"):
                os.environ[name]=volatile
            stage="native_provider_import"
            import torch
            from accelerate.data_loader import BatchSamplerShard,SeedableRandomSampler,prepare_data_loader
            from datasets import Dataset,disable_caching
            from swift.dataset import load_dataset
            from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
            from transformers import Seq2SeqTrainingArguments,Trainer
            need(not torch.cuda.is_initialized(),"NO_ACTUAL_GPU_CONTEXT")
            def deny_device(*_args,**_kw):
                raise Reject("GPU_CONTEXT_FORBIDDEN")
            torch.cuda._lazy_init=deny_device
            torch.set_num_threads(2)
            trainer=provider(Trainer)
            need(trainer["source_SHA256"]==version["trainer"]["SHA256"]
                 and trainer["source_bytes"]==version["trainer"]["bytes"],"EXACT_845_TARGET_TRANSFORMERS_TRAINER_BYTES")
            original_getter=provider(Seq2SeqTrainer.get_train_dataloader)
            disable_caching()
            stage="actual_swift_arrow_source_mapping"
            unshuffled=swift_load(load_dataset,False)
            shuffled=swift_load(load_dataset,True)
            lineage=physical_lineage(raw,unshuffled,shuffled,Dataset)
            base.update(actual_swift_physical_rows=len(lineage),swift_arrow_lineage_SHA256=digest(canonical(lineage)))
            check()
            stage="actual_installed_trainer_sampler_rank_projections"
            ranks,observed=stock_projections(shuffled,lineage,torch,Seq2SeqTrainer,Seq2SeqTrainingArguments,
                prepare_data_loader,BatchSamplerShard,SeedableRandomSampler,torch.utils.data.BatchSampler,workers,check)
            observed.update(swift_dataset=provider(load_dataset),arrow_index_provider=provider(Dataset),
                            actual_trainer_loader_method=original_getter,accelerate_prepare=provider(prepare_data_loader))
            need(not torch.cuda.is_initialized(),"NO_ACTUAL_GPU_CONTEXT")
            raw_index={"schema":"R28_NATIVE_CPU_COMPONENT_800_ORDER_V1","request":args.request,"step":step,
                       "baseline_view_SHA256":PINS[VIEW][1],"original_plan_SHA256":PINS[ORIGINAL_PLAN][1],
                       "physical_ordinals_by_dataset_index":lineage,"ranks":ranks,"providers":observed,
                       "actual_rank_processes":0,"actual_native_torch_accelerate_projections":8,
                       "production_sampler_adoption":False}
            reference=save("PRIVATE_CPU_NATIVE_ORDER.json",raw_index)
            base.update(state="ACTUAL_NATIVE_CPU_COMPONENT_WITNESS_PRODUCTION_BIND_REQUIRED",passed=True,
                        actual_native_rank_projections=8,actual_distinct_physical_slots=800,
                        actual_pad_or_repeat_slots=0,private_projection=reference)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,ImportError,RuntimeError) as error:
        base.update(state="NATIVE_CPU_PROBE_BLOCKED",passed=False,reason=str(error) if isinstance(error,Reject)
                    else type(error).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        base.update(phase=stage)
        save("RESULT.json",base)
    return int(not base["passed"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","node","request","manifest"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--epoch",type=float,required=True)
    return run(parser.parse_args())


if __name__=="__main__":
    sys.exit(main())
