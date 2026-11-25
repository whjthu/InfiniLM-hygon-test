"""Compare the exact live Swift shard algorithm to the sealed .3406 order."""

import argparse
import ast
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import stat
import sys
import time


PARENT,NODE="1103606","f13r1n12"
SHARD=Path("/swift-runtime/site/swift/dataloader/shard.py")
PLAN=Path("/gemma-r45-plan.json")
STAGE=Path("/gemma-r45-index")
SOURCE_SHA="ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c"
OLD_WITNESS_SHA="63726d99926add468a4efacb167301cfb3e72e16a915d379f7c5599221e013d0"
PLAN_SHA="12d67618d386b04ac973e59e4580f32e8cb938e77d7322fb2bd5e4c31bf64de7"
TERMINAL_SHA="74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02"
LINEAGE_SHA="2ab626c9203b233ee4ff87594f9063024492d34f9153e835550dff3907efea25"
ZERO_BASED_LIVE_ORDER_SHA="6cfc9a36eee275885a2f01a92086db9d66d6cf879f71293cbc5dc6af96c34104"
INDEX_SCOPE="6d14e9df9832432b97f1f46f032b49f3"
WORLD,SIZE,PER_RANK,COMPARE=8,800,100,8


def need(ok,code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def read(path,sha=None,size=None,cap=65536):
    before=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap
         and (size is None or before.st_size==size),"G45_PINNED_BOUNDED_FILE")
    ident=lambda m:(m.st_dev,m.st_ino,m.st_size,m.st_mtime_ns,m.st_ctime_ns)
    with path.open("rb") as f:
        opened=os.fstat(f.fileno())
        raw=f.read(cap+1)
        after=os.fstat(f.fileno())
    got={"SHA256":sha256(raw).hexdigest(),"bytes":len(raw)}
    need(ident(before)==ident(opened)==ident(after)==ident(path.lstat()) and len(raw)==before.st_size
         and (sha is None or got["SHA256"]==sha),"G45_STABLE_EXACT_FILE")
    return raw,got


def check_source(raw):
    need(len(raw)==3553 and sha256(raw).hexdigest()==SOURCE_SHA,"G45_EXACT_3465_LIVE_SWIFT_SHARD")
    tree=ast.parse(raw)
    shard,=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="BatchSamplerShard"]
    by_name={func.name:func for func in shard.body if isinstance(func,ast.FunctionDef)}
    algo=ast.unparse(by_name["__iter__"])
    epoch=ast.unparse(by_name["set_epoch"])
    need(all(token in algo for token in ("generator = torch.Generator()","generator.manual_seed(self.curr_seed)",
             "torch.randperm(self.total_samples * self.world_size, generator=generator)",
             "total_idx[self.rank::self.world_size]")) and "self.curr_seed = self.base_seed + epoch" in epoch,
         "G45_EXACT_LIVE_SHUFFLE_AND_RANK_PARTITION")
    return sha256(ast.dump(by_name["__iter__"],include_attributes=False).encode()).hexdigest()


def check_plan(raw):
    frozen=json.loads(raw)
    command=frozen["command"]
    keys={"--per_device_train_batch_size":"1","--gradient_accumulation_steps":"4","--max_steps":"25",
          "--seed":"42","--data_seed":"42","--dataset_shuffle":"true","--dataloader_drop_last":"false"}
    need(all(command.count(k)==1 and command[command.index(k)+1]==v for k,v in keys.items())
         and frozen["resume_from_checkpoint"] is None,"G45_ORIGINAL_CONFIG_SEEDS_BATCH_AND_FRESH_EPOCH")
    return {"hypothesis_seed":42,"epoch_position":0,"per_device_batch":1,"gradient_accumulation":4,
            "expected_world":8,"observed_production_swift_seed_fields":False}


def fixed_private():
    terminal_raw,terminal_pin=read(STAGE/"INDEX_INDEPENDENT_TERMINAL.json",TERMINAL_SHA,484)
    terminal=json.loads(terminal_raw)
    result_pin=terminal["result"]
    intent_pin=terminal["intent"]
    result_raw,result_got=read(STAGE/"INDEX_RESULT.json",result_pin["SHA256"],result_pin["bytes"])
    intent_raw,intent_got=read(STAGE/"INDEX_INTENT.json",intent_pin["SHA256"],intent_pin["bytes"])
    result=json.loads(result_raw);intent=json.loads(intent_raw)
    row=terminal["accounting"]
    need(terminal["independent"] is True and terminal["GPU"]==terminal["optimizer_updates"]==0
         and result["GPU"]==result["actual_optimizer_updates"]==0 and result["actual_signatures"]==64
         and result["passed"] is True and terminal["request"]==result["request"]==intent["request"]==INDEX_SCOPE
         and row[0]==result["step"]=="1103606.3406" and row[2:4]==["COMPLETED","0:0"]
         and result["epoch"]==intent["epoch"],"G45_INDEPENDENT_ORIGINAL_3406_CPU_SCOPE")
    lineage_raw,lineage_pin=read(STAGE/"lineage-private.json",LINEAGE_SHA,14975)
    lineage=json.loads(lineage_raw)
    pin=result["native_index"]
    index_raw,index_pin=read(STAGE/"native64-index-private.json",pin["SHA256"],pin["bytes"])
    index=json.loads(index_raw)
    need(lineage["physical_rows"]==800 and lineage["request"]==index["request"]==INDEX_SCOPE
         and index["GPU"]==index["optimizer_updates"]==0 and index["count"]==64,"G45_EXACT_LINEAGE_AND_64")
    projected={r:[p["ordinal"] for p in rows] for r,rows in index["rank"].items()}
    need(projected==lineage["rank_first64"],"G45_BOTH_INDEPENDENT_64_REPRESENTATIONS")
    base=lineage["raw_ordinals_by_dataset_index"]
    need(len(base)==800 and all(type(x) is int for x in base) and set(base)==set(range(1,801))
         and sha256(canonical([x-1 for x in base])).hexdigest()==ZERO_BASED_LIVE_ORDER_SHA,
         "G45_ACTUAL_LIVE_FULL_800_ARROW_ORDER")
    return base,projected,dict(lineage=lineage_pin,native_index=index_pin,index_result=result_got,index_intent=intent_got,
                              original_independent_3406=terminal_pin)


def comparison(draw,arrow,expected):
    need(type(draw) is list and len(draw)==SIZE and all(type(v) is int for v in draw)
         and set(draw)==set(range(SIZE)) and set(expected)=={str(k) for k in range(WORLD)}
         and all(len(slots)==COMPARE for slots in expected.values()),"G45_EXACT_PROJECTION_DENOMINATOR")
    actual={}
    first=None
    correct=0
    for rank in range(WORLD):
        indexes=draw[rank::WORLD]
        need(len(indexes)==PER_RANK,"G45_100_CONSUMED_LOGICAL_SLOTS_PER_RANK")
        values=[arrow[i] for i in indexes[:COMPARE]]
        actual[str(rank)]=values
        for position,(lhs,rhs) in enumerate(zip(values,expected[str(rank)],strict=True),1):
            if lhs==rhs:
                correct+=1
            elif first is None:
                first={"rank":rank,"microbatch":position,"optimizer_boundary":1 if position<=4 else 2}
    return {"logical_rows":SIZE,"rank_denominator":WORLD,"slots_per_rank":PER_RANK,"compared_slots":64,
            "matching_slots":correct,"first_difference":first,
            "logical_permutation_SHA256":sha256(canonical(draw)).hexdigest(),
            "observed_first64_SHA256":sha256(canonical(actual)).hexdigest(),
            "expected_first64_SHA256":sha256(canonical(expected)).hexdigest()}


def step_guard(epoch,request):
    require_env=os.environ
    num=require_env.get("SLURM_STEP_ID","")
    need(re.fullmatch(r"[0-9a-f]{32}",request) and math.isfinite(epoch) and 0<=time.time()-epoch<120
         and require_env.get("SLURM_JOB_ID")==PARENT and re.fullmatch(r"[0-9]+",num)
         and num not in {"3406","3464","3465","3466"} and socket.gethostname().split(".")[0]==NODE
         and require_env.get("SLURM_CPUS_PER_TASK")=="4","G45_NEW_EXACT_ZERO_GRES_NUMERIC_SCOPE")
    need(require_env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(require_env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not any(k in require_env.get("SLURM_TRES_PER_TASK","").lower() for k in ("gres/","gpu","dcu"))
         and all(require_env.get(k)=="-1" for k in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")),
         "G45_ZERO_VISIBLE_DCU")
    groups=Path("/proc/self/cgroup").read_text("ascii").splitlines()
    need(any({"job_"+PARENT,"step_"+num}<=set(row.split(":",2)[-1].split("/")) for row in groups),"G45_REAL_NUMERIC_STEP_CGROUP")
    return PARENT+"."+num


def run(args,result):
    result["step"]=step_guard(args.epoch,args.request)
    raw,source_pin=read(SHARD,SOURCE_SHA,3553)
    runtime_ast=check_source(raw)
    plan,plan_pin=read(PLAN,PLAN_SHA)
    facts=check_plan(plan)
    arrow,expected,pins=fixed_private()
    result.update(source=source_pin,source_iter_AST_SHA256=runtime_ast,plan=plan_pin,exact_private_pins=pins,
                  model_scope=facts,old_3465_source_receipt_SHA256=OLD_WITNESS_SHA,
                  observed_3465_arrow_SHA256=ZERO_BASED_LIVE_ORDER_SHA)
    import torch
    need(torch.version.hip is not None and torch.version.cuda is None and not torch.cuda.is_initialized(),
         "G45_ACTUAL_PRODUCTION_HIP_TORCH_IN_CPU_ONLY_PROCESS")
    provider=Path(torch.__file__).resolve(strict=True)
    _,torch_pin=read(provider,cap=4<<20)
    result["torch_provider"]=torch_pin
    result["torch_version"]=str(torch.__version__)
    rng=torch.random.get_rng_state().clone()
    python=random.getstate()
    generator=torch.Generator(device="cpu")
    generator.manual_seed(42)
    draw=torch.randperm(SIZE,generator=generator,device="cpu").tolist()
    result.update(comparison(draw,arrow,expected))
    untouched=torch.equal(rng,torch.random.get_rng_state()) and python==random.getstate()
    result.update(global_CPU_and_python_RNG_unchanged=untouched)
    need(untouched and not torch.cuda.is_initialized() and time.time()<args.epoch+120,"G45_RNG_GPU_AND_EPOCH")
    result["completed"]=True


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch",type=float,required=True)
    p.add_argument("--request",required=True)
    args=p.parse_args()
    out=dict(request=args.request,epoch=args.epoch,completed=False,actual_training=False,
             native_live_sampler_iterated=False,training_view_body_read=False,model_weight_io=False,
             actual_GPU=0,optimizer_updates=0,logical_rows=800,planned_comparisons=64)
    def expired(*_):
        raise TimeoutError("G45_NEW_120S_EPOCH")
    signal.signal(signal.SIGALRM,expired)
    try:
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(120,args.epoch+120-time.time())))
        run(args,out)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as e:
        code=str(e)
        out.update(error_class=type(e).__name__,predicate=code if re.fullmatch(r"G45_[A-Z0-9_]+",code) else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False),flush=True)
    raise SystemExit(0 if out["completed"] else 1)
