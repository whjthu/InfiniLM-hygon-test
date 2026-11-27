"""Bind one recovered ABO cache to the exact common CLI and observe actual Muon steps."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import runpy
import sys

from contract import CACHE, CACHE_SHA, CACHE_BYTES, CLI_SHA, LAUNCH_SHA, MODEL, NATIVE_ALIAS, REQUEST, RUN, load, need, read, save

ORIGINAL_CACHE_SHA = "9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b"
BASE_RATE = 1e-5


def bind(builder, case, launch, mapping):
    need(case=="T09.sft" and sha256(launch).hexdigest()==LAUNCH_SHA,
         "ABO_EXACT_ORIGINAL_A100_RECIPE")
    original=json.loads(launch)
    config=original["config"]
    old_path=config["training_cache"]
    needed={old_path:str(CACHE),original["source"]:str(NATIVE_ALIAS),original["weights"]:str(MODEL)}
    need(mapping==needed and config["training_cache_sha"]==ORIGINAL_CACHE_SHA
         and config["learning_rate"]==BASE_RATE and config["warmup_steps"]==5 and config["max_steps"]==20
         and config["seed"]==20260911 and config["checkpoint_steps"]==[10,20]
         and config["validate_recovery"] is False and "resume_from_checkpoint" not in config,
         "ABO_SOLE_DOMAIN_DATASET_INPUT_OVERRIDE")
    plan=builder(case,launch,mapping)
    rebound=dict(config,training_cache=str(CACHE))
    need(plan["config"]==rebound and plan["case"]==case and plan["a100_launch_sha256"]==LAUNCH_SHA
         and (plan["model"],plan["action"],plan["cards"],plan["backend_entrypoint"])==
         ("hunyuanvideo-1.5","train",1,"gencuda")
         and plan["weights"]==str(MODEL) and plan["source"]==str(NATIVE_ALIAS),
         "ABO_COMMON_GENERATION_PLAN_DRIFT")
    return dict(plan,config=dict(rebound,training_cache_sha=CACHE_SHA))


class MuonWitness:
    def __init__(self, backend):
        self.backend=Path(backend)
        self.steps=[]
        self.optimizer=None

    def observe(self, optimizer, _args, _kwargs, *, frame=None):
        frame=sys._getframe(1) if frame is None else frame
        while frame is not None:
            if frame.f_code.co_name=="train_step" and Path(frame.f_code.co_filename)==self.backend:
                break
            frame=frame.f_back
        if frame is None:
            return
        step=frame.f_locals.get("global_step")
        rates=[group["lr"] for group in optimizer.param_groups]
        need(frame.f_locals.get("optimizer") is optimizer and type(optimizer).__name__=="Muon"
             and (self.optimizer is None or self.optimizer is optimizer)
             and type(step) is int and step==len(self.steps)+1 and step<=25
             and 0<len(rates)<=16 and all(type(rate) in (int,float) and math.isfinite(rate) and rate>0 for rate in rates),
             "ABO_EXACT_ACTUAL_MUON_POSITIVE_RATE_AND_STEP")
        self.optimizer=optimizer
        self.steps.append({"global_step":step,"rates":rates})


def run_common(torch, argv, ctx, cli_path, backend):
    cli_path=Path(cli_path)
    raw_source, source=read(cli_path, 1<<20)
    need(source["SHA256"]==CLI_SHA and sha256(raw_source).hexdigest()==CLI_SHA,"ABO_EXACT_DEPLOYED_COMMON_CLI")
    _,cache=read(CACHE,CACHE_BYTES)
    need(cache["bytes"]==CACHE_BYTES and cache["SHA256"]==CACHE_SHA,"ABO_EXACT_CACHE_CURRENT_BYTES")
    mapper, mapper_pin=load(RUN/"assets.json")
    module=runpy.run_path(str(cli_path),run_name="_pinned_abo_common_training")
    entry=module["main"]
    namespace=entry.__globals__
    builder=namespace["build_generation_plan"]
    calls=0
    def build_once(case,launch,mapping):
        nonlocal calls
        need(calls==0 and mapping==mapper,"ABO_EXACT_ONCE_COMMON_PLAN")
        plan=bind(builder,case,launch,mapping)
        calls+=1
        save(RUN/"domain-cache-ingress.json",dict(ctx,process_PID=os.getpid(),cache=cache,
            asset_map=mapper_pin,source=source,original_launch_SHA256=LAUNCH_SHA,
            unique_training_rows=1,plan_calls=calls,original_plan_only_difference=["training_cache","training_cache_sha"],
            selected_case="T09.sft",frozen_warmup=5,frozen_measured=20,original_first_update_replay=False,
            fresh_flow_matching_noise_native=True))
        return plan
    observer=MuonWitness(backend)
    from torch.optim.optimizer import register_optimizer_step_post_hook
    need(callable(register_optimizer_step_post_hook) and torch.__name__=="torch","ABO_EXACT_NATIVE_OPTIMIZER_HOOK")
    handle=register_optimizer_step_post_hook(observer.observe)
    namespace["build_generation_plan"]=build_once
    returned=None
    try:
        returned=entry(argv[1:])
        if returned==0:
            need(calls==1 and len(observer.steps)==25,"ABO_FULL_POSITIVE_MUON_STEP_DENOMINATOR")
        return returned
    finally:
        namespace["build_generation_plan"]=builder
        handle.remove()
        save(RUN/"domain-muon.json",dict(ctx,process_PID=os.getpid(),planned=25,
             observed=len(observer.steps),missing=25-len(observer.steps),steps=observer.steps,
             common_return=returned,unique_training_rows=1,performance_qualified=False))


def progress(work,ctx):
    planned=25
    try:
        witness, pin=load(Path(work)/"domain-muon.json")
    except FileNotFoundError:
        return {"planned":planned,"observed":None,"missing":None,"evidence":"MISSING"}
    count=witness["observed"]
    need(all(witness[key]==ctx[key] for key in ("request","epoch","step"))
         and type(count) is int and 0<=count<=planned and witness["planned"]==planned
         and witness["missing"]==planned-count and len(witness["steps"])==count
         and [s["global_step"] for s in witness["steps"]]==list(range(1,count+1))
         and all(row["rates"] and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in row["rates"])
                 for row in witness["steps"]), "ABO_REAL_POSITIVE_RATE_PARTIAL_DENOMINATOR")
    return {"planned":planned,"observed":count,"missing":planned-count,
            "evidence":"BOUND_ACTUAL_MUON_POST_HOOK","pin":pin}


def review(work, ctx):
    work=Path(work)
    ingress, ingress_pin=load(work/"domain-cache-ingress.json")
    witness, witness_pin=load(work/"domain-muon.json")
    need(all(ingress[key]==witness[key]==ctx[key] for key in ("request","epoch","step"))
         and ingress["process_PID"]==witness["process_PID"] and ingress["cache"]["SHA256"]==CACHE_SHA
         and ingress["cache"]["bytes"]==CACHE_BYTES and ingress["source"]["SHA256"]==CLI_SHA
         and ingress["unique_training_rows"]==witness["unique_training_rows"]==1
         and ingress["plan_calls"]==1 and ingress["original_first_update_replay"] is False
         and progress(work,ctx)["observed"]==25 and witness["common_return"]==0,
         "ABO_REAL_COMPLETE_POSITIVE_RATE_NATIVE_INGRESS")
    return {"ingress":ingress_pin,"optimizer":witness_pin,"actual_positive_LR_Muon_steps":25,"unique_training_rows":1,
            "domain_effect_qualified":False,"performance_qualified":False}
