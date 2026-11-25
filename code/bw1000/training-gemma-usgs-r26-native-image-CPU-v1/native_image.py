"""Run Gemma's original image template and collator on twelve new CPU-only rows."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import sys
import time

from admission import (FRAMEWORK_MANIFEST, LEGACY_AUDIT, LEGACY_EXPOSURE, PAIRS,
    PROCESSOR_MANIFEST, ROWS, VERSIONS, WORK_SECONDS, admit, encoded, need, pin, read)


SOURCE=Path("/audit-src")
DATA=Path("/r26-train")
OUTPUT=Path("/audit-out")
FRAMEWORK=Path("/input-framework")
PROCESSOR=Path("/gemma-processor")
GATE_ENV=("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")


def placement(parent,node,epoch,now,env,host,affinity):
    step=env.get("SLURM_STEP_ID","")
    tres=env.get("SLURM_TRES_PER_TASK","").lower()
    need(parent==env.get("SLURM_JOB_ID") and re.fullmatch("[0-9]+",parent) and re.fullmatch("[0-9]+",step)
         and node==host and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",node)
         and env.get("SLURM_CPUS_PER_TASK")=="4" and affinity>=4,"EXACT_NUMERIC_CPU_STEP")
    need(env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(env.get(name,"") in ("","0","(null)") for name in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not any(device in tres for device in ("gpu","dcu"))
         and all(env.get(key)=="-1" for key in GATE_ENV)
         and all(env.get(key)=="1" for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE")),
         "ZERO_VISIBLE_DEVICE_OFFLINE_CPU")
    need(math.isfinite(epoch) and 0<epoch<=now<epoch+WORK_SECONDS,"SINGLE_CPU_EPOCH")
    return parent+"."+step


def package(source_sha):
    members={"admission.py","native_image.py","audit_gemma_training_native.py","build_gemma_train_exposure.py",
             "MODEL_PROCESSOR_FILES.json"}
    manifest=json.loads(read(SOURCE/"SOURCE_FILES.json",None,source_sha,16384))
    need(set(manifest["files"])==members,"EXACT_CPU_SOURCE_MEMBERS")
    for name,record in manifest["files"].items():
        read(SOURCE/name,record["bytes"],record["SHA256"])
    read(SOURCE/"audit_gemma_training_native.py",*LEGACY_AUDIT)
    read(SOURCE/"build_gemma_train_exposure.py",*LEGACY_EXPOSURE)
    return manifest


def validate_assets(check):
    mp=json.loads(read(SOURCE/"MODEL_PROCESSOR_FILES.json",*PROCESSOR_MANIFEST))["files"]
    need(len(mp)==14 and {path.name for path in PROCESSOR.iterdir()}==set(mp),"EXACT_PROCESSOR_ONLY_INVENTORY")
    for name,record in mp.items():
        check()
        need(Path(name).name==name and not name.endswith((".safetensors",".pt",".bin")),"PROCESSOR_METADATA_ONLY")
        read(PROCESSOR/name,record["bytes"],record["SHA256"],64<<20)
    manifest=json.loads(read(Path("/qualification/fresh-framework-files.json"),*FRAMEWORK_MANIFEST))["files"]
    need(len(manifest)==396,"EXACT_ORIGINAL_845_FRAMEWORK")
    for name,record in manifest.items():
        check()
        relative=Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts,"FRAMEWORK_MEMBER_PATH")
        read(FRAMEWORK/relative,record["bytes"],record["SHA256"],16<<20)
    need({name:importlib.metadata.version(name) for name in VERSIONS}==VERSIONS,"ORIGINAL_845_RUNTIME_VERSIONS")
    return mp


def tensor_image(value):
    if isinstance(value,(tuple,list)):
        return any(tensor_image(item) for item in value)
    shape=getattr(value,"shape",())
    return len(shape)>=3 and all(dimension>0 for dimension in shape) and getattr(getattr(value,"device",None),"type",None)=="cpu"


class ImageTemplate:
    def __init__(self, template, progress, key):
        self.template=template
        self.progress=progress
        self.key=key

    def encode(self, row):
        need(len(row["images"])==1 and row["images"][0].startswith("/r26-train/train/media/"),
             "ONE_BOUND_REAL_IMAGE_PER_ENCODE")
        self.progress[self.key+"_attempts"]+=1
        value=self.template.encode(row)
        need(tensor_image(value.get("pixel_values")),"REAL_CPU_ENCODED_IMAGE")
        self.progress[self.key+"_completed"]+=1
        return value

    def data_collator(self, rows):
        self.progress["collator_attempts"]+=1
        value=self.template.data_collator(rows)
        need(tensor_image(value.get("pixel_values")),"REAL_CPU_COLLATED_IMAGE")
        self.progress["collator_completed"]+=1
        return value


def save(path,value):
    raw=encoded(value)
    need(len(raw)<=65536,"PRIVATE_METADATA_EXTENT")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(raw)


def expire(*_):
    raise TimeoutError("CPU_WORK_DEADLINE")


def execute(args):
    step=placement(args.parent,args.node,args.epoch,time.time(),os.environ,socket.gethostname().split(".")[0],
                   len(os.sched_getaffinity(0)))
    need(re.fullmatch(r"[0-9a-f]{32}",args.request) and re.fullmatch(r"[0-9a-f]{40}",args.publication)
         and re.fullmatch(r"[0-9a-f]{64}",args.source_manifest_sha256)
         and "torch" not in sys.modules,"FRESH_BOUND_CPU_REQUEST")
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:4]))
    need(len(os.sched_getaffinity(0))==4,"BOUND_FOUR_CPU_AFFINITY")
    remaining=args.epoch+WORK_SECONDS-time.time()
    end=time.monotonic()+remaining
    def check():
        need(time.monotonic()<end and time.time()<args.epoch+WORK_SECONDS,"CPU_WORK_DEADLINE")
    def offline(event,_):
        if event in ("socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"):
            raise RuntimeError("OFFLINE_CPU_WORKER")
    sys.addaudithook(offline)
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
    progress={name:0 for name in ("full_attempts","full_completed","capped_attempts","capped_completed",
                                  "collator_attempts","collator_completed","verified_rows")}
    report={"schema":"GEMMA_R27_NATIVE_TRAIN_IMAGE_CPU_V1","request":args.request,"publication":args.publication,
            "step":step,"node":args.node,"epoch":args.epoch,"deadline":args.epoch+WORK_SECONDS,
            "source_manifest_SHA256":args.source_manifest_sha256,"planned_rows":ROWS,"planned_pairs":PAIRS,
            "training_ready":False,"actual_optimizer_updates":0,"model_loaded":False}
    need(OUTPUT.is_dir() and not any(OUTPUT.iterdir()),"EXCLUSIVE_CPU_RESULT_DIRECTORY")
    save(OUTPUT/"INTENT.json",report)
    phase="source_assets"
    records=[]
    try:
        package(args.source_manifest_sha256)
        data=admit(DATA,check)
        processor_inventory=validate_assets(check)
        phase="native_processor"
        import torch
        need(torch.cuda.is_initialized() is False,"DEVICE_CONTEXT_ALREADY_INITIALIZED")
        def forbid(*_,**__):
            raise RuntimeError("GPU_CONTEXT_FORBIDDEN")
        torch.cuda._lazy_init=forbid
        torch.set_num_threads(2)
        random.seed(42)
        torch.random.default_generator.manual_seed(42)
        sys.path[:0]=[str(SOURCE),str(FRAMEWORK)]
        from vlm.runtime import first_update_audit
        from swift.model import get_model_processor
        from swift.template import get_template
        from audit_gemma_training_native import audit_rows
        need(Path(first_update_audit.__file__).resolve()==FRAMEWORK/"vlm/runtime/first_update_audit.py",
             "ACTUAL_ORIGINAL_FRAMEWORK_FINGERPRINT_PROVIDER")
        model,processor=get_model_processor(str(PROCESSOR),model_type="gemma3_vision",load_model=False,
            download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
        need(model is None,"MODEL_WEIGHTS_MUST_NOT_LOAD")
        options=dict(template_type="gemma3_vision",max_pixels=262144,padding_side="right",
                     padding_free=False,truncation_strategy="raise")
        full=get_template(processor,max_length=None,**options)
        capped=get_template(processor,max_length=512,**options)
        full.set_mode("train")
        capped.set_mode("train")
        declared=capped.template_meta.suffix
        need(declared and declared==full.template_meta.suffix,"NATIVE_SUPERVISED_SUFFIX")
        suffix=capped._encode_context_list(declared,[1.]*len(declared))[0]
        need(suffix==full._encode_context_list(declared,[1.]*len(declared))[0],"IDENTICAL_UNCAPPED_SUFFIX")
        generation=processor_inventory["generation_config.json"]
        configured=json.loads(read(PROCESSOR/"generation_config.json",generation["bytes"],generation["SHA256"]))["eos_token_id"]
        eos={configured} if type(configured) is int else set(configured)
        eos.add(processor.tokenizer.eos_token_id)
        need(all(type(token) is int and token>=0 for token in eos),"PINNED_NATIVE_EOS")
        rows=[]
        for row,source_pin in data["entries"].values():
            original_image=row["images"][0]
            rows.append((encoded({**row,"images":[str(DATA/original_image)]}),source_pin,
                         data["pairs"][row["id"]],data["files"][original_image]["SHA256"]))
        def mark(record):
            _,source_pin,pair,image_pin=rows[record["ordinal"]-1]
            need(record["shifted_supervised_tokens"]>0,"NONEMPTY_SHIFTED_SUPERVISION")
            records.append({"row_ordinal":record["ordinal"],"original_row_SHA256":source_pin,
                "image_SHA256":image_pin,**pair,"tokens":record["tokens"],
                "shifted_supervised_tokens":record["shifted_supervised_tokens"],
                "encoded_SHA256":record["encoded"]["SHA256"],"collated_SHA256":record["collated"]["SHA256"]})
            progress["verified_rows"]+=1
        phase="native_image_encoding"
        audit_rows([row[0] for row in rows],ImageTemplate(full,progress,"full"),ImageTemplate(capped,progress,"capped"),
                   first_update_audit.fingerprint,list(suffix),eos,check,mark)
        need(len(records)==progress["verified_rows"]==progress["full_completed"]==progress["capped_completed"]
             ==progress["collator_completed"]==ROWS and torch.cuda.is_initialized() is False,"COMPLETE_NATIVE_CPU_IMAGE_DENOMINATOR")
        capture=save(OUTPUT/"PRIVATE_ROW_IDENTITIES.json",{"rows":records})
        report.update(state="NATIVE_IMAGE_CPU_PASS",passed=True,private_row_identities=capture,
                      stable_pair_ids=PAIRS,processor_metadata_files=14,verified_framework_files=396,
                      planned_missing_rows=0,native_max_tokens=max(row["tokens"] for row in records))
    except (OSError,ValueError,TypeError,KeyError,AttributeError,ImportError,ArithmeticError,RuntimeError) as error:
        report.update(state="NATIVE_IMAGE_CPU_FAILED",passed=False,error_type=type(error).__name__,
                      planned_missing_rows=ROWS-progress["verified_rows"])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report.update(phase=phase,progress=progress)
        save(OUTPUT/"RESULT.json",report)
    return int(not report["passed"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","node","request","publication","source-manifest-sha256"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--epoch",required=True,type=float)
    return execute(p.parse_args())


if __name__=="__main__":
    sys.exit(main())
