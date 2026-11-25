"""Produce the exact 64 planned native input signatures using processor-only CPU."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from native_inputs import build_index
from prepare import NEW,REQUEST,ROOT,digest,emit_file,encoded,need,pin,read


SOURCE=Path(__file__).resolve().parent
PROCESSOR=ROOT/"training-gemma-processor-assets-v1/models/gemma-3-4b-it"
ASSET=ROOT/"training-gemma-usgs72-fresh-initial-domain25-v1"
FRAMEWORK=ASSET/"framework"
PROCESSOR_PIN="eaab7b89d89e5cbe27391c3601ab36acfa5aa35f8c0ea18f6f1730d5424d228b"
FRAMEWORK_PIN="40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329"
WORK=240
VERSIONS={"torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580","transformers":"5.12.1","ms-swift":"4.4.2",
          "Pillow":"11.3.0","tokenizers":"0.22.2"}


def verify_inventory(check):
    processor=json.loads(read(SOURCE/"MODEL_PROCESSOR_FILES.json",PROCESSOR_PIN,4096))["files"]
    need(len(processor)==14 and {path.name for path in PROCESSOR.iterdir()}==set(processor),"R39_PROCESSOR_ONLY_EXACT_CLOSURE")
    for name,expected in processor.items():
        check()
        need(Path(name).name==name and not name.endswith((".pt",".bin",".safetensors")),"R39_CPU_NO_MODEL_PAYLOAD")
        read(PROCESSOR/name,expected["SHA256"],64<<20,expected["bytes"])
    framework=json.loads(read(ASSET/"fresh-framework-files.json",FRAMEWORK_PIN))["files"]
    need(len(framework)==396 and {name:importlib.metadata.version(name) for name in VERSIONS}==VERSIONS,
         "R39_ACTUAL_845_SOURCE_AND_VERSIONS")
    for name,expected in framework.items():
        check()
        rel=Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts,"R39_FRAMEWORK_SOURCE_PATH")
        read(FRAMEWORK/rel,expected["SHA256"],16<<20,expected["bytes"])
    return framework


def placement(args):
    env=os.environ
    numeric=env.get("SLURM_STEP_ID","")
    step="1103606."+numeric
    need(env.get("SLURM_JOB_ID")=="1103606" and numeric.isascii() and numeric.isdecimal()
         and socket.gethostname().split(".")[0]=="f13r1n12" and env.get("SLURM_CPUS_PER_TASK")=="4"
         and len(os.sched_getaffinity(0))>=4 and args.epoch<=time.time()<args.epoch+WORK
         and "torch" not in sys.modules,"R39_CPU_EXACT_NUMERIC_ORIGINAL_STEP")
    need(env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(env.get(n,"0") in ("","0","(null)") for n in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not any(v in env.get("SLURM_TRES_PER_TASK","").lower() for v in ("gpu","dcu"))
         and all(env.get(n)=="-1" for n in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"))
         and all(env.get(n)=="1" for n in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE")),
         "R39_EXACT_CPU_OFFLINE_INDEPENDENT_TRES_REQUIRED")
    return step


def execute(args):
    step=placement(args)
    limit=time.monotonic()+args.epoch+WORK-time.time()
    def check():
        need(time.monotonic()<limit and time.time()<args.epoch+WORK,"R39_NATIVE_INDEX_CPU_EPOCH")
    def timeout(*_):
        raise TimeoutError("R39_NATIVE_INDEX_CPU_TIMEOUT")
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(.001,limit-time.monotonic()))
    report={"request":REQUEST,"step":step,"planned_signatures":64,"actual_optimizer_updates":0,"GPU":0,
            "epoch":args.epoch,"passed":False,"phase":"private_view"}
    def offline(event,values):
        if event in ("socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"):
            raise RuntimeError("R39_NO_NETWORK_OR_SUBPROCESS")
        if event=="open" and values and type(values[0]) in (str,bytes):
            name=Path(os.fsdecode(values[0]))
            if name.suffix in (".safetensors",".pt",".pth",".bin") or set(name.parts)&{"dev","blind"} and name.is_absolute() and str(name).startswith(str(ROOT)):
                raise RuntimeError("R39_CPU_WEIGHT_AND_HELDOUT_BODY_FORBIDDEN")
    sys.addaudithook(offline)
    try:
        lineage=json.loads(read(NEW/"lineage-private.json",args.lineage_sha256,65536))
        view=read(NEW/"train800-swift.jsonl",lineage["view"]["SHA256"],size=lineage["view"]["bytes"])
        report["phase"]="exact_runtime_processor"
        framework=verify_inventory(check)
        import torch
        need(not torch.cuda.is_initialized(),"R39_NO_CPU_GPU_CONTEXT")
        def deny(*_,**__):
            raise RuntimeError("R39_GPU_CONTEXT_FORBIDDEN")
        torch.cuda._lazy_init=deny
        torch.set_num_threads(2)
        from vlm.runtime import first_update_audit
        path=Path(first_update_audit.__file__).resolve()
        expected=framework["vlm/runtime/first_update_audit.py"]
        need(path==FRAMEWORK/"vlm/runtime/first_update_audit.py","R39_EXACT_FINGERPRINT_PROVIDER")
        read(path,expected["SHA256"],size=expected["bytes"])
        from swift.model import get_model_processor
        from swift.template import get_template
        model,processor=get_model_processor(str(PROCESSOR),model_type="gemma3_vision",load_model=False,
            download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
        need(model is None,"R39_INDEX_MUST_NOT_LOAD_ORIGINAL_MODEL")
        template=get_template(processor,template_type="gemma3_vision",max_length=512,max_pixels=262144,
                              padding_side="right",padding_free=False,truncation_strategy="raise")
        template.set_mode("train")
        report["phase"]="actual_native_64_signature_index"
        index=build_index(view,lineage,template,check=check)
        check()
        raw=encoded(index)
        emit_file(NEW/"native64-index-private.json",raw)
        report.update(passed=True,native_index=pin(raw),actual_signatures=64,new_unique_signatures=12,
                      original_weights_opened=False,recall_blind_data_loaded=False)
    except (OSError,ValueError,ImportError,KeyError,TypeError,RuntimeError,AttributeError) as error:
        report["failure_class"]=type(error).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        emit_file(NEW/"INDEX_RESULT.json",encoded(report))
    return report["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch",required=True,type=float)
    p.add_argument("--lineage-sha256",required=True)
    a=p.parse_args()
    raise SystemExit(not execute(a))
