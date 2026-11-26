"""Observe the exact Swift label-prepare path with one original CPU-collated example."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
from types import SimpleNamespace


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from alignment import alignment, digest, positions, require, shifted_one_is_rejected


BASE=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
NATIVE=BASE/"results/training-gemma-usgs-r120-compute-source-ready-v1/src"
SWIFT=Path("/swift-runtime/site/swift")
GEMMA="c7938a2c0fdf04c55da2dbe8e03d12d22501f06f5db917bb1e7ce0189835418e"
STEP_BUDGET=120
ORIGINAL="e49db3405b64892b1042e62ab79ceb36e6aebcc593d7d5438e16ee48f393c5be"
AST_EVIDENCE="177457577ea5a3cb2aae3d1b02a86475d55ed3f51399e8edeef86290baecf52a"
NATIVE_MANIFEST="4bbd7b6ea7252543db84711ec71f816eabe55bed568b6c0e86f4541f91622690"
PINS={
    "mixin.py":(64818,"6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9"),
    "seq2seq_trainer.py":(12688,"28fe9cb927390ad423add8b98411a90fb2d1c801ead124b354802bdd0779ecf6"),
    "arguments.py":(16906,"e3fd302d795e2ab09ef07124ad6c92991b9c0be486c8c12c0ef6e648eca38683"),
    "trainer.py":(2784,"32b4140841d1fb8f066fca841751354645d8c29eb95f07fc0acb0a8e99813a62")}
MODES=(False,True,None)


def checked(path,size,expected=None,cap=None):
    require(path.is_file() and not path.is_symlink(),"EXACT_EXISTING_REGULAR_INPUT")
    with path.open("rb") as f:
        raw=f.read((cap or size)+1)
    require((size is None or len(raw)==size) and len(raw)<=(cap or size)
            and (expected is None or sha256(raw).hexdigest()==expected),"FROZEN_INPUT_SHA_OR_SIZE")
    return raw


def gate(step_epoch):
    env=os.environ
    stage=env.get("SLURM_STEP_ID","")
    require(env.get("SLURM_JOB_ID")=="1103606" and re.fullmatch("[0-9]+",stage)
            and socket.gethostname().split(".")[0]=="f13r1n12"
            and env.get("SLURM_CPUS_PER_TASK")=="4" and len(os.sched_getaffinity(0))>=4
            and all(env.get(k)=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
            and all(env.get(k)=="1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"))
            and env.get("SLURM_STEP_GPUS","") in ("","none","N/A","(null)")
            and all(env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_PER_TASK","SLURM_GPUS_ON_NODE"))
            and not re.search("gpu|dcu",env.get("SLURM_TRES_PER_TASK",""),re.I)
            and 0<=time.time()-step_epoch<STEP_BUDGET,"NUMERIC_4CPU_ZERO_GRES_ORIGINAL_IMAGE_REQUIRED")
    return "1103606."+stage


def bounded_audit(event,values):
    if event in {"socket.connect","socket.getaddrinfo","subprocess.Popen","os.system"}:
        raise RuntimeError("NO_NETWORK_OR_SUBPROCESS")
    if event=="open" and values and isinstance(values[0],(str,bytes)):
        p=Path(os.fsdecode(values[0]))
        heldout_path = bool({"blind","dev"} & set(p.parts)) and (
            not p.is_absolute() or p.is_relative_to(BASE))
        if p.suffix in {".safetensors",".pt",".pth",".bin"} or heldout_path:
            raise RuntimeError("NO_WEIGHTS_OR_HELDOUT_MEDIA")


def source_bind(manifest_sha):
    require(manifest_sha==NATIVE_MANIFEST,"ORIGINAL_4177_EFFECTIVE_MANIFEST_PIN_ONLY")
    mf=json.loads(checked(NATIVE/"SOURCE_FILES.json",3734,manifest_sha))["files"]
    require(len(mf)==32,"EXACT_4177_EFFECTIVE_MEMBER_COUNT")
    for name,identity in mf.items():
        require(Path(name).name==name and not name.startswith("."),"FROZEN_SOURCE_MEMBER_NAME")
        checked(NATIVE/name,identity["bytes"],identity["SHA256"],65536)
    sys.path.insert(0,str(NATIVE))
    for name,(length,sha) in PINS.items():
        checked(SWIFT/"trainers"/name,length,sha)


def tensor_metadata(t):
    return dict(shape=list(t.shape),dtype=str(t.dtype),device=t.device.type)


def first_collated(torch):
    from cpu_index import verify_inventory,PROCESSOR
    from native_worker import trust_cpu_stage,NEW,read
    from gemma_usgs_r39_inputs import input_signature,batch_gate_metadata
    from swift.model import get_model_processor
    from swift.template import get_template

    stage,index=trust_cpu_stage()
    verify_inventory(lambda:None)
    lineage=json.loads(read(NEW/"lineage-private.json",stage["lineage"]["SHA256"],size=stage["lineage"]["bytes"]))
    index_doc=json.loads(read(NEW/"native64-index-private.json",index["native_index"]["SHA256"],size=index["native_index"]["bytes"]))
    raw=read(NEW/"train800-swift.jsonl",stage["view"]["SHA256"],size=stage["view"]["bytes"]).splitlines(keepends=True)
    source=index_doc["rank"]["0"][0]
    ordinal=source["ordinal"]
    require(len(raw)==800 and type(ordinal) is int and 1<=ordinal<=800 and source["forward"]==1
            and lineage["rank_first64"]["0"][0]==ordinal
            and sha256(raw[ordinal-1]).hexdigest()==source["line_SHA256"]==lineage["selected_line_SHA256"][str(ordinal)],
            "EXACT_RANK0_FIRST_NATIVE_SLOT")
    record=json.loads(raw[ordinal-1])
    model,processor=get_model_processor(str(PROCESSOR),model_type="gemma3_vision",load_model=False,
        download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
    require(model is None,"NO_MODEL_WEIGHTS_LOADED")
    template=get_template(processor,template_type="gemma3_vision",max_length=512,max_pixels=262144,
         padding_side="right",padding_free=False,truncation_strategy="raise")
    template.set_mode("train")
    batch=template.data_collator([template.encode({k:record[k] for k in ("messages","images")})])
    signature=input_signature(batch)
    require(signature==source["input_SHA256"],"EXACT_3406_FULL_COLLATOR_SIGNATURE")
    return batch,template,processor,input_signature,batch_gate_metadata,dict(index=index["native_index"],
           private_source_line_SHA256=source["line_SHA256"],collator_signature_SHA256=signature,
           full_batch=batch_gate_metadata(batch,torch.int64))


def provider(torch,template,processor):
    from swift.trainers import Seq2SeqTrainer
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    methods={name:getattr(Seq2SeqTrainer,name) for name in ("_prepare_inputs","get_use_logits_to_keep","prepare_logits_to_keep")}
    expected={"_prepare_inputs":("seq2seq_trainer.py",101),"get_use_logits_to_keep":("mixin.py",210),
              "prepare_logits_to_keep":("mixin.py",1179)}
    pins={}
    for name,method in methods.items():
        file,line=expected[name]
        require(Path(inspect.getfile(method)).resolve()==SWIFT/"trainers"/file
                and inspect.getsourcelines(method)[1]==line,"ACTUAL_IMPORTED_SWIFT_CALLABLE_IDENTITY")
        pins[name]=dict(line=line, SHA256=sha256("".join(inspect.getsourcelines(method)[0]).encode()).hexdigest())
    forward=Gemma3ForConditionalGeneration.forward
    path=Path(inspect.getfile(inspect.unwrap(forward))).resolve()
    checked(path,None,GEMMA,2<<20)
    require("logits_to_keep" in inspect.signature(forward).parameters,
            "FROZEN_GEMMA_FORWARD_CAPABILITY_REQUIRED")
    require(template.sequence_parallel_size==1,"ORIGINAL_NO_SEQUENCE_PARALLEL_TEMPLATE")
    # The exact forward callable is visible but unreachable. The proxy carries
    # actual processor metadata and intentionally has no model state.
    meta=getattr(processor,"model_meta",getattr(template,"model_meta",None))
    info=getattr(processor,"model_info",getattr(template,"model_info",None))
    require(meta is not None and meta.is_multimodal is True and info is not None and info.is_moe_model is False,
            "REAL_PROCESSOR_MULTIMODAL_DENSE_MODEL_METADATA_REQUIRED")
    model=SimpleNamespace(forward=forward,model_meta=meta,model_info=info)
    return Seq2SeqTrainer,methods,model,pins


def native_prepare(torch,trainer_type,methods,model,template,batch,option):
    trainer=object.__new__(trainer_type)
    trainer.args=SimpleNamespace(use_logits_to_keep=option,tuner_backend="peft",device=torch.device("cpu"))
    trainer.template=template
    trainer.model=model
    trainer.is_deepspeed_enabled=False
    trainer.compute_loss_func=None
    codes={fn.__code__:name for name,fn in methods.items()}
    called=[]
    decision=[]
    def observe(frame,event,arg):
        name=codes.get(frame.f_code)
        if event=="call" and name:
            called.append(name)
        if event=="return" and name=="get_use_logits_to_keep":
            decision.append(arg if type(arg) is bool else None)
    previous=sys.getprofile()
    try:
        sys.setprofile(observe)
        outcome=trainer._prepare_inputs(deepcopy(batch))
    finally:
        sys.setprofile(previous)
    require(len(decision)==1 and type(decision[0]) is bool
            and called==["_prepare_inputs","get_use_logits_to_keep"]+(["prepare_logits_to_keep"] if decision[0] else []),
            "NATIVE_SWIFT_MUTATION_CALL_SEQUENCE")
    return outcome,decision[0],called


def analyze(torch,before,prepared,enabled,methods,signature,metadata):
    require(torch.equal(before["input_ids"],prepared["input_ids"]) and torch.equal(before["pixel_values"],prepared["pixel_values"]),
            "PREPARE_CHANGED_TEXT_OR_MEDIA")
    native=prepared.get("logits_to_keep",0)
    if isinstance(native,torch.Tensor):
        require(native.device.type=="cpu" and native.ndim==1 and native.dtype==torch.bool,"EXPECTED_NATIVE_BS1_CAUSAL_MASK")
        selector=native.tolist()
        kind="native_mask"
    else:
        require(type(native) is int,"NATIVE_SELECTOR_TYPE")
        selector=native
        kind="native_suffix" if enabled else "no_compression"
    if not enabled:
        require("logits_to_keep" not in prepared and signature(prepared)==signature(before),
                "UNCOMPRESSED_FULL_GATE_CHANGED")
    full=before["labels"][0].tolist()
    candidate=prepared["labels"][0].tolist()
    keep=positions(selector,len(full))
    compared=alignment(full,candidate,keep,mode=kind)
    reject=None
    if enabled:
        reject=shifted_one_is_rejected(full,candidate,keep,mode=kind)
    full_gate=True
    try:
        signature(prepared)
    except ValueError as error:
        require(str(error)=="R39_EXACT_BS1_CAPPED_NATIVE_BATCH","UNEXPECTED_FULL_GATE_REJECTION")
        full_gate=False
    require(full_gate is metadata(prepared,torch.int64)["checks"]["token_label_shape_match"],
            "ACTUAL_FULL_GUARD_BEHAVIOR_UNEXPLAINED")
    return dict(selector_kind=kind,selector_count=len(keep),prepared=tensor_metadata(prepared["labels"]),
                input=tensor_metadata(prepared["input_ids"]),loss_scale_present="loss_scale" in prepared,
                full_strict_gate_passed=full_gate,off_by_one_rejected=reject,reference=compared,
                effective_decision=enabled,observed_call_sequence=methods)


def collect(args,report):
    report["phase"]="exact_deployed_source"
    source_bind(args.native_manifest_sha256)
    import torch
    require(not torch.cuda.is_initialized(),"UNEXPECTED_CUDA_INITIALIZATION")
    def deny(*_,**__):
        raise RuntimeError("GPU_CONTEXT_FORBIDDEN")
    torch.cuda._lazy_init=deny
    torch.set_num_threads(2)
    report["phase"]="first_real_processor_and_collator"
    before,template,processor,signature,metadata,receipt=first_collated(torch)
    report["original_rank0_slot"]=receipt
    report["actual_collated_slots"]=1
    report["phase"]="exact_effective_trainer_methods"
    c,methods,model,pins=provider(torch,template,processor)
    report["effective_callable_pins"]=pins
    report["phase"]="three_native_selector_options"
    for mode in MODES:
        row=dict(requested_option=mode,passed=False)
        report["cases"].append(row)
        try:
            result,enabled,called=native_prepare(torch,c,methods,model,template,before,mode)
            row.update(effective_decision=enabled,observed_call_sequence=called)
            row.update(analyze(torch,before,result,enabled,called,signature,metadata),passed=True)
        except (AttributeError,KeyError,TypeError,ValueError,RuntimeError) as error:
            row["failure"]=failure(error)
    require(not torch.cuda.is_initialized() and len(report["cases"])==3
            and all(case["passed"] for case in report["cases"])
            and report["cases"][0]["effective_decision"] is False
            and report["cases"][1]["effective_decision"] is True,"THREE_CASE_BOUNDARY")
    report["passed"]=True


def failure(error):
    label=str(error) if type(error) is ValueError else ""
    field=getattr(error,"name",None) if type(error) is AttributeError else None
    return dict(type=type(error).__name__, code=label if re.fullmatch("[A-Z0-9_]{1,90}",label) else None,
                missing_attribute=field if isinstance(field,str) and re.fullmatch("[A-Za-z0-9_]{1,64}",field) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch",type=float,required=True)
    p.add_argument("--request",required=True)
    p.add_argument("--native-manifest-sha256",default=NATIVE_MANIFEST)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    require(re.fullmatch("[0-9a-f]{32}",args.request) and args.request not in {
            "886fb0896c094de9acc384dc0d13ebb2","b298f9ce6be74ce6b9035fc2f61d7f2e"},"NEW_CPU_ONLY_REQUEST")
    require(args.out.is_absolute() and args.request in args.out.parent.name and not args.out.exists(),"NEW_PRIVATE_RESULT_REQUIRED")
    report=dict(request=args.request, actual_step=gate(args.epoch), original_failure_SHA256=ORIGINAL,original_AST_SHA256=AST_EVIDENCE,
                planned_slots=1,planned_options=3,actual_collated_slots=0, actual_GPU=0,actual_model_forwards=0,
                actual_optimizer_updates=0,passed=False,cases=[],provider_context="CPU_PROCESSOR_AND_REAL_FORWARD_SIGNATURE_ONLY",
                original_eight_rank_effective_option=None,loss_or_logits_equivalence_qualified=False)
    def stop(*_):
        raise TimeoutError("CPU_PROBE_DEADLINE")
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,args.epoch+STEP_BUDGET-time.time())
    sys.addaudithook(bounded_audit)
    try:
        with open(os.devnull,"w") as silent, redirect_stdout(silent),redirect_stderr(silent):
            collect(args,report)
    except Exception as error:
        report["failure"]=failure(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    blob=(json.dumps(report,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.out.open("xb") as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(request=args.request,result_SHA256=sha256(blob).hexdigest(),passed=report["passed"],
                          structural_cases=len(report["cases"]),actual_GPU=0),sort_keys=True))
    return int(not report["passed"])


if __name__=="__main__":
    raise SystemExit(main())
