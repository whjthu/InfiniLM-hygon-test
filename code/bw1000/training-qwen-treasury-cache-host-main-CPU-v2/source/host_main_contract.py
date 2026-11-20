"""Real imported Swift CPU/API handoff, with explicit return before model materialization."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import os
import sys
import time

from torch_provider import write_and_validate

SWIFT_ROOT=Path("/swift-runtime/site/swift")
MODEL_CONFIG_SHA="5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
CACHE_KEYS=("TMPDIR","XDG_CACHE_HOME","TORCH_EXTENSIONS_DIR","TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","AMD_COMGR_CACHE_DIR","MIOPEN_USER_DB_PATH","HOME","HF_HOME","HF_MODULES_CACHE")


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path,limit=2*1024**2):
    require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=limit,"unsafe Python/config provider")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider(obj):
    module=inspect.getmodule(obj)
    require(module is not None and module.__file__,"provider lacks actual __file__")
    path=Path(module.__file__).resolve(strict=True)
    require(path.is_relative_to(SWIFT_ROOT),"Swift provider escaped frozen runtime")
    return dict(module=module.__name__,file=str(path),SHA256=sha(path),signature=str(inspect.signature(obj)))


def check_host_main(args,requests,input_root,torch,AdapterRequest,InferRequest,RequestConfig,TransformersEngine,get_model_processor,get_template,engine_pins):
    require(args.host_main_preflight is True and args.arm in ("base","old_domain","domain"),"HOST main optin/arm missing")
    require(time.time()<args.deadline and len(requests)==108,"HOST main input/deadline differs")
    output=args.output
    require(output.is_dir() and not output.is_symlink(),"HOST receipt output must be owned existing directory")
    run=output.parent.resolve(strict=True)
    require(Path.cwd().resolve(strict=True)==run,"declared owned RUN cwd differs")
    for key in CACHE_KEYS:
        path=Path(os.environ.get(key,""))
        require(path.is_absolute() and path.resolve(strict=True).is_relative_to(run/"cache") and path.is_dir() and not path.is_symlink(),"owned cache environment differs")
    cache_receipt=json.loads((run/"owned-cache-readiness.json").read_bytes())
    require(cache_receipt["all_cache_canaries_verified"] is True and cache_receipt["total_canary_bytes"]<=1024**2,"actual owned cache write qualification missing")
    require(sha(args.model/"config.json")==MODEL_CONFIG_SHA,"model config identity differs")
    exports={"AdapterRequest":AdapterRequest,"InferRequest":InferRequest,"RequestConfig":RequestConfig,"TransformersEngine":TransformersEngine,"get_model_processor":get_model_processor,"get_template":get_template}
    providers={key:provider(value) for key,value in exports.items()}
    torch_provider=write_and_validate(torch, output/"torch-provider.json")
    # Exact existing ENGINE_PINS were already read by native_generation.generate before real imports.
    for relative,digest in engine_pins.items():
        require(sha(SWIFT_ROOT/relative)==digest,"real engine file differs")
    real_requests=[InferRequest(messages=deepcopy(row["messages"]),images=[str(input_root/row["images"][0])]) for row in requests]
    require(len(real_requests)==108,"InferRequest full denominator differs")
    config=RequestConfig(max_tokens=64,temperature=0,stream=False,num_beams=1,n=1)
    require(config.max_tokens==64 and config.temperature==0 and config.stream is False and config.num_beams==1 and config.n==1,"actual RequestConfig differs")
    adapter_paths={
        "old_domain":"/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-sft-v3/continuous/train/checkpoint-5",
        "domain":"/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5",
    }
    adapter_requests={name:AdapterRequest(name="treasury_step5",path=path) for name,path in adapter_paths.items()}
    adapter=None if args.arm=="base" else adapter_requests[args.arm]
    if args.arm!="base":
        require(str(args.adapter)==adapter_paths[args.arm],"selected frozen adapter differs")
    if adapter is not None:
        require(str(args.adapter).startswith("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/") and args.adapter.is_dir(),"actual adapter path missing")
    # Exercise the real CPU processor/template path; explicitly forbid materialized models.
    cpu_model,processor=get_model_processor(str(args.model),model_type="qwen3_vl",load_model=False,download_model=False,device_map="cpu")
    require(cpu_model is None,"CPU processor returned a materialized model")
    template=get_template(processor,template_type="qwen3_vl",max_length=4096,max_pixels=262144,padding_side="right",padding_free=False,truncation_strategy="raise")
    template.set_mode("transformers")
    # Signature checks do not instantiate an Engine or pretend a real model exists.
    bound_model_call=inspect.signature(get_model_processor).bind(str(args.model),model_type="qwen3_vl",load_model=True,download_model=False,device_map={"":0},torch_dtype=torch.bfloat16,attn_impl="sdpa")
    inspect.signature(TransformersEngine).bind("HOST_MODEL_NOT_MATERIALIZED",template=template,max_batch_size=1)
    receipt=dict(state="TRUE_ARGPARSE_MAIN_REAL_ENGINE_IMPORTS_AND_CPU_PROCESSOR_RETURNED_BEFORE_LOAD_MODEL_TRUE",arm=args.arm,
        native_main_file=str(Path(sys.argv[0]).resolve()),native_main_SHA256=sha(Path(sys.argv[0])),input_inventory_SHA256=sha(args.inventory),
        actual_InferRequest_objects=len(real_requests),actual_RequestConfig=dict(max_tokens=config.max_tokens,temperature=config.temperature,num_beams=config.num_beams,n=config.n,stream=config.stream),
        actual_AdapterRequest_objects=len(adapter_requests),selected_AdapterRequest_constructed=adapter is not None,adapter_class_module=type(adapter).__module__ if adapter else None,
        real_CPU_processor_type=type(processor).__module__+"."+type(processor).__name__,real_template_type=type(template).__module__+"."+type(template).__name__,
        providers=providers,torch_provider=torch_provider,ENGINE_PINS_verified=True,additional_provider_pins_observed_not_prequalified=True,materialize_call_signature_bound=True,Engine_signature_bound_only=True,
        load_model_True_calls=0,Engine_instances_created=0,generation_calls=0,CUDA_success_mocked=False,explicit_CUDA_probe_calls=0,library_implicit_provider_or_device_queries="UNKNOWN",
        cache_readiness_SHA256=sha(run/"owned-cache-readiness.json"),cwd=str(Path.cwd()),cache_environment={k:os.environ[k] for k in CACHE_KEYS},
        default_mode_is_GPU_math_unchanged=True,default_HOST_mode=False,cache_canary_only_proves_tested_bytes=True,user_quota_and_arbitrary_JIT_capacity_qualified=False,GPU_GO=False,formal_quality=False,formal_performance=False)
    with (output/"host-main-receipt.json").open("x",encoding="utf8") as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
