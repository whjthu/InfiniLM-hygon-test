"""Read the exact .4560 adapter into a fresh materialized CPU transformer."""
import argparse
import ast
import hashlib
import json
from math import prod
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
import sys
import time
from typing import Any

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT,NODE = "1103606","f13r3n14"
ORIGINAL_LABEL = "training-hunyuanvideo-t09-abo-domain-n14-fd-v3"
ORIGINAL_SCOPE = "1b22ddd9e5854853a43e646807e92b3f"
ORIGINAL = ROOT/"results"/ORIGINAL_LABEL
ORIGINAL_MANIFEST = "0a073ca414d30d035100818d9dc21c02e1ba55a0a50cde3d2e652a33aca34643"
ADAPTER = ORIGINAL/"outputs/hunyuanvideo-1.5/train"/ORIGINAL_LABEL/"adapter-final.safetensors"
ADAPTER_BYTES = 3570312
ADAPTER_SHA = "2436d2a23b0a9415f9f8893e261008846afe37b8185dade67ce7b1fc65e02b76"
NAMED_SHA = "bd9dfa706ee5dca1d4732caebada76d0cdb0a14094d5487acee52634d3b60221"
WORKER_ELEMS = 1769472
WORKER_TENSORS = 216
MODEL = ROOT/"results/training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5/transformer/480p_t2v"
WEIGHT = MODEL/"diffusion_pytorch_model.safetensors"
COMPUTE_STAMP = (49,11025132338,33306632192,1794715911000000000,1794715912000000000)
BASE_HEADER_BYTES = 199544
BASE_HEADER_SHA = "a6feb5932d0522fb8824f66eebef5569b0f3072cfb2cc2a95ad921280d62e0c2"
BASE_COUNT = 1793
BASE_ELEMENTS = 8326608160
BASE_BF16_BYTES = 16653216320
FULL_SHA_RECEIPT = ROOT/"results/T09-native25-original-transformer-rebind-fullsha-CPU-v1/normalized-independent-terminal.json"
FULL_SHA_RECEIPT_SHA = "8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4"
WEIGHT_SHA = "71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86"
CONFIG_SHA = "689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae"
NATIVE = ROOT/"results/training-hunyuanvideo-original-source-v1/source"
NATIVE_ALIAS = Path("/t09-native/hunyuanvideo-1.5")
SOURCE = ORIGINAL/"src"
FRAME = ROOT/"results/training-qwen-a68952c-swift-initial-CPU-v1"
FRAMEWORK = FRAME/"framework"
FRAMEWORK_MANIFEST = FRAME/"source/FRAMEWORK_FILES.json"
BACKEND = FRAMEWORK/"generation/src/gencuda/backends/hunyuanvideo_training_walltime.py"
BACKEND_SHA = "2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae"
NATIVE_MODEL_FILE = "hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py"
NATIVE_SHA = "d1a702891a92e04c80cdc072bc06cffe6e631fb962dae702d3c9244d347302d0"
NATIVE_MANIFEST_SHA = "26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7"
RUNTIME = Path("/training-runtime")
RUNTIME_SHA = "56f65e63b28f8ec30c7dc1efb3f6dc1df291d4a5d981f91d681f2fad41a03b6a"
LAUNCH_SHA = "3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f"
TORCH_VERSION_SHA = "d324ffd50535bca68131043b76605e36d588a85ec41b2aed9d798ece471aa534"
TERM = "1103606.4560"
MEM_LIMIT_MIB = 64000
CPU = 8


def need(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def read(path, cap, expected=None):
    before=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,"CANONICAL_BOUNDED_READ")
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb") as stream:
        first=os.fstat(stream.fileno())
        raw=stream.read(cap+1)
        last=os.fstat(stream.fileno())
    need(stamp(before)==stamp(first)==stamp(last)==stamp(path.lstat()) and len(raw)==first.st_size,
         "PINNED_FILE_CHANGED")
    need(expected is None or digest(raw)==expected,"PINNED_FILE_SHA")
    return raw


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes":len(raw),"SHA256":digest(raw)}


def source_helpers(raw):
    need(digest(raw)==BACKEND_SHA,"EXACT_4560_BACKEND_SHA")
    initial=ast.parse(raw,filename=str(BACKEND))
    helpers=[node for node in initial.body if isinstance(node,ast.FunctionDef)
             and node.name in ("_adapter_sha","_apply_gradient_checkpointing")]
    need({n.name for n in helpers}=={"_adapter_sha","_apply_gradient_checkpointing"} and len(helpers)==2,
         "EXACT_TRAINING_HELPERS")
    scope={"Any":Any,"hashlib":hashlib}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(BACKEND),"exec"),scope)
    return scope["_adapter_sha"],scope["_apply_gradient_checkpointing"]


def checked_recipe(raw):
    need(digest(raw)==LAUNCH_SHA,"EXACT_ORIGINAL_LAUNCH")
    cfg=json.loads(raw)["config"]
    need(cfg["transformer_version"]=="480p_t2v" and cfg["seed"]==20260911
         and cfg["lora_rank"]==4 and cfg["lora_alpha"]==8 and cfg["lora_dropout"]==0.0
         and cfg["lora_target_modules"]==["img_attn_q","img_attn_v"] and cfg["gradient_checkpointing"] is True,
         "EXACT_TRAINED_LORA_RECIPE")
    return cfg


def installed_source():
    meta=json.loads(read(SOURCE/"SOURCE_FILES.json",16384,ORIGINAL_MANIFEST))
    need(meta["request"]==ORIGINAL_SCOPE and meta["label"]==ORIGINAL_LABEL and len(meta["files"])==35,"EXACT_V3_SOURCE_SCOPE")
    for name,item in meta["files"].items():
        need(Path(name).name==name and len(read(SOURCE/name,item["bytes"],item["SHA256"]))==item["bytes"],
             "ORIGINAL_V3_MEMBER")
    sys.path.insert(0,str(SOURCE))
    from framework_binding import verify_framework
    from flash_compat import validate_execution_compatibility
    framework=verify_framework(FRAMEWORK,FRAMEWORK_MANIFEST)
    need(framework["framework_files"]==393,"EXACT_V3_FRAMEWORK_393")
    frozen=json.loads(read(SOURCE/"NATIVE_BINDING.json",32768,NATIVE_MANIFEST_SHA))["files"]
    need(len(frozen)==38,"EXACT_V3_NATIVE38")
    for name,item in frozen.items():
        p=PurePosixPath(name)
        need(not p.is_absolute() and ".." not in p.parts and p.as_posix()==name,"NATIVE_MEMBER_NAME")
        need(len(read(NATIVE.joinpath(*p.parts),item["bytes"],item["SHA256"]))==item["bytes"],"ORIGINAL_NATIVE_MEMBER")
    need(len(read(RUNTIME/"receipt.json",2124,RUNTIME_SHA))==2124,"ORIGINAL_DTK_RUNTIME_MANIFEST")
    validate_execution_compatibility()
    return framework


def original_model_binding():
    need(WEIGHT.resolve(strict=True)==WEIGHT and stat.S_ISREG(WEIGHT.lstat().st_mode)
         and stamp(WEIGHT.lstat())==COMPUTE_STAMP,"EXACT_4560_CURRENT_BASE_WEIGHT_IDENTITY")
    raw=read(FULL_SHA_RECEIPT,16384,FULL_SHA_RECEIPT_SHA)
    previous=json.loads(raw)
    need(previous["passed"] is True and previous["weight_path"]==str(WEIGHT)
         and previous["weight_bytes"]==COMPUTE_STAMP[2] and previous["weight_SHA256"]==WEIGHT_SHA
         and previous["payload_bytes_read"]==COMPUTE_STAMP[2] and previous["stable_stat"] is True
         and previous["file_identity"]==list(COMPUTE_STAMP),"EXACT_HISTORICAL_FULL_BASE_SHA_AND_CURRENT_IDENTITY")
    read(MODEL/"config.json",968,CONFIG_SHA)


def is_cpu_bf16(bf16, parameter):
    return not parameter.is_meta and parameter.device.type=="cpu" and parameter.dtype==bf16


def adapter_pairs(model, state, bf16, *, count=WORKER_TENSORS, elements=WORKER_ELEMS):
    pairs={name:parameter for name,parameter in model.named_parameters() if parameter.requires_grad}
    need(len(state)==len(pairs)==count and set(pairs)==set(state),"EXACT_ADAPTER_KEY_COVERAGE")
    need(sum(p.numel() for p in pairs.values())==elements,"EXACT_ADAPTER_ELEMENT_COUNT")
    for name, parameter in pairs.items():
        source=state[name]
        need(re.search(r"(?:^|\.)lora_[AB]\.default\.weight$",name) is not None,"EXACT_ORIGINAL_NAMED_LORA")
        need(is_cpu_bf16(bf16,source) and is_cpu_bf16(bf16,parameter),"EXACT_CPU_BF16_DESTINATION_AND_SOURCE")
        need(tuple(source.shape)==tuple(parameter.shape),"EXACT_ADAPTER_SHAPE")
    return pairs


def load_values(torch, model, state, original_sha, *, count=WORKER_TENSORS, elements=WORKER_ELEMS, expected=NAMED_SHA):
    targets=adapter_pairs(model,state,torch.bfloat16,count=count,elements=elements)
    completed=0
    with torch.no_grad():
        for name in sorted(targets):
            source,target=state[name],targets[name]
            need(bool(torch.isfinite(source).all().item()),"NONFINITE_EXPORTED_ADAPTER")
            target.copy_(source,non_blocking=False)
            need(torch.equal(target.detach(),source),"EXACT_PER_NAMED_PARAMETER_COPY")
            completed+=1
    actual=original_sha(model)
    need(actual==expected,"EXACT_ORIGINAL_ADAPTER_SHA")
    return dict(copied_and_compared=completed,params=elements,dtype="BF16",native_adapter_SHA256=actual)


def strict_object(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,"DUPLICATE_SAFETENSORS_JSON_KEY")
        result[key]=value
    return result


def header_inventory(raw, payload, count=BASE_COUNT, elements=BASE_ELEMENTS):
    doc=json.loads(raw,object_pairs_hook=strict_object)
    need(type(doc) is dict,"EXACT_SAFETENSORS_HEADER_OBJECT")
    metadata=doc.pop("__metadata__",{})
    need(type(metadata) is dict and all(type(v) is str for v in metadata.values()),"EXACT_SAFETENSORS_METADATA")
    need(len(doc)==count,"EXACT_PUBLISHED_HEADER_COUNT")
    shapes,intervals,total={},[],0
    for name,item in doc.items():
        need(0<len(name)<=512 and type(item) is dict and set(item)=={"dtype","shape","data_offsets"}
             and item["dtype"]=="F32","EXACT_PUBLISHED_HEADER_F32_MEMBER")
        shape,offsets=item["shape"],item["data_offsets"]
        need(type(shape) is list and len(shape)<=16 and all(type(x) is int and x>0 for x in shape)
             and type(offsets) is list and len(offsets)==2 and all(type(x) is int for x in offsets),
             "EXACT_SAFETENSORS_SHAPE_AND_OFFSETS")
        size=1
        for dimension in shape:
            size*=dimension
        start,end=offsets
        need(0<=start<end<=payload and end-start==size*4,"EXACT_F32_PAYLOAD_SPAN")
        shapes[name]=tuple(shape)
        intervals.append((start,end))
        total+=size
    position=0
    for start,end in sorted(intervals):
        need(start==position,"EXACT_GAP_FREE_SAFETENSORS_PAYLOAD")
        position=end
    need(position==payload and total==elements,"EXACT_PUBLISHED_HEADER_ELEMENTS")
    return shapes


def actual_header():
    before=WEIGHT.lstat()
    need(stamp(before)==COMPUTE_STAMP and stat.S_ISREG(before.st_mode),"EXACT_HEADER_SOURCE_INODE")
    with os.fdopen(os.open(WEIGHT,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),"rb") as stream:
        first=os.fstat(stream.fileno())
        prefix=stream.read(8)
        need(len(prefix)==8 and struct.unpack("<Q",prefix)[0]==BASE_HEADER_BYTES,"EXACT_FROZEN_HEADER_EXTENT")
        raw=stream.read(BASE_HEADER_BYTES)
        last=os.fstat(stream.fileno())
    need(stamp(before)==stamp(first)==stamp(last)==stamp(WEIGHT.lstat())
         and len(raw)==BASE_HEADER_BYTES and digest(raw)==BASE_HEADER_SHA,"EXACT_FROZEN_HEADER_SHA_AND_FD")
    return header_inventory(raw,before.st_size-8-BASE_HEADER_BYTES)


def tensor_identity(torch, tensor, shape):
    need(is_cpu_bf16(torch.bfloat16,tensor) and tensor.layout==torch.strided and tensor.is_contiguous(),
         "ORIGINAL_MATERIALIZED_CPU_BF16_CONTIGUOUS")
    need(tuple(tensor.shape)==shape and tensor.element_size()==2 and tensor.numel()==prod(shape)>0,
         "ORIGINAL_TENSOR_SHAPE_AND_EXTENT")
    data,storage=tensor.data_ptr(),tensor.untyped_storage()
    base,extent,offset=storage.data_ptr(),storage.nbytes(),tensor.storage_offset()
    size=tensor.numel()*2
    need(base>0 and offset>=0 and data==base+2*offset and 2*offset+size<=extent,"ORIGINAL_TENSOR_STORAGE_SPAN")
    return (shape,tuple(tensor.stride()),data,base,extent,offset,size)


def distinct_named(pairs):
    result={}
    for key,obj in pairs:
        need(key not in result,"DUPLICATE_REGISTERED_TENSOR_NAME")
        result[key]=obj
    return result


def capture_base(torch,model,expected,diag):
    params=distinct_named(model.named_parameters(remove_duplicate=False))
    buffers=distinct_named(model.named_buffers(remove_duplicate=False))
    nonpersistent=set()
    for prefix,module in model.named_modules(remove_duplicate=False):
        for base in module._non_persistent_buffers_set:
            if module._buffers.get(base) is not None:
                nonpersistent.add((prefix+"." if prefix else "")+base)
    need(nonpersistent<=buffers.keys(),"REGISTERED_NONPERSISTENT_BUFFER_COVERAGE")
    state=model.state_dict(keep_vars=True)
    persistent={key:obj for key,obj in buffers.items() if key not in nonpersistent}
    registered=params|persistent
    diag.update(header_tensors=len(expected),state_tensors=len(state),parameters=len(params),
                persistent_buffers=len(persistent),nonpersistent_buffers=len(nonpersistent),
                missing_header_keys=len(expected.keys()-state.keys()),extra_state_keys=len(state.keys()-expected.keys()),
                missing_registered_keys=len(expected.keys()-registered.keys()),extra_registered_keys=len(registered.keys()-expected.keys()),
                nonpersistent_buffer_schema_SHA256=digest(json.dumps([
                    [key,list(buffers[key].shape),str(buffers[key].dtype)] for key in sorted(nonpersistent)
                    ],separators=(",",":"),ensure_ascii=True).encode()))
    need(not params.keys()&buffers.keys() and set(expected)==set(state)==set(registered),"EXACT_HEADER_STATE_REGISTRATION_BIJECTION")
    for key in nonpersistent:
        value=buffers[key]
        need(value.device.type=="cpu" and not value.is_meta,"EXACT_NONPERSISTENT_CPU_BUFFER")
    captured={}
    spans=[]
    for key,shape in expected.items():
        value=registered[key]
        diag["current_key_SHA256"]=digest(key.encode())
        need(state[key] is value and id(value) not in captured,"EXACT_KEEP_VARS_OBJECT_BIJECTION")
        signature=tensor_identity(torch,value,shape)
        captured[id(value)]=(value,signature,key in params,diag["current_key_SHA256"])
        spans.append((signature[2],signature[2]+signature[-1],diag["current_key_SHA256"]))
    end=0
    for start,limit,key_sha in sorted(spans):
        diag["current_key_SHA256"]=key_sha
        need(start>=end,"ORIGINAL_DISTINCT_LIVE_TENSOR_STORAGE")
        end=limit
    diag.pop("current_key_SHA256",None)
    return captured


def check_injection(torch,model,captured,diag,adapter_count=WORKER_TENSORS):
    params=[p for _,p in model.named_parameters(remove_duplicate=False)]
    buffers=[p for _,p in model.named_buffers(remove_duplicate=False)]
    live_parameters={id(p):p for p in params}
    live_buffers={id(p):p for p in buffers}
    base_ids={i for i,item in captured.items() if item[2]}
    diag.update(original_parameters=len(base_ids),post_injection_parameters=len(params),
                missing_original_parameters=len(base_ids-live_parameters.keys()),
                added_parameters=len(live_parameters.keys()-base_ids))
    need(len(params)==len(live_parameters) and base_ids<=live_parameters.keys(),"BASE_PARAMETER_OBJECT_COVERAGE_AFTER_LORA")
    added=[p for i,p in live_parameters.items() if i not in base_ids]
    need(len(added)==adapter_count and all(p.requires_grad for p in added),"EXACT_ADAPTER_ONLY_ADDED_PARAMETERS")
    for ident,(value,signature,is_parameter,key_sha) in captured.items():
        diag["current_key_SHA256"]=key_sha
        live=live_parameters if is_parameter else live_buffers
        need(live.get(ident) is value and not value.requires_grad,"EXACT_FROZEN_BASE_OBJECT_AFTER_LORA")
        need(tensor_identity(torch,value,signature[0])==signature,"EXACT_BASE_STORAGE_AFTER_LORA")
    diag.pop("current_key_SHA256",None)


def physical_model(torch, captured):
    model_bytes=0
    params=0
    pages=0
    page=os.sysconf("SC_PAGESIZE")
    for tensor,signature,is_parameter,_ in captured.values():
        if is_parameter:
            params+=tensor.numel()
        model_bytes+=signature[-1]
        raw=tensor.detach().reshape(-1).view(torch.uint8)
        for start in range(0,raw.numel(),page*32768):
            selected=raw[start:start+page*32768:page]
            selected.sum(dtype=torch.int64).item()
            pages+=selected.numel()
        raw[-1].item()
    need(len(captured)==BASE_COUNT and model_bytes==BASE_BF16_BYTES and pages>0,"EXACT_FULL_BASE_STORAGE_TOTAL")
    return dict(base_parameter_elements=params,base_tensor_bytes=model_bytes,pages_sampled=pages,
                extra_working_storage="page_slice_32KiB_and_original_3.6MB_adapter",intentional_full_model_copy=False)


def rss():
    values=Path("/proc/self/status").read_text().splitlines()
    lines=[line.split() for line in values if line.startswith("VmRSS:")]
    [row]=lines
    need(len(row)==3 and row[2]=="kB","REQUIRED_CPU_RESIDENCY_METADATA")
    return int(row[1])*1024


def ensure_zero():
    need(all(os.environ.get(k)=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
         and os.environ.get("SLURM_STEP_GPUS") in (None,"","(null)")
         and os.environ.get("SLURM_GPUS_ON_NODE") in (None,"","0"),"ZERO_GRES_GPU_VISIBILITY")
    for name in Path("/proc/self/fd").iterdir():
        try:
            target=os.readlink(name)
        except FileNotFoundError:
            continue
        need(target!="/dev/kfd" and not target.startswith("/dev/dri/"),"NO_OWN_GPU_DESCRIPTORS")


def run(args):
    import resource
    started=time.monotonic()
    phase="initial"
    result=dict(request=args.request,passed=False,planned_adapter_tensors=WORKER_TENSORS,actual_GPU=0,
                original_step=TERM,actual_forward=0,actual_optimizer_updates=0,actual_raw_exports=0)
    try:
        def expired(*unused):
            raise TimeoutError("CPU_RELOAD_DEADLINE")
        signal.signal(signal.SIGALRM,expired)
        signal.alarm(450)
        gate=json.loads(read(args.gate,8192,args.gate_sha))
        step=os.environ.get("SLURM_STEP_ID","")
        need(re.fullmatch("[0-9a-f]{32}",args.request) and gate["request"]==args.request
             and step.isdecimal() and gate["step"]==PARENT+"."+step and gate["node"]==NODE
             and gate["zero_GRES"] is True and gate["CPU"]==CPU and gate["memory_MiB"]==MEM_LIMIT_MIB,
             "EXACT_NUMBERED_ZERO_GRES_GATE")
        result["actual_step"]=gate["step"]
        cgroups=Path("/proc/self/cgroup").read_text().splitlines()
        need(os.environ.get("SLURM_JOB_ID")==PARENT
             and any({"job_"+PARENT,"step_"+step}<=set(line.split(":",2)[-1].split("/")) for line in cgroups),"EXACT_COMPUTE_CGROUP")
        ensure_zero()
        phase="source_and_mount"
        for path in (ROOT,RUNTIME,FRAMEWORK,FRAMEWORK_MANIFEST,NATIVE_ALIAS,SOURCE,MODEL,ADAPTER):
            need(os.statvfs(path).f_flag&os.ST_RDONLY,"ORIGINAL_ASSET_NOT_READ_ONLY")
        result["framework"]=installed_source()
        original_model_binding()
        phase="pinned_safetensors_header"
        expected=actual_header()
        result["base_checkpoint_header"]=dict(SHA256=BASE_HEADER_SHA,bytes=BASE_HEADER_BYTES,
             source_dtype="F32",tensors=BASE_COUNT,elements=BASE_ELEMENTS,expected_resident_BF16_bytes=BASE_BF16_BYTES)
        sha_live,checkpointing=source_helpers(read(BACKEND,1<<20,BACKEND_SHA))
        cfg=checked_recipe(read(SOURCE/"ORIGINAL_LAUNCH.json",32768,LAUNCH_SHA))
        adapter=read(ADAPTER,ADAPTER_BYTES,ADAPTER_SHA)
        need(len(adapter)==ADAPTER_BYTES,"EXACT_4560_ADAPTER_FILE_BYTES")
        result["adapter_file_SHA256"]=ADAPTER_SHA
        phase="providers"
        sys.path.insert(0,str(NATIVE_ALIAS))
        import torch
        version=Path(torch.version.__file__)
        read(version,32768,TORCH_VERSION_SHA)
        need(bool(torch.version.hip) and not torch.cuda.is_initialized(),"ORIGINAL_HIP_PROVIDER_WITHOUT_GPU_INIT")
        torch.cuda._lazy_init=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError("CPU_SCOPE_GPU_INIT_FORBIDDEN"))
        torch.set_num_threads(4)
        from peft import LoraConfig
        from safetensors.torch import load as load_safetensors
        import hyvideo.models.transformers.hunyuanvideo_1_5_transformer as native
        need(Path(native.__file__)==NATIVE_ALIAS/NATIVE_MODEL_FILE,"EXACT_NATIVE_CLASS_SOURCE_PATH")
        read(Path(native.__file__),65536,NATIVE_SHA)
        cls=native.HunyuanVideo_1_5_DiffusionTransformer
        phase="full_cpu_model"
        result["RSS_before_model_bytes"]=rss()
        torch.manual_seed(cfg["seed"])
        model=cls.from_pretrained(MODEL,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                                  local_files_only=True,use_safetensors=True)
        result["model_loads"]=1
        phase="base_state_inventory_before_lora"
        before=result["base_state_inventory"]={}
        captured=capture_base(torch,model,expected,before)
        phase="exact_original_lora_registration"
        model.set_attn_mode("torch")
        model.add_adapter(LoraConfig(r=cfg["lora_rank"],lora_alpha=cfg["lora_alpha"],
                         target_modules=cfg["lora_target_modules"],lora_dropout=cfg["lora_dropout"],
                         bias="none",task_type="FEATURE_EXTRACTION"),adapter_name="default")
        checkpointing(model)
        model.train()
        phase="frozen_base_identity_after_lora"
        after=result["post_lora_inventory"]={}
        check_injection(torch,model,captured,after)
        phase="material_residency"
        result["base"]=physical_model(torch,captured)
        result["RSS_after_base_touch_bytes"]=rss()
        need(result["RSS_after_base_touch_bytes"]>=result["base"]["base_tensor_bytes"]*0.90,"FULL_CPU_BASE_RSS_NOT_PROVED")
        phase="named_native_adapter_reload"
        result["adapter"]=load_values(torch,model,load_safetensors(adapter),sha_live)
        phase="same_original_asset_readback"
        original_model_binding()
        need(read(ADAPTER,ADAPTER_BYTES,ADAPTER_SHA)==adapter,"EXACT_ADAPTER_BYTES_UNCHANGED")
        ensure_zero()
        need(not torch.cuda.is_initialized(),"CPU_RELOAD_NO_DEVICE_INITIALIZATION")
        result.update(passed=True,phase="COMPLETE_CPU_RELOAD",model_loads=1,grad_checkpoint_wrappers_reused=True,
                      full_new_base_file_byte_SHA=False,model_base_file_original_SHA256=WEIGHT_SHA)
    except Exception as error:
        reason=str(error)
        result.update(phase=phase,error_type=type(error).__name__,errno=getattr(error,"errno",None),
                      error_code=reason if re.fullmatch("[A-Z0-9_]{1,110}",reason) else None,
                      detail_SHA256=digest(reason.encode("utf8","replace")))
    finally:
        signal.alarm(0)
        result["seconds"]=time.monotonic()-started
        result["peak_RSS_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        pin=save(args.output,result)
    print(json.dumps(dict(request=args.request,passed=result["passed"],phase=result["phase"],receipt=pin,
                          actual_GPU=0),sort_keys=True),flush=True)
    return 0 if result["passed"] else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("request","gate-sha"):
        parser.add_argument("--"+key,required=True)
    for key in ("gate","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
