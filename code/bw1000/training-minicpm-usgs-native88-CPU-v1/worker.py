"""MiniCPM's actual native train-only USGS88 processor/collator, CPU only."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
from providers import verify_wheels,function_chain,file_identity
from input_contract import check_record,check_encoded

ROOT=Path("/pilot")
DATA=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-domain-usgs-table-reading-v1/data/train/swift.jsonl")
MODEL=ROOT/"metadata-only/MiniCPM-V-4.6"
DATA_SHA256="0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62"
VERSIONS={"ms-swift":"4.4.2","transformers":"5.12.1","torch":"2.10.0+das.opt1.dtk2604.2606161421.g91a580"}


def save(name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+"\n").encode()
    temporary=ROOT/(name+".tmp")
    temporary.write_bytes(raw)
    temporary.replace(ROOT/name)
    return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def own_device_fds():
    result=[]
    with os.scandir("/proc/self/fd") as descriptors:
        for item in descriptors:
            try:
                target=os.readlink(item.path)
            except FileNotFoundError:
                continue
            if target=="/dev/kfd" or target.startswith("/dev/dri/"):
                result.append(dict(fd=int(item.name),target=target))
    return result


def main():
    started=time.monotonic()
    intent=json.loads((ROOT/"intent.json").read_bytes())
    deadline=min(intent["deadline"]-55,time.time()+210)
    if not all(math.isfinite(x) for x in (intent["epoch"],intent["deadline"],deadline)) or not intent["epoch"]<=time.time()<deadline:
        raise ValueError("Original CPU scope epoch is invalid or exhausted")
    def expired(*_):
        raise TimeoutError("Original native CPU epoch exhausted")
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
    affinity=sorted(os.sched_getaffinity(0))
    if len(affinity)<8:
        raise ValueError("Expected eight allocated CPU cores")
    os.sched_setaffinity(0,affinity[:8])
    progress=dict(planned_rows=88,planned_encode_calls=176,planned_collator_calls=88,
                  completed_rows=0,encode_entered=0,encode_completed=0,collator_entered=0,collator_completed=0,
                  planned_suffix_metadata_encodes=2,suffix_metadata_encode_entered=0,suffix_metadata_encode_completed=0,
                  phase="admission",actual_optimizer_updates=0,dev_or_blind_read=False)
    save("progress.json",progress)
    phase="admission"
    try:
        device_fds_before=own_device_fds()
        if device_fds_before:
            raise ValueError("CPU worker inherited device descriptors")
        if any(os.environ.get(x)!="-1" for x in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")):
            raise ValueError("CPU visibility masks differ")
        if os.environ.get("SLURM_STEP_GPUS") not in (None,"","(null)"):
            raise ValueError("GPU was assigned to a CPU scope")
        wheels=verify_wheels()
        if {k:metadata.version(k) for k in VERSIONS}!=VERSIONS:
            raise ValueError("Original native software versions differ")
        assets=json.loads((ROOT/"src/ASSETS.json").read_bytes())
        if {p.name for p in MODEL.iterdir()}!=set(assets["metadata"]):
            raise ValueError("Metadata-only model view differs or contains unexpected files")
        for name,pin in assets["metadata"].items():
            p=MODEL/name
            raw=p.read_bytes()
            if p.is_symlink() or len(raw)!=pin["bytes"] or hashlib.sha256(raw).hexdigest()!=pin["SHA256"]:
                raise ValueError("Native model metadata changed")
        raw=DATA.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=DATA_SHA256 or len(raw)>2*1024**2:
            raise ValueError("Existing train-only USGS data changed")
        rows=[json.loads(x) for x in raw.splitlines()]
        if len(rows)!=88 or len({row["id"] for row in rows})!=88:
            raise ValueError("Train row denominator differs")
        phase="native-provider"
        import torch
        torch.set_num_threads(8)
        def no_device(*_,**__):
            raise RuntimeError("Explicit GPU lazy initialization forbidden in native CPU scope")
        torch.cuda._lazy_init=no_device
        from swift.model import get_model_processor
        from swift.template import get_template
        model,processor=get_model_processor(str(MODEL),model_type="minicpmv4_6",load_model=False,
             download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=torch.bfloat16)
        if model is not None:
            raise ValueError("Processor scope unexpectedly loaded a model")
        options=dict(template_type="minicpmv4_6",max_pixels=262144,padding_side="right",padding_free=False,
                     truncation_strategy="raise")
        full=get_template(processor,max_length=None,**options)
        limited=get_template(processor,max_length=4096,**options)
        full.set_mode("train");limited.set_mode("train")
        # The original119 established encode, not this new collator workload.
        # Verify the actual bound method accepts the public one-batch API before
        # processing any row, then exercise that very method on all88 samples.
        inspect.signature(limited.data_collator).bind([{}])
        providers=dict(get_model_processor=function_chain(get_model_processor),get_template=function_chain(get_template),
            encode=function_chain(limited.encode),data_collator=function_chain(limited.data_collator),set_mode=function_chain(limited.set_mode),
            encode_context=function_chain(limited._encode_context_list),tokenize=function_chain(limited._tokenize),
            template_metadata=file_identity(__import__(type(limited.template_meta).__module__,fromlist=["_"]).__file__),
            processor=file_identity(__import__(type(processor).__module__,fromlist=["_"]).__file__))
        providers["actual_method_signatures"]={"encode":str(inspect.signature(limited.encode)),
             "data_collator":str(inspect.signature(limited.data_collator)),"set_mode":str(inspect.signature(limited.set_mode))}
        save("providers-private.json",providers)
        tokenizer=processor.tokenizer
        eos=tokenizer.eos_token_id
        if type(eos) is not int or str(tokenizer.eos_token)!="<|im_end|>":
            raise ValueError("Native tokenizer has no single EOS identity")
        if full.template_meta.suffix!=["<|im_end|>\n"] or limited.template_meta.suffix!=full.template_meta.suffix:
            raise ValueError("Actual MiniCPM ChatML suffix differs from the frozen registration")
        phase="native-suffix-encoder"
        progress["phase"]=phase
        progress["suffix_metadata_encode_entered"]+=1;save("progress.json",progress)
        suffix_full=full._encode_context_list(full.template_meta.suffix,[1.]*len(full.template_meta.suffix))[0]
        progress["suffix_metadata_encode_completed"]+=1
        progress["suffix_metadata_encode_entered"]+=1;save("progress.json",progress)
        suffix_ids=limited._encode_context_list(limited.template_meta.suffix,[1.]*len(limited.template_meta.suffix))[0]
        progress["suffix_metadata_encode_completed"]+=1;save("progress.json",progress)
        if suffix_ids!=suffix_full or not suffix_ids or eos not in suffix_ids:
            raise ValueError("Actual native suffix encoder differs or contains no EOS")
        suffix_identity=save("native-suffix-private.json",dict(template_suffix=["<|im_end|>\n"],
             native_encoder="_encode_context_list",suffix_token_count=len(suffix_ids),
             suffix_token_ids_SHA256=hashlib.sha256(json.dumps(suffix_ids,separators=(",",":")).encode()).hexdigest(),
             EOS_offsets=[i for i,x in enumerate(suffix_ids) if x==eos],
             native_suffix_metadata_encode_calls=2,full_and4096_suffix_equal=True))
        def identity(value):
            if isinstance(value,torch.Tensor):
                if value.device.type!="cpu" or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
                    raise ValueError("Non-CPU or nonfinite native input")
                raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                return dict(kind="tensor",dtype=str(value.dtype),shape=list(value.shape),SHA256=hashlib.sha256(raw).hexdigest())
            if isinstance(value,dict):
                return {str(k):identity(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):
                if all(type(x) in (int,float,bool) for x in value):
                    raw=json.dumps(value,separators=(",",":"),allow_nan=False).encode()
                    return dict(kind="numeric-sequence",length=len(value),SHA256=hashlib.sha256(raw).hexdigest())
                return [identity(x) for x in value]
            if type(value) is str:
                raw=value.encode()
                return dict(kind="str",bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
            if value is None or type(value) in (int,float,bool):
                return value
            raise TypeError("Unknown native input field type")
        records=[]
        for ordinal,row in enumerate(rows):
            phase="encode-row"
            progress["phase"]=phase
            check_record(row)
            progress["encode_entered"]+=1;save("progress.json",progress)
            a=full.encode(row)
            progress["encode_completed"]+=1
            progress["encode_entered"]+=1;save("progress.json",progress)
            b=limited.encode(row)
            progress["encode_completed"]+=1
            first,second=identity(a),identity(b)
            if first!=second:
                raise ValueError("Full native sample was changed by the 4096 contract")
            ids,labels=b["input_ids"],b["labels"]
            if hasattr(ids,"tolist"):
                ids=ids.tolist()
            if hasattr(labels,"tolist"):
                labels=labels.tolist()
            counts=check_encoded(ids,labels,eos,suffix_ids)
            phase="collate-row"
            progress["phase"]=phase;progress["collator_entered"]+=1;save("progress.json",progress)
            batch=limited.data_collator([b])
            batched=identity(batch)
            progress["collator_completed"]+=1
            if (tuple(batch["input_ids"].shape)!=(1,len(ids)) or tuple(batch["labels"].shape)!=(1,len(ids))
                    or batch["input_ids"][0].tolist()!=list(ids) or batch["labels"][0].tolist()!=list(labels)):
                raise ValueError("Native collator altered the complete sample or mask")
            records.append(dict(ordinal=ordinal+1,source_id_SHA256=hashlib.sha256(row["id"].encode()).hexdigest(),
                **counts,
                EOS_supervised=True,full_untruncated=True,encoded=second,collated=batched))
            progress["completed_rows"]+=1;save("progress.json",progress)
        identities=save("native-identities-private.json",records)
        device_fds_after=own_device_fds()
        if device_fds_after:
            raise ValueError("Native CPU scope opened a device descriptor")
        result=dict(passed=True,state="MINICPM_USGS88_NATIVE_INPUT_CPU_COMPLETE",completed_rows=88,planned_rows=88,
          encode_calls=176,collator_calls=88,EOS_supervised=88,full_untruncated=88,unique_media=22,
          native_suffix=suffix_identity,native_suffix_metadata_encode_calls=2,
          native_identities=identities,source_dataset_SHA256=DATA_SHA256,versions=VERSIONS,wheels=wheels,
          token_range=[min(x["length"] for x in records),max(x["length"] for x in records)],
          actual_shifted_supervised_tokens=sum(x["shifted_supervised_tokens"] for x in records),
          model_loaded=False,metadata_only_model_view=True,caller_weights_read=False,
          GPU_allocation=0,explicit_GPU_initialization=False,implicit_native_loader_GPU_access="UNKNOWN",
          own_device_FDs_before=device_fds_before,own_device_FDs_after=device_fds_after,
          actual_optimizer_updates=0,dev_or_blind_read=False,affinity=sorted(os.sched_getaffinity(0)),
          peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          wall_seconds=time.monotonic()-started,training_GO=False,Model_GO=False)
        save("worker-result.json",result)
        return 0
    except BaseException as exc:
        progress["phase"]=phase;save("progress.json",progress)
        save("failure-private.json",dict(phase=phase,error_type=type(exc).__name__,message=str(exc)[:2048],
             planned_rows=88,completed_rows=progress["completed_rows"],actual_optimizer_updates=0))
        save("worker-result.json",dict(passed=False,phase=phase,error_type=type(exc).__name__,progress=progress,
             model_or_training_qualification=False,wall_seconds=time.monotonic()-started))
        return 1


if __name__=="__main__":
    raise SystemExit(main())
