"""One call-local cache recovery for the actual Swift Hugging Face dataset path."""

import ast
from contextlib import contextmanager
import errno
from functools import wraps
from hashlib import sha256
import inspect
import os
from pathlib import Path
import re
import sys


PARAMETERS={
    "bf16","fp16","bf16_full_eval","fp16_full_eval","use_cpu","half_precision_backend","torch_dtype",
    "device","parallel_mode","ddp_backend","fsdp","deepspeed","gradient_checkpointing","tf32",
    "dataloader_num_workers","evaluation_strategy","eval_strategy","load_best_model_at_end","metric_for_best_model",
    "save_strategy","save_steps","eval_steps","warmup_steps","warmup_ratio","max_steps","accelerator_config",
}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def path(value):
    if not isinstance(value,(str,bytes,os.PathLike)):
        return None
    raw=os.fsdecode(value)
    if "://" in raw:
        return None
    return Path(os.path.abspath(raw))


def trace_frames(exc):
    current=exc.__traceback__
    while current is not None:
        frame=current.tb_frame
        yield frame.f_globals.get("__name__",""),frame.f_code.co_name,frame.f_locals,current.tb_lineno,frame.f_code.co_filename
        current=current.tb_next


def sft_value_error(exc):
    if not isinstance(exc,ValueError):
        return {"category":"NOT_VALUE_ERROR","parameter_names":[]}
    names=set()
    proven_frames=[]
    for module,fn,_,line,filename in trace_frames(exc):
        if module not in {"transformers.training_args","swift.arguments.sft_args","swift.trainers.arguments",
                          "swift.trainers.trainer_factory"}:
            continue
        proven_frames.append((module,fn))
        if module!="transformers.training_args":
            continue
        try:
            with Path(filename).open("rb") as stream:
                raw=stream.read(1024*1024+1)
            if len(raw)>1024*1024:
                continue
            tree=ast.parse(raw)
            cases=[item for item in ast.walk(tree) if isinstance(item,ast.If) and
                   item.lineno<=line<=(item.end_lineno or item.lineno)]
            if cases:
                clause=max(cases,key=lambda item:item.lineno)
                names.update(node.attr for node in ast.walk(clause.test) if isinstance(node,ast.Attribute)
                             and node.attr in PARAMETERS)
        except (OSError,SyntaxError,ValueError):
            continue
    # Human text is used only as a local enum classifier, never persisted or output.
    message=str(exc)[:8192].lower()
    if proven_frames:
        names.update(item for item in PARAMETERS-{"device"} if
                     re.search(r"(?<![a-z0-9_])"+item+r"(?![a-z0-9_])",message))
    precision=bool(names&{"bf16","fp16","bf16_full_eval","fp16_full_eval","tf32","half_precision_backend","torch_dtype"})
    category=("CPU_OR_DEVICE_PRECISION_CONFLICT" if precision and
              ("cpu" in message or "gpu" in message or "device" in message) else
              "PRECISION_VALIDATION" if precision else "TRAINING_ARGUMENT_VALIDATION" if proven_frames else "UNCLASSIFIED_VALUE_ERROR")
    return {"category":category,"parameter_names":sorted(names),"raw_text_included":False}


class CacheGate:
    def __init__(self, scratch, default_root, category, report):
        self.scratch=scratch.resolve(strict=True)
        self.default_root=path(default_root)
        self.category=category
        self.report=report
        self.repaired=False
        self.stage=None
        self.front_depth=0

    def project(self, value):
        p=path(value)
        if p is None:
            return {"category":"UNAVAILABLE_OR_NONLOCAL","path_SHA256":None,"within_request_scratch":False}
        return {"category":self.category(p),"path_SHA256":sha256(os.fsencode(str(p))).hexdigest(),
                "within_request_scratch":p.is_relative_to(self.scratch)}

    def cache_argument(self, signature, bound):
        present="cache_dir" in bound.arguments
        arg=bound.arguments.get("cache_dir")
        default=signature.parameters["cache_dir"].default
        if present and arg is not None:
            return "EXPLICIT",arg
        if present:
            return "EXPLICIT_NONE_DATASETS_CONFIG",self.default_root
        if default is inspect.Parameter.empty or default is None:
            return "OMITTED_DATASETS_CONFIG_DEFAULT",self.default_root
        return "OMITTED_SIGNATURE_DEFAULT",default

    def proven_cache_failure(self, exc, effective):
        """Require actual builder-owned cache locals and their first attempted mkdir."""
        if not isinstance(exc,OSError) or exc.errno!=errno.EROFS or self.front_depth<1:
            return None
        failed=path(exc.filename)
        intended=None
        roots=[]
        descended=False
        for module,fn,locals_,_,_ in trace_frames(exc):
            if module=="datasets.builder" and fn=="__init__":
                builder=locals_.get("self")
                root=path(getattr(builder,"_cache_dir_root",None))
                cache=path(getattr(builder,"_cache_dir",None))
                if root is not None:
                    roots.append((root,cache))
                descended=True
            if descended and module=="os" and fn=="makedirs" and intended is None:
                intended=path(locals_.get("name"))
        if failed is None or intended is None or len(roots)!=1:
            return None
        [(root,cache)]=roots
        effective_path=path(effective)
        blocked={"ORIGINAL_TRAIN800_JSONL","ORIGINAL_TRAIN_MEDIA","ORIGINAL_FRAMEWORK",
                 "PROCESSOR_ONLY_METADATA","REQUEST_SOURCE","REQUEST_RESULT"}
        if (root.is_relative_to(self.scratch) or self.category(root) in blocked
            or not intended.is_relative_to(root) or not intended.is_relative_to(failed)
            or (cache is not None and not cache.is_relative_to(root))
            or (effective_path is not None and root!=effective_path and not root.is_relative_to(effective_path))):
            return None
        return {"actual_builder_root":self.project(root),"effective_builder_cache":self.project(cache),
                "first_mkdir_target":self.project(intended),"failed_path":self.project(failed)}

    def wrap(self, real, role):
        signature=inspect.signature(real)
        need("cache_dir" in signature.parameters,"R30_NATIVE_HF_CACHE_ARGUMENT_SIGNATURE_REQUIRED")
        @wraps(real)
        def wrapped(*args,**kwargs):
            bound=signature.bind(*args,**kwargs)
            origin,target=self.cache_argument(signature,bound)
            log=self.report["calls"]
            need(len(log)<12,"R30_BOUNDED_NATIVE_CALL_LOG")
            event={"phase":self.stage,"provider_role":role,"cache_dir_origin":origin,
                   "effective_target":self.project(target),"rerouted":False}
            log.append(event)
            if role=="hf_front":
                self.front_depth+=1
                try:
                    return real(*args,**kwargs)
                finally:
                    self.front_depth-=1
            try:
                return real(*args,**kwargs)
            except OSError as error:
                event["errno"]=error.errno
                event["failed_path"]=self.project(error.filename)
                proof=self.proven_cache_failure(error,target)
                if proof is None or self.repaired:
                    event["outcome"]="EROFS_UNPROVEN_OR_RECOVERY_ALREADY_SPENT" if error.errno==errno.EROFS else "OTHER_OS_ERROR"
                    raise
                event["actual_builder_mkdir_proof"]=proof
                cache=self.scratch/"one-builder-recovery"
                cache.mkdir(mode=0o700)
                self.repaired=True
                bound.arguments["cache_dir"]=str(cache)
                event["rerouted"]=True
                event["new_target"]=self.project(cache)
                self.report["recovery_calls"]=1
                # Only this call receives one replacement argument. Exceptions from
                # the retry are returned unchanged, without further fallback.
                return real(*bound.args,**bound.kwargs)
        return wrapped


@contextmanager
def native_binding(scratch,category,report):
    import datasets
    import datasets.load as dl
    configured=datasets.config.HF_DATASETS_CACHE
    front,builder=dl.load_dataset,dl.load_dataset_builder
    need(datasets.load_dataset is front and callable(builder),"R30_EXACT_DATASETS_MODULE_ALIAS")
    callback=CacheGate(scratch,configured,category,report)
    hfront,hbuilder=callback.wrap(front,"hf_front"),callback.wrap(builder,"hf_builder")
    previous=[]
    for name,module in tuple(sys.modules.items()):
        if module is None or not (name=="datasets" or name=="datasets.load" or name.startswith("swift.")):
            continue
        for attr,value in tuple(vars(module).items()):
            if value is front or value is builder:
                if name.startswith("swift.") and len(report["swift_namespace_references"])<12:
                    report["swift_namespace_references"].append({"module":name,"attribute":attr,
                        "role":"hf_front" if value is front else "hf_builder"})
                previous.append((module,attr,value))
                setattr(module,attr,hfront if value is front else hbuilder)
    try:
        yield callback
    finally:
        for module,attr,value in reversed(previous):
            setattr(module,attr,value)
        for name,module in tuple(sys.modules.items()):
            if module is not None and name.startswith("swift."):
                for attr,value in tuple(vars(module).items()):
                    if value is hfront or value is hbuilder:
                        setattr(module,attr,front if value is hfront else builder)
