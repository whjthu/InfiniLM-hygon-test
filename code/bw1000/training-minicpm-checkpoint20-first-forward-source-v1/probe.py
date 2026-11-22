"""Capture one native resumed loss, then stop before Trainer backward."""

from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import time


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return sha256(raw).hexdigest()


def context(torch):
    cuda=torch.backends.cuda
    return dict(grad_enabled=torch.is_grad_enabled(),inference=torch.is_inference_mode_enabled(),
                autocast_enabled=torch.is_autocast_enabled("cuda"),autocast_dtype=str(torch.get_autocast_dtype("cuda")),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=cuda.matmul.allow_tf32,
                matmul_allow_bf16_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
                enabled_SDPA={name:getattr(cuda,name+"_sdp_enabled")() for name in ("flash","mem_efficient","math","cudnn")
                              if hasattr(cuda,name+"_sdp_enabled")})


def modules(model,dropout_type):
    named=list(model.named_modules())
    modes={name:bool(m.training) for name,m in named}
    dropout={name:dict(type=type(m).__name__,training=bool(m.training),p=float(m.p),inplace=bool(m.inplace))
             for name,m in named if isinstance(m,dropout_type)}
    require(0<len(modes)<=20000 and 0<len(dropout)<=4096,"real module-mode or dropout inventory differs")
    return dict(root_training=bool(model.training),training_modes_SHA256=digest(modes),module_count=len(modes),
                training_count=sum(modes.values()),dropout=dropout)


def profile_operators(rows):
    selected={}
    wrappers={}
    dropout={}
    for row in rows:
        key=row.key
        require(type(row.count) is int and row.count>0,"invalid CPU operator count")
        lower=key.lower()
        require(not (key.startswith("autograd::engine::evaluate_function") or
                     key.startswith("aten::") and "backward" in lower),"backward observed before the requested stop")
        if key.startswith("aten::_scaled_dot_product"):
            selected[key]=row.count
        elif key.startswith("aten::scaled_dot_product"):
            wrappers[key]=row.count
        elif key in ("aten::dropout","aten::native_dropout","aten::_fused_dropout","aten::feature_dropout"):
            dropout[key]=row.count
    require(len(selected)+len(wrappers)+len(dropout)<=64,"excess selected operator inventory")
    return dict(selected_SDPA=selected,SDPA_wrappers=wrappers,native_dropout=dropout,
                SDPA_route_observed=bool(selected))


def callable_source(method):
    file=inspect.getsourcefile(method)
    require(file is not None,"native compute_loss source cannot be verified")
    with Path(file).open("rb") as stream:
        raw=stream.read(2*1024**2+1)
    require(0<len(raw)<=2*1024**2,"native compute_loss source exceeds bound")
    return dict(module=method.__module__,name=method.__qualname__,SHA256=sha256(raw).hexdigest())


def await_owned_cohort(root,rank,identity,check_time,*,seconds=45,clock=time.monotonic,sleep=time.sleep):
    """Allow success only after all eight peers finished their existing forward."""
    local=dict(rank=rank,world_size=8,checkpoint_step=20,run=identity,state="FORWARD_RETURNED")
    marker=root/"first-forward-ready.json"
    require(not marker.exists(),"owner forward-ready marker already exists")
    pending=marker.with_suffix(".tmp")
    stream=pending.open("x",encoding="utf8")
    try:
        with stream:
            json.dump(local,stream,sort_keys=True,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending,marker)
    finally:
        pending.unlink()
    until=clock()+seconds
    missing=set(range(8))
    while missing:
        check_time()
        if clock()>=until:
            raise TimeoutError("missing original zero-update ranks: "+",".join(map(str,sorted(missing))))
        for peer in tuple(missing):
            path=root.parent/f"rank-{peer:02d}"/marker.name
            try:
                with path.open("rb") as stream:
                    raw=stream.read(4097)
            except FileNotFoundError:
                continue
            require(0<len(raw)<=4096 and json.loads(raw)==dict(local,rank=peer),"foreign peer forward-ready marker")
            missing.remove(peer)
        if missing:
            sleep(.05)


class FirstForward:
    """Attach after the original TreasuryResume hooks without changing its modes or RNG."""

    def __init__(self, callback, torch, rng, profiler, cohort):
        self.callback,self.torch,self.rng=callback,torch,rng
        self.profiler,self.cohort=profiler,cohort
        self.trainer,self.witness,self.model=callback.trainer,callback.witness,callback.model
        self.entered=0
        self.root_context=self.inner_context=self.language_context=None
        self.handles=[]
        self.original=self.trainer.compute_loss
        self.source=callable_source(self.original)
        self.original_instance=self.trainer.__dict__.get("compute_loss")
        self.had_instance="compute_loss" in self.trainer.__dict__
        self.bound_call=self.compute_once

    def attach(self):
        require(callable(self.original) and self.trainer.state.global_step==20,
                "native Trainer compute_loss or checkpoint-20 identity differs")
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.before_root,prepend=True,with_kwargs=True))
            language=self.model.get_base_model().model.language_model
            require(language.config._attn_implementation=="sdpa","native language attention configuration changed")
            self.handles.append(language.register_forward_pre_hook(self.before_language,prepend=True))
            for name,module in self.model.named_modules():
                if isinstance(module,self.torch.nn.Linear):
                    def inside(_module,_args,*,path=name):
                        if self.inner_context is None:
                            self.inner_context=dict(module=path,context=context(self.torch))
                    self.handles.append(module.register_forward_pre_hook(inside,prepend=True))
            require(len(self.handles)>2,"no real inner Linear autocast witness")
            self.trainer.compute_loss=self.bound_call
        except BaseException:
            self.detach()
            raise

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
        if self.trainer.compute_loss is self.bound_call:
            if self.had_instance:
                self.trainer.compute_loss=self.original_instance
            else:
                del self.trainer.compute_loss

    def before_root(self,_module,_argv,_kwargs):
        self.entered+=1
        self.callback.check_time()
        require(self.entered==1 and self.trainer.state.global_step==20 and self.callback.skip_seen
                and len(self.witness.rng_loads)==1 and not self.witness.inputs,
                "root forward is not the first native resumed microbatch")
        self.root_context=dict(RNG=self.rng(),context=context(self.torch),
                              model=modules(self.model,self.torch.nn.modules.dropout._DropoutNd))

    def before_language(self,module,_args):
        if self.language_context is None:
            self.language_context=dict(context=context(self.torch),training=bool(module.training),
                                       attention_config=module.config._attn_implementation)

    def compute_once(self,*args,**kwargs):
        try:
            self.callback.check_time()
            require(self.entered==0 and self.trainer.state.global_step==20 and not self.witness.steps,
                    "first native compute_loss boundary differs")
            entering_rng=self.rng()
            wrapped=self.trainer.model_wrapped
            ddp={k:getattr(wrapped,k,None) for k in ("require_backward_grad_sync","require_forward_param_sync","broadcast_buffers")}
            ddp["accelerator_sync_gradients"]=self.trainer.accelerator.sync_gradients
            ddp["wrapped_model"]=type(wrapped).__module__+"."+type(wrapped).__qualname__
            with self.profiler() as profile:
                self.original(*args,**kwargs)
            self.callback.check_time()
            require(self.entered==1 and all(row is not None for row in (self.root_context,self.inner_context,self.language_context))
                    and len(self.witness.inputs)==len(self.witness.outputs)==1 and not self.witness.steps
                    and self.trainer.state.global_step==20, "original first forward/loss boundary incomplete")
            leaving_rng=self.rng()
            original_input=self.witness.inputs[0]
            original_loss=self.witness.outputs[0]
            require(original_input["ordinal"]==original_loss["ordinal"]==1 and original_input["global_step"]==20
                    and original_input["runtime_cursor"]["micro_in_update"]==1
                    and math.isfinite(original_loss["loss_scalar"]), "native first input/loss receipt differs")
            restored=self.witness.training_state()
            require(all(restored[k]==self.witness.expected[k] for k in ("adapter","optimizer","scheduler"))
                    and not any(p.grad is not None and bool(self.torch.count_nonzero(p.grad))
                                for p in self.model.parameters() if p.requires_grad), "model or optimizer mutated before stop")
            evidence=dict(rank=self.callback.rank,world_size=8,state="STOP_BEFORE_BACKWARD",checkpoint_step=20,
                          global_step=20,planned_forwards=1,completed_forwards=1,backward_calls=0,optimizer_updates=0,
                          actual_input=original_input,actual_loss=original_loss,restored_state=restored,
                          RNG_after_native_restore=self.witness.rng_loads[0]["actual"],RNG_before_compute_loss=entering_rng,
                          root_before_forward=self.root_context,inner_linear=self.inner_context,RNG_after_forward=leaving_rng,
                          language_before_forward=self.language_context,
                          ddp=ddp,compute_loss_source=self.source,operators=profile_operators(profile.key_averages()))
            self.witness.write("first-forward-state.json",evidence)
            self.detach()
            self.witness.detach()
            self.cohort()
            self.witness.write("first-forward-stop.json",dict(rank=self.callback.rank,state="ALL_RANKS_STOPPED",
                                                              global_step=20,backward_calls=0,optimizer_updates=0))
        except Exception as error:
            self.detach()
            self.witness.fail(error)
            raise
        raise SystemExit(0)
