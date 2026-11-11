"""Observe two native optimizer updates without changing the frozen MiniCPM recipe."""
import hashlib,json,math,os,random,signal,time,sys
from pathlib import Path
from deadline_contract import remaining
from recipe_model_contract import validate_native_recipe
epoch=float(os.environ["FIRST_UPDATE_EPOCH"])
deadline=float(os.environ["FIRST_UPDATE_DEADLINE"])
budget=remaining(epoch,deadline,time.time())
def capture_expired(*args):raise TimeoutError("registered native capture deadline reached")
signal.signal(signal.SIGALRM,capture_expired)
signal.alarm(max(1,int(budget)))
import numpy as np
import torch
from swift.callbacks import TrainerCallback,callbacks_map
KEY="original_first_updates"
def cpu(value):
    if isinstance(value,torch.Tensor):
        result=value.detach().cpu().clone()
        if result.is_floating_point() and not torch.isfinite(result).all():raise RuntimeError("nonfinite captured tensor")
        return result
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(x) for x in value]
    if isinstance(value,tuple):return tuple(cpu(x) for x in value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError("unsupported captured object "+type(value).__name__)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for raw in iter(lambda:f.read(1024*1024),b""):h.update(raw)
    return h.hexdigest()
class FirstUpdates(TrainerCallback):
    def __init__(self,swift_args,trainer):
        super().__init__(swift_args,trainer)
        self.trainer=trainer
        self.swift_args=swift_args
        self.files=[];self.inputs=0;self.outputs=0;self.updates=0;self.clip_calls=0;self.capture_bytes=0
    def save(self,name,value):
        path=self.root/name
        assert not path.exists()
        with path.open("xb") as f:torch.save(value,f)
        assert path.stat().st_size<2*1024**3,"Per-artifact full-payload budget exceeded"
        self.capture_bytes+=path.stat().st_size
        assert self.capture_bytes<=8*1024**3,"Per-rank full-payload budget exceeded"
        self.files.append({"name":name,"bytes":path.stat().st_size,"SHA256":digest(path)})
        self.progress()
    def progress(self):
        row={"rank":self.rank,"planned_updates":2,"completed_updates":self.updates,
             "inputs_saved":self.inputs,"outputs_saved":self.outputs,"files":self.files}
        temp=self.root/"progress.tmp"
        temp.write_bytes(json.dumps(row).encode());temp.replace(self.root/"progress.json")
    def parameters(self):
        return {name:cpu(p) for name,p in self.named}
    def gradients(self):
        return {name:None if p.grad is None else cpu(p.grad) for name,p in self.named}
    def rng(self):
        state=np.random.get_state()
        numpy_state={"algorithm":state[0],"keys":state[1].tolist(),"position":int(state[2]),
                     "has_gauss":int(state[3]),"cached_gaussian":float(state[4])}
        return {"python":random.getstate(),"numpy":numpy_state,
                "torch_CPU":torch.get_rng_state(),"torch_device":torch.cuda.get_rng_state(),
                "device_index":torch.cuda.current_device()}
    def on_train_begin(self,args,state,control,**kw):
        assert state.global_step==0 and args.max_steps==25 and args.gradient_accumulation_steps==4
        assert args.per_device_train_batch_size==1 and args.warmup_steps==5
        self.rank=int(os.environ["RANK"]);assert int(os.environ["WORLD_SIZE"])==8 and 0<=self.rank<8
        self.root=Path(os.environ["FIRST_UPDATE_CAPTURE"])/("rank-%02d"%self.rank)
        self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        with (self.root/"owner.json").open("x") as f:
            json.dump({"PID":os.getpid(),"PGID":os.getpgrp(),
                       "startticks":int(Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19]),
                       "epoch":epoch,"deadline":deadline,"rank":self.rank},f)
        runtime=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-fast-overlay-v1")
        providers={}
        for module_name in ("triton","triton._C.libtriton","causal_conv1d_cuda","fla.ops.gated_delta_rule"):
            assert module_name in sys.modules,module_name
            path=Path(sys.modules[module_name].__file__).resolve()
            expected=runtime/("triton-v1/site" if module_name.startswith("triton") else "site")
            assert path.is_relative_to(expected),module_name
            providers[module_name]={"path":str(path),"SHA256":digest(path)}
        with (self.root/"runtime-providers.json").open("x") as f:json.dump(providers,f,indent=2)
        self.named,frozen=validate_native_recipe(kw.get("model",self.trainer.model),args,10822656)
        assert self.named and all(p.device.type=="cuda" for _,p in self.named)
        self.names={id(p):name for name,p in self.named}
        self.progress()
        self.save("initial.pt",{"parameters":self.parameters(),"RNG":self.rng(),
            "manifest":[{"name":n,"shape":list(p.shape),"dtype":str(p.dtype),"numel":p.numel()} for n,p in self.named],
            "global_step":state.global_step})
        model=kw.get("model",self.trainer.model)
        with (self.root/"frozen-boundaries.json").open("x") as f:json.dump(frozen,f,indent=2)
        def before(module,argv,kwargs):
            if self.trainer.state.global_step>=2:return
            self.inputs+=1
            self.save("forward-%02d-input.pt"%self.inputs,{"global_step":self.trainer.state.global_step,
                "micro_ordinal":self.inputs,"args":cpu(argv),"kwargs":cpu(kwargs),"RNG_before_forward":self.rng()})
        def after(module,argv,kwargs,result):
            if self.trainer.state.global_step>=2:return
            self.outputs+=1
            self.save("forward-%02d-output.pt"%self.outputs,{"global_step":self.trainer.state.global_step,
                "micro_ordinal":self.outputs,"result":cpu(dict(result))})
        self.before=model.register_forward_pre_hook(before,with_kwargs=True)
        self.after=model.register_forward_hook(after,with_kwargs=True)
        self.original_clip=self.trainer._clip_grad_norm
        def clipped(wrapped_model):
            if self.trainer.state.global_step<2:
                self.clip_calls+=1
                self.save("update-%02d-preclip.pt"%(self.trainer.state.global_step+1),{"gradients":self.gradients(),
                    "global_step":self.trainer.state.global_step,"max_grad_norm":args.max_grad_norm})
            result=self.original_clip(wrapped_model)
            if self.trainer.state.global_step<2:
                self.save("update-%02d-postclip.pt"%(self.trainer.state.global_step+1),{"gradients":self.gradients(),
                    "reported_preclip_norm":cpu(result),"global_step":self.trainer.state.global_step})
            return result
        self.trainer._clip_grad_norm=clipped
        return control
    def optimizer(self):
        opt=self.trainer.optimizer
        groups=[]
        for group in opt.param_groups:
            names=[self.names[id(p)] for p in group["params"]]
            groups.append({"parameter_names":names,**{k:cpu(v) for k,v in group.items() if k!="params"}})
        state={name:cpu(opt.state.get(p,{})) for name,p in self.named}
        return {"state_by_name":state,"param_groups":groups}
    def on_pre_optimizer_step(self,args,state,control,**kw):
        if state.global_step<2:
            self.save("update-%02d-before-optimizer.pt"%(state.global_step+1),{"parameters":self.parameters(),
                "optimizer":self.optimizer(),"global_step":state.global_step,"RNG":self.rng()})
        return control
    def on_optimizer_step(self,args,state,control,**kw):
        if state.global_step<2:
            self.updates+=1
            self.save("update-%02d-after-optimizer.pt"%(state.global_step+1),{"parameters":self.parameters(),
                "optimizer":self.optimizer(),"global_step_before_increment":state.global_step,"RNG":self.rng()})
        return control
    def on_step_end(self,args,state,control,**kw):
        if state.global_step==2:
            assert self.inputs==self.outputs==8 and self.updates==self.clip_calls==2
            self.before.remove();self.after.remove();self.trainer._clip_grad_norm=self.original_clip
            receipt={"state":"TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED","rank":self.rank,"world_size":8,
                     "completed_global_step":2,"micro_forwards":8,"optimizer_updates":2,"clip_calls":2,
                     "files":self.files,"note":"First LR warmup update can have zero parameter delta; capture the following positive-LR update as well. Full native 25-update run must still finish."}
            with (self.root/"receipt.json").open("x") as f:json.dump(receipt,f,indent=2)
        return control
    def on_train_end(self,args,state,control,**kw):
        assert state.global_step==25 and self.updates==2
        with (self.root/"native25-complete.json").open("x") as f:json.dump({"rank":self.rank,"global_step":state.global_step},f)
        return control
assert KEY not in callbacks_map
callbacks_map[KEY]=FirstUpdates
