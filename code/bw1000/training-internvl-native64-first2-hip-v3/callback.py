"""Private bounded observation of the first two real Swift optimizer updates."""
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from scope import (CALLBACK, CAPTURE_NAMES, INITIAL, INITIAL_BYTES, INITIAL_SHA,
                   NAMED_SHA, NATIVE, REQUEST, finish_step, need, pin, write_once)

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from vlm.plugins.native_first_updates import rng
from vlm.runtime.checkpointing_lifecycle import validate_checkpointing_lifecycle
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import FirstUpdateCapture
from vlm.runtime.initial_trainables import named_parameter_sha256, restore_initial_trainables
from vlm.runtime.native_caller_budget import validate_capture_conditions
from vlm.runtime.native_recipe_contract import validate_native_recipe
from vlm.runtime.provider_sources import model_provider_sources


class MetadataCapture(FirstUpdateCapture):
    """Keep each normal capture point, with hashes and statistics in place of tensor dumps."""
    def save(self, name, value):
        self.check_deadline()
        need(name in CAPTURE_NAMES and all(row["name"]!=name for row in self.files),"Unexpected or duplicate capture point")
        parts = {key:fingerprint(part) for key,part in value.items()}
        document = dict(point=name,fields=parts,rank=self.rank,request=REQUEST,
                        raw_tensor_bytes_written=0)
        if name.endswith("-input.pt"):
            document["native_input_SHA256"]=fingerprint({key:value[key] for key in ("args","kwargs")})["SHA256"]
        if name.endswith("-output.pt") and value["result"].get("loss") is not None:
            loss = value["result"]["loss"]
            document["loss"] = float(loss.item()) if isinstance(loss,torch.Tensor) else float(loss)
        path = self.root/(name.removesuffix(".pt")+".json")
        sha = write_once(path,document)
        self.files.append(dict(name=name,bytes=path.stat().st_size,SHA256=sha))
        self.progress()

    def step_end(self):
        self.check_deadline()
        if self.trainer.state.global_step == 2:
            need(not self.failed,"Native capture previously failed")
            finish_step(2, updates=self.updates,inputs=self.inputs,outputs=self.outputs,clips=self.clip_calls,
                        names=[row["name"] for row in self.files])
            self.detach()

    def train_end(self):
        self.check_deadline()
        need(not self.failed and self.detached and self.trainer.state.global_step==2,
             "Native trainer did not stop normally after two real updates")
        self.write_json("terminal.json",dict(rank=self.rank,world_size=8,request=REQUEST,
            actual_stop_step=2,native_schedule_steps=25,optimizer_updates=self.updates,
            micro_forwards=self.inputs,clip_calls=self.clip_calls,files=self.files,
            raw_tensor_bytes_written=0,rank00_trainable_initial_digest=NAMED_SHA,
            cross_platform_RNG_equivalent=False,formal_numerical_alignment=False))


def distributed_initialization(named, trainer, rank, step):
    """Validate and apply the original rank 0 archive once, then verify each native replica."""
    dist=torch.distributed
    flag=torch.empty((),dtype=torch.int32,device=named[0][1].device)

    def agrees(ok):
        flag.fill_(int(ok))
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        return bool(flag.item())

    optimizer=trainer.optimizer
    expected={id(value) for _,value in named}
    actual=[id(value) for group in optimizer.param_groups for value in group["params"]]
    need(agrees(step==0 and not optimizer.state and len(actual)==504 and set(actual)==expected
          and all(value.grad is None for _,value in named)),"Fresh target optimizer/gradients differ across ranks")
    error=None
    if rank==0:
        try:
            restore_initial_trainables(named,optimizer,global_step=step,path=INITIAL,file_bytes=INITIAL_BYTES,
                                       file_SHA256=INITIAL_SHA,named_SHA256=NAMED_SHA)
        except Exception as failure:
            error=failure
    if not agrees(error is None):
        raise ValueError("Original rank-00 initialization was rejected") from error
    with torch.no_grad():
        for _,parameter in named:
            dist.broadcast(parameter,src=0)
    need(agrees(named_parameter_sha256(named)==NAMED_SHA),"Broadcast source values differ on a native replica")
    return agrees


class InternVLFirstTwo(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args,trainer)
        self.trainer=trainer
        self.capture=None

    def on_train_begin(self,args,state,control,**kwargs):
        epoch=float(os.environ["T03_FIRST2_EPOCH"])
        deadline=float(os.environ["T03_FIRST2_DEADLINE"])
        binding=json.loads(pin(Path(os.environ["T03_FIRST2_BINDING"]),expected=os.environ["T03_FIRST2_BINDING_SHA256"]))
        rank=int(os.environ["RANK"])
        local=int(os.environ["LOCAL_RANK"])
        dist=torch.distributed
        need(deadline==epoch+NATIVE and time.time()<deadline and torch.version.hip=="6.3.26113"
             and torch.cuda.device_count()==8 and 0<=rank==local<8 and torch.cuda.current_device()==local
             and int(os.environ["WORLD_SIZE"])==8 and dist.is_initialized()
             and (dist.get_rank(),dist.get_world_size())==(rank,8),"Actual HIP rank and original epoch differ")
        model=kwargs.get("model",self.trainer.model)
        method=getattr(self.trainer,"_prepare_gradient_checkpointing",None)
        need(self.trainer.template.template_meta.template_type=="internvl_hf" and args.max_steps==25,
             "Native InternVL template or original schedule differs")
        named,frozen=validate_native_recipe(model,args,binding,checkpointing_method=method)
        lifecycle=validate_checkpointing_lifecycle(model.get_base_model(),args,binding,method)
        validate_capture_conditions(model,binding["capture_budget_conditions"],
                                    native_logits_option=self.trainer.args.use_logits_to_keep)
        provider=model_provider_sources(model,binding["model_source_SHA256"],"T03.sft")
        need(len(named)==504 and all(p.device.type=="cuda" and p.device.index==local for _,p in named),
             "Native rank-owned 504 tensors required")
        agrees=distributed_initialization(named,self.trainer,rank,state.global_step)
        root=Path(os.environ["T03_FIRST2_CAPTURE"])/f"rank-{rank:02d}"
        self.capture=MetadataCapture(self.trainer,root,named,rank=rank,world_size=8,epoch=epoch,deadline=deadline,
            max_seconds=NATIVE,rng=rng,input_fingerprints=binding["input_fingerprints_by_rank"][str(rank)],
            input_agreement=agrees)
        proc=Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()
        self.capture.write_json("owner.json",dict(rank=rank,PID=os.getpid(),startticks=int(proc[19]),
            initial_trainables_SHA256=NAMED_SHA,
            checkpointing=lifecycle,frozen=frozen,target_provider=provider,A100_RNG_restored=False))
        def expired(*_):
            error=TimeoutError("Original T03 native budget")
            self.capture.fail(error)
            raise error
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
        self.capture.guarded(self.capture.attach)(model)
        return control

    def on_step_begin(self,args,state,control,**kwargs):
        need(state.global_step<2,"Unexpected third native optimizer step")
        return control

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        self.capture.guarded(self.capture.before_optimizer)()
        return control

    def on_optimizer_step(self,args,state,control,**kwargs):
        self.capture.guarded(self.capture.after_optimizer)()
        return control

    def on_step_end(self,args,state,control,**kwargs):
        self.capture.guarded(self.capture.step_end)()
        if state.global_step==2:
            control.should_training_stop=True
        return control

    def on_train_end(self,args,state,control,**kwargs):
        self.capture.guarded(self.capture.train_end)()
        return control


need(CALLBACK not in callbacks_map,"Repeated InternVL callback registration")
callbacks_map[CALLBACK]=InternVLFirstTwo
