"""Audit a single native continuation from the observed CP3 cursor through CP25."""
import functools
import inspect
import json
import math
import os
from pathlib import Path

import torch
from swift.callbacks import callbacks_map
from vlm.runtime.first_update_audit import fingerprint
from vlm.runtime.first_update_capture import cpu
from vlm.runtime.native_recipe_contract import validate_native_recipe

from domain_checks import PREPARE_SHA, after_sample, before_sample, verified_split
from resume_common import ContinueOnce, parameters, verify_resume
from scope import CALLBACK, REQUEST, need, pin, write_once
import series_contract as contract
from series_live import SeriesCursor


class ContinueSeries(ContinueOnce):
    def on_train_begin(self,args,state,control,**kwargs):
        self.clock()
        need((state.global_step,state.max_steps,state.epoch)==(3,25,1.5) and
             args.max_steps==25 and args.gradient_accumulation_steps==4 and args.per_device_train_batch_size==1
             and args.data_seed==42 and args.dataloader_num_workers==4 and args.ignore_data_skip is False
             and args.resume_only_model is False,"EXACT_NATIVE_CP3_RESUME_CURSOR_AND_RECIPE")
        need(torch.version.hip=="6.3.26113" and torch.cuda.device_count()==8 and
             (self.rank,8)==(torch.distributed.get_rank(),torch.distributed.get_world_size()) and
             self.rank==int(os.environ["LOCAL_RANK"])==torch.cuda.current_device(),"ORIGINAL_HIP_EIGHT_RANKS")
        binding=json.loads(pin(Path(os.environ["T03_FIRST2_BINDING"]),
                               expected=os.environ["T03_FIRST2_BINDING_SHA256"]))
        self.model=kwargs["model"]
        self.named,_=validate_native_recipe(self.model,args,binding,
                  checkpointing_method=self.trainer._prepare_gradient_checkpointing)
        self.mapping=self.consensus(lambda:verify_resume(self.trainer,self.named))
        self.agree(self.mapping,"EIGHT_RANK_ADAMW_NAME_MAPPING")
        self.output.mkdir(mode=0o700)
        self.roster,original,order=verified_split()
        plans=contract.load_plan(contract.CPU_ROOT,pin)
        contract.verify_epochs_from_live(plans,order,original)
        self.cursor=SeriesCursor(self.rank,plans[self.rank],order)
        self.consensus(lambda:self.cursor.attach(kwargs["train_dataloader"],pin,torch.distributed))
        pin(Path(inspect.getsourcefile(self.original_prepare)),expected=PREPARE_SHA,extent=12688)
        self.checkpoints={}
        self.last_hash=contract.CP3_FP
        self.saves=[]
        self.records=[]
        self.losses=[]

        @functools.wraps(self.original_prepare)
        def prepare(inputs):
            self.clock()
            def observe():
                offset=len(self.records)*4+len(self.losses)
                need(self.rng_calls==1 and self.pending is None and len(self.losses)<4 and offset<88,
                     "EXACT_PREPARE_AFTER_ONE_NATIVE_CP3_RNG_RESTORE")
                source=before_sample(inputs,self.roster,lambda value:fingerprint(cpu(value)))
                return self.cursor.prepare(offset,source)
            self.pending=self.consensus(observe)
            return self.original_prepare(inputs)

        self.trainer._prepare_inputs=prepare
        self.forward=self.model.register_forward_pre_hook(self.before_forward,with_kwargs=True)
        self.forward_result=self.model.register_forward_hook(self.after_forward)
        write_once(self.output/"restore.json",dict(request=REQUEST,rank=self.rank,step=3,
                   adapter_sha256=contract.CP3_FP,restored_named_optimizer_states=504,
                   optimizer_index_to_name_SHA256=self.mapping))
        return control

    def agree(self,value,gate):
        agreed=[None]*8
        torch.distributed.all_gather_object(agreed,value)
        need(agreed==[value]*8,gate)

    def before_forward(self,model,args,kwargs):
        self.clock()
        def observe():
            need(self.pending is not None and len(self.losses)<4,"EXACT_PREPARE_FORWARD_PAIR")
            observed=after_sample(kwargs,self.pending,lambda value:fingerprint(cpu(value)))
            observed.update(logical=self.pending["logical"],physical=self.pending["physical"])
            self.cursor.forward(len(self.records)*4+len(self.losses),observed)
            self.pending=None
        self.consensus(observe)

    def after_forward(self,model,inputs,output):
        def observe():
            loss=output["loss"]
            need(torch.is_tensor(loss) and loss.numel()==1 and bool(torch.isfinite(loss).all()),
                 "ACTUAL_FINITE_NATIVE_FORWARD_LOSS")
            return float(loss.detach().item())
        self.losses.append(self.consensus(observe))
        step=4+len(self.records)
        write_once(self.output/f"forward-{step}-{len(self.losses)}.json",
                   dict(request=REQUEST,rank=self.rank,step=step,accumulation=len(self.losses)))

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        self.clock()
        def observe():
            step=4+len(self.records)
            opt=self.trainer.optimizer
            need(step in contract.STEPS and state.global_step==step-1 and self.pending is None and
                 len(self.losses)==4 and len(self.cursor.consumed)==(step-3)*4 and self.rng_calls==1 and
                 all(row["update"]==step for row in self.cursor.consumed[-4:]) and len(opt.state)==504 and
                 all(opt.state[param]["step"].item()==step-1 for _,param in self.named),
                 "EXACT_NATIVE_OPTIMIZER_BEFORE_STEP_AND_MICROBATCHES")
            rates=[item["lr"] for item in opt.param_groups]
            need(rates and all(math.isfinite(lr) and lr==contract.lr(step-1) and lr>0 for lr in rates),
                 "EXACT_POSITIVE_NATIVE_SCHEDULER_LR")
            return rates
        self.rates=self.consensus(observe)
        return control

    def on_optimizer_step(self,args,state,control,**kwargs):
        step=4+len(self.records)
        def observe():
            need(state.global_step==step-1 and self.trainer.accelerator.optimizer_step_was_skipped is False and
                 all(self.trainer.optimizer.state[param]["step"].item()==step for _,param in self.named),
                 "EXACT_ACTUAL_COMPLETED_ADAMW_UPDATE")
        self.consensus(observe)
        if step in contract.SAVE_STEPS:
            value=self.consensus(lambda:parameters(self.named))
            self.agree(value,"EIGHT_RANK_NATIVE_CHECKPOINT_VALUE_HASHES")
            need(value!=self.last_hash,"NATIVE_CHECKPOINT_PARAMETER_CHANGE_REQUIRED")
            self.checkpoints[str(step)]=self.last_hash=value
        row=dict(step=step,rates=self.rates,losses=self.losses,slots=self.cursor.consumed[-4:])
        self.records.append(row)
        write_once(self.output/f"update-{step}.json",dict(request=REQUEST,rank=self.rank,**row))
        self.losses=[]
        return control

    def on_step_end(self,args,state,control,**kwargs):
        self.clock()
        step=state.global_step
        schedule=self.trainer.lr_scheduler.state_dict()
        self.consensus(lambda:need(self.records[-1]["step"]==step and schedule["last_epoch"]==step and
                      schedule["_step_count"]==step+1 and
                      all(lr==contract.lr(step) for lr in schedule["_last_lr"]),"EXACT_NATIVE_COSINE_CURSOR"))
        if step in contract.SAVE_STEPS:
            control.should_save=True
        if step==25:
            control.should_training_stop=True
        return control

    def on_save(self,args,state,control,**kwargs):
        need(state.global_step in contract.SAVE_STEPS and state.global_step not in self.saves,
             "EXACT_CP10_CP20_CP25_NATIVE_SAVE_EVENTS")
        self.saves.append(state.global_step)
        write_once(self.output/f"save-{state.global_step}.json",
                   dict(request=REQUEST,rank=self.rank,step=state.global_step,
                        parameter_SHA256=self.checkpoints[str(state.global_step)]))
        return control

    def on_train_end(self,args,state,control,**kwargs):
        self.clock()
        need(state.global_step==25 and self.rng_calls==1 and self.cursor.consumed==self.cursor.plan and
             self.cursor.epochs==list(contract.EPOCHS) and self.cursor.skip_consumed is True and
             self.saves==list(contract.SAVE_STEPS) and [row["step"] for row in self.records]==list(contract.STEPS),
             "EXACT_TERMINAL_22_UPDATES_AND_CHECKPOINTS")
        self.forward.remove()
        self.forward_result.remove()
        self.cursor.detach()
        self.trainer._prepare_inputs=self.original_prepare
        self.trainer._load_rng_state=self.original_rng
        write_once(self.output/"resume25.json",dict(request=REQUEST,rank=self.rank,final_global_step=25,
                   updates=22,forward_slots=88,native_skip_batches=4,device_rng_restore_calls=self.rng_calls,
                   optimizer_index_to_name_SHA256=self.mapping,checkpoint_steps=self.saves,checkpoints=self.checkpoints))
        return control


need(CALLBACK not in callbacks_map,"UNIQUE_NATIVE_CP25_CONTINUATION_CALLBACK")
callbacks_map[CALLBACK]=ContinueSeries
