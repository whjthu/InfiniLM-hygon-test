"""Read the native microbatches and optimizer state without changing the stock sampler."""

import inspect
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from contract import check_compute_cache

sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence import (DATASET, DATA_SHA, GA, NATIVE_OUT, RANK_EPOCH_BATCHES, ROWS, WORLD,
                      dataset_rows, fingerprints, identify, native_report, need, read, schedule_position)
from vlm.runtime.first_update_capture import remaining
from vlm.runtime.initial_trainables import named_parameter_sha256
from vlm.runtime.native_recipe_contract import validate_native_recipe


KEY = "qwen_street_scene_first_two"
BINDING_SHA = "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"


def write(path, value):
    with path.open("x",encoding="utf8") as target:
        json.dump(value,target,sort_keys=True,allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def sampler(dataloader, rank):
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, SeedableRandomSampler
    from torch.utils.data import BatchSampler, RandomSampler
    shard=dataloader.batch_sampler
    need(type(dataloader) is DataLoaderShard and type(shard) is BatchSamplerShard
         and len(dataloader)==RANK_EPOCH_BATCHES and len(dataloader.dataset)==ROWS
         and shard.num_processes==WORLD and shard.process_index==rank and shard.batch_size==1
         and shard.split_batches is False and shard.even_batches is True and shard.drop_last is False
         and type(shard.batch_sampler) is BatchSampler,"STOCK_TRAINER_BATCH_SAMPLER_GEOMETRY_REQUIRED")
    underlying=shard.batch_sampler.sampler
    need(type(underlying) in (RandomSampler,SeedableRandomSampler) and underlying.replacement is False
         and underlying.num_samples==len(underlying.data_source)==ROWS and len(shard.batch_sampler)==ROWS,
         "STOCK_TRAINER_41_WITHOUT_REPLACEMENT_REQUIRED")
    classes=(type(dataloader),type(shard),type(underlying))
    return {cls.__module__+"."+cls.__name__:sha256(read(inspect.getfile(cls),1024**2)).hexdigest()
            for cls in classes}


def flattened(batch):
    ids=batch["input_ids"].detach().cpu().tolist()
    labels=batch["labels"].detach().cpu().tolist()
    attention=batch["attention_mask"].detach().cpu().tolist()
    grid=batch["image_grid_thw"].detach().cpu().tolist()
    need(len(ids)==len(labels)==len(attention)==len(grid)==1
         and len(ids[0])==len(labels[0])==len(attention[0]) and len(grid[0])==3,
         "ACTUAL_NATIVE_ONE_EXAMPLE_MICROBATCH_REQUIRED")
    mask=attention[0]
    length=sum(mask)
    need(0<length<=4096 and mask==[1]*length+[0]*(len(mask)-length)
         and all(label==-100 for label in labels[0][length:]),"ACTUAL_NATIVE_RIGHT_PADDING_REQUIRED")
    return ids[0][:length],labels[0][:length],grid[0]


def counters(optimizer, named):
    opt=getattr(optimizer,"optimizer",optimizer)
    need(type(opt) is torch.optim.AdamW,"ORIGINAL_ADAMW_PROVIDER_REQUIRED")
    all_parameters=[id(p) for group in opt.param_groups for p in group["params"]]
    need(len(all_parameters)==len(named) and set(all_parameters)=={id(p) for _,p in named},
         "EXACT_ORIGINAL_ADAMW_TRAINABLES")
    ordered=[]
    for name,p in named:
        value=opt.state.get(p,{}).get("step",0)
        if isinstance(value,torch.Tensor):
            value=value.item()
        need(math.isfinite(value) and value>=0 and int(value)==value,"INTEGER_ADAMW_STEP_COUNTER")
        ordered.append(int(value))
    return ordered


class NativeFirstUpdates(TrainerCallback):
    """Keep the original initialization callback ordering contract by class name."""

    def __init__(self, swift_args, trainer):
        super().__init__(swift_args,trainer)
        self.trainer=trainer
        self.rank=int(os.environ["RANK"])
        self.events=[]
        self.epoch=-1
        self.updates=0
        self.pending=None
        self.root=None
        self.original_step=None
        self.finished=False
        self.wall=(float(os.environ["FIRST_UPDATE_EPOCH"]),float(os.environ["FIRST_UPDATE_DEADLINE"]),
                   float(os.environ["FIRST_UPDATE_MAX_SECONDS"]))

    def check_time(self):
        remaining(self.wall[0],self.wall[1],time.time(),self.wall[2])

    def fail(self,error):
        if self.root is not None and not (self.root/"failure.json").exists():
            write(self.root/"failure.json",dict(error_type=type(error).__name__,rank=self.rank,
                  updates_observed=self.updates,batches_observed=len(self.events),planned_updates=2,
                  operation_pending=self.pending is not None))

    def on_train_begin(self,args,state,control,**kw):
        try:
            self.check_time()
            local=int(os.environ["LOCAL_RANK"])
            check_compute_cache()
            need(int(os.environ["WORLD_SIZE"])==WORLD and self.rank==local and 0<=local<WORLD
                 and torch.version.hip=="6.3.26113" and str(torch.__version__)=="2.10.0"
                 and torch.cuda.device_count()==WORLD
                 and torch.cuda.current_device()==local and state.global_step==0,
                 "FRESH_ORIGINAL_EIGHT_RANK_HIP_TRAINING_REQUIRED")
            self.root=Path(os.environ["FIRST_UPDATE_CAPTURE"])/("rank-%02d"%self.rank)
            self.root.mkdir(mode=0o700,parents=True)
            required={"ms-swift":"4.4.2","transformers":"5.12.1","peft":"0.19.1","qwen-vl-utils":"0.0.14"}
            runtime={name:version(name) for name in required}
            need(runtime==required,"EXACT_ORIGINAL_220_AND_1251_NATIVE_RUNTIME_VERSIONS")
            binding_path=Path(os.environ["FIRST_UPDATE_BINDING"])
            binding=json.loads(read(binding_path,2*1024**2,BINDING_SHA))
            model=kw.get("model",self.trainer.model)
            self.named,frozen=validate_native_recipe(model,args,binding)
            core=model.get_base_model()
            read(inspect.getfile(type(core)),2*1024**2,binding["model_source_SHA256"])
            initial=named_parameter_sha256(self.named)
            need(initial==binding["initial_trainables_SHA256"] and not any(counters(self.trainer.optimizer,self.named)),
                 "ORIGINAL_RESTORED_TRAINABLES_AND_FRESH_OPTIMIZER_REQUIRED")
            live_loader=kw.get("train_dataloader",self.trainer.callback_handler.train_dataloader)
            providers=sampler(live_loader,self.rank)
            dataset=Path(DATASET)
            need(Path.cwd().resolve()==dataset.parent,"EXACT_STAGED_TRAIN_CWD_REQUIRED")
            self.index=fingerprints(native_report(Path(NATIVE_OUT)),dataset_rows(dataset,dataset.parent/"MEDIA_PINS.json"))
            self.original_step=self.trainer.training_step
            self.trainer.training_step=self.training_step
            write(self.root/"initial.json",dict(rank=self.rank, world_size=WORLD, nominal_schedule_steps=25,
                  initial_trainables_SHA256=initial, original_model_source_SHA256=binding["model_source_SHA256"],
                  dataset_SHA256=DATA_SHA, frozen=frozen, sampler_provider_identities=providers,
                  runtime_versions=runtime, accelerate_version=version("accelerate"),
                  scheduler_initial_lrs=[float(g["lr"]) for g in self.trainer.optimizer.param_groups]))
            return control
        except BaseException as error:
            self.fail(error)
            raise

    def on_epoch_begin(self,args,state,control,**kw):
        try:
            self.check_time()
            self.epoch+=1
            need(len(self.events)==self.epoch*RANK_EPOCH_BATCHES and not self.finished,
                 "EXACT_NATIVE_EPOCH_BEGIN_BOUNDARY")
            write(self.root/("epoch-%02d.json"%self.epoch),dict(rank=self.rank,epoch=self.epoch,
                  global_step=state.global_step,rank_batches_before_epoch=len(self.events)))
            return control
        except BaseException as error:
            self.fail(error)
            raise

    def training_step(self, model, inputs, *args, **kwargs):
        try:
            self.check_time()
            old=self.trainer.state.global_step
            expected_epoch=len(self.events)//RANK_EPOCH_BATCHES
            state_epoch=self.trainer.state.epoch or 0.0
            need(not self.finished and old==self.updates and old<2 and expected_epoch==self.epoch
                 and 0<=expected_epoch-math.floor(state_epoch)<=1,
                 "EXACT_NATIVE_MICROBATCH_EPOCH_BOUNDARY")
            row=identify(self.index,*flattened(inputs))
            record={**row,**schedule_position(self.rank,len(self.events)),"optimizer_update":old+1,
                    "trainer_epoch_before_batch":state_epoch,
                    "accelerator_sync_gradients":self.trainer.accelerator.sync_gradients}
            write(self.root/("batch-%02d-input.json"%len(self.events)),record)
            loss=self.original_step(model,inputs,*args,**kwargs)
            loss_value=float(loss.detach().item())
            need(math.isfinite(loss_value),"FINITE_NATIVE_TRAINING_LOSS_REQUIRED")
            record["training_step_returned"]=True
            record["loss"]=loss_value
            write(self.root/("batch-%02d-done.json"%len(self.events)),record)
            self.events.append(record)
            return loss
        except BaseException as error:
            self.fail(error)
            raise

    def on_pre_optimizer_step(self,args,state,control,**kw):
        try:
            self.check_time()
            batch=[row for row in self.events if row["optimizer_update"]==state.global_step+1]
            need(self.pending is None and state.global_step==self.updates<2 and 1<=len(batch)<=GA and
                 batch[-1]["accelerator_sync_gradients"] is True and
                 all(row["epoch"]==batch[0]["epoch"] for row in batch) and
                 all(row["accelerator_sync_gradients"] is False for row in batch[:-1]),
                 "EXACT_NATIVE_ACCELERATOR_ACCUMULATION_BOUNDARY")
            opt=self.trainer.optimizer
            self.pending=dict(rank=self.rank,global_step=self.updates+1,batches=batch,
                parameter_count=len(self.named),parameters_before_SHA256=named_parameter_sha256(self.named),
                counters_before=counters(opt,self.named),lrs=[float(group["lr"]) for group in opt.param_groups])
            write(self.root/("update-%02d-before.json"%(self.updates+1)),self.pending)
            return control
        except BaseException as error:
            self.fail(error)
            raise

    def on_optimizer_step(self,args,state,control,**kw):
        try:
            self.check_time()
            need(self.pending is not None and state.global_step==self.updates<2,"ACTUAL_OPTIMIZER_SEQUENCE_REQUIRED")
            row=self.pending
            after=counters(self.trainer.optimizer,self.named)
            changed=named_parameter_sha256(self.named)
            advances=[b-a for a,b in zip(row["counters_before"],after)]
            skipped=self.trainer.accelerator.optimizer_step_was_skipped
            row.update(counters_after=after, counter_advance=not skipped and all(v==1 for v in advances),
                       optimizer_step_was_skipped=skipped, parameters_after_SHA256=changed,
                       parameter_bytes_changed=changed!=row["parameters_before_SHA256"])
            write(self.root/("update-%02d-after.json"%(self.updates+1)),row)
            return control
        except BaseException as error:
            self.fail(error)
            raise

    def on_step_end(self,args,state,control,**kw):
        try:
            self.check_time()
            need(self.pending is not None and state.global_step==self.updates+1 and
                 "parameters_after_SHA256" in self.pending,"ACTUAL_GLOBAL_STEP_INCREMENT_REQUIRED")
            self.updates+=1
            self.pending["trainer_epoch_after_update"]=state.epoch
            write(self.root/("update-%02d.json"%self.updates),self.pending)
            self.pending=None
            if self.updates==2:
                control.should_training_stop=True
                control.should_save=False
                control.should_evaluate=False
            return control
        except BaseException as error:
            self.fail(error)
            raise

    def on_train_end(self,args,state,control,**kw):
        try:
            self.check_time()
            need(self.updates==state.global_step==2 and args.max_steps==25 and control.should_training_stop
                 and self.pending is None,"EXACT_TWO_OBSERVED_UPDATES_AND_NOMINAL25_REQUIRED")
            write(self.root/"complete.json",dict(rank=self.rank,global_step=2,batches=len(self.events),
                  actual_optimizer_updates_observed=2,Model_GO=False))
            self.finished=True
            return control
        except BaseException as error:
            self.fail(error)
            raise
        finally:
            if self.original_step is not None:
                self.trainer.training_step=self.original_step


if KEY in callbacks_map:
    raise ValueError("duplicate street-scene native recorder")
callbacks_map[KEY]=NativeFirstUpdates
