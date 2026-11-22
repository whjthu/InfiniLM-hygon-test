"""Swift external plugin for the fresh, zero-update native checkpoint-20 scope."""

from hashlib import sha256
import os
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch.profiler import ProfilerActivity, profile
from swift.callbacks import callbacks_map


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
try:
    from probe import FirstForward, await_owned_cohort, require
finally:
    sys.path.pop(0)
require(Path(sys.modules["probe"].__file__).resolve()==HERE/"probe.py","another probe is already imported")

PARENT_SHA="17498637d7f7e2a2a5d9faf1add5c020903a74ece595d16ed8a20261d45378d8"
UPDATE_SHA="e973c0e48f9538996953922cfaf7e5c1f7c3a2aed81fb3f8be71c95d0afa6f95"
KEY="minicpm_usgs_resume_witness"


def read_pin(path,sha,size):
    with path.open("rb") as handle:
        value=handle.read(size+1)
    require(len(value)==size and sha256(value).hexdigest()==sha,"frozen MiniCPM observer dependency differs")
    return value


def load_parent(path):
    value=read_pin(path,PARENT_SHA,12581)
    read_pin(path.with_name("update_contract.py"),UPDATE_SHA,1438)
    require(KEY not in callbacks_map,"the parent callback already exists in this new scope")
    parent=ModuleType("minicpm_usgs_frozen_parent")
    parent.__file__=str(path)
    sys.path.insert(0,str(path.parent))
    try:
        exec(compile(value,str(path),"exec"),parent.__dict__)
    finally:
        sys.path.pop(0)
    require(callbacks_map.get(KEY) is parent.TreasuryResume,"wrong original MiniCPM resume callback")
    return parent


def bind(path):
    parent=load_parent(path)

    class NativeFirstResumedForward(parent.TreasuryResume):
        def on_train_begin(self,args,state,control,**kw):
            result=super().on_train_begin(args,state,control,**kw)
            native_skip=parent.trainer_module.skip_first_batches

            def rng():
                return parent.rng_fingerprint(parent.rng_values(torch.cuda.get_rng_state_all))

            def cohort():
                self.check_time()
                require(torch.distributed.is_initialized() and torch.distributed.get_world_size()==8
                        and torch.distributed.get_rank()==self.rank,"original native eight-rank cohort differs")
                torch.cuda.synchronize()
                require(parent.trainer_module.skip_first_batches is native_skip and native_skip is not self.original_skip
                        and self.skip_seen,
                        "original native skip was not observed")
                run=dict(config_SHA256=os.environ["MINICPM_USGS_CONFIG_SHA256"],epoch=self.epoch,deadline=self.deadline,
                         actual_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"])
                await_owned_cohort(self.root,self.rank,run,self.check_time)
                parent.trainer_module.skip_first_batches=self.original_skip

            try:
                self.observer=FirstForward(self,torch,rng,lambda:profile(activities=[ProfilerActivity.CPU]),cohort)
                self.observer.attach()
            except Exception as error:
                if parent.trainer_module.skip_first_batches is native_skip:
                    parent.trainer_module.skip_first_batches=self.original_skip
                self.witness.fail(error)
                raise
            return result

        def on_step_end(self,*args,**kwargs):
            raise RuntimeError("native first-forward diagnostic reached optimizer step_end")

        def on_save(self,*args,**kwargs):
            raise RuntimeError("native first-forward diagnostic reached checkpoint save")

        def on_train_end(self,*args,**kwargs):
            raise RuntimeError("native first-forward diagnostic missed its stop boundary")

    callbacks_map[KEY]=NativeFirstResumedForward


bind(Path(os.environ["MINICPM_FIRST_FORWARD_FROZEN_WITNESS"]).resolve(strict=True))
