"""New 88->800/full accumulation and step20 cursor checks; no Torch/data."""
import hashlib
import json
from pathlib import Path
from data_view import schedule
from mid_epoch_recovery import patch_cursor,BASE_PINS,CURSOR


def check(repo):
    order=schedule()
    assert len(order)==800 and set(order)==set(range(88))
    counts=[order.count(i) for i in range(88)]
    assert counts==[10]*8+[9]*80 and sum(counts)==800
    source=Path(repo)/"code/bw1000/training-vlm-resume-cursor-v1/framework/vlm/runtime/resume_cursor.py"
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_PINS[CURSOR]
    module={"__name__":"new_Qwen_exposure_cursor_CPU"}
    exec(compile(patch_cursor(raw),str(source),"exec"),module)
    plan=module["plan_epoch_boundary_resume"]
    state=dict(global_step=20,max_steps=25,train_batch_size=1,epoch=.8)
    args=dict(max_steps=25,per_device_train_batch_size=1,gradient_accumulation_steps=4,
              ignore_data_skip=False,resume_only_model=False,callbacks=["qwen_treasury_continuous_witness"])
    options=dict(checkpoint_step=20,max_steps=25,world_size=8,per_device_batch=1,
                 gradient_accumulation_steps=4,dataloader_batches_per_rank=100,allow_mid_epoch=True)
    result=plan(state,args,**options)
    assert result["expected_batches_to_skip_in_resumed_epoch"]==80 and result["planned_micro_forwards_per_rank"]==20
    assert result["effective_batch"]==32 and result["remaining_optimizer_updates"]==5
    rejected=0
    for changed_state,changed_options in ((dict(state,epoch=0),options),
            (state,dict(options,dataloader_batches_per_rank=11)),
            (state,dict(options,checkpoint_step=25))):
        try:
            plan(changed_state,args,**changed_options)
        except ValueError:
            rejected+=1
    assert rejected==3
    return dict(passed=True,valid_contracts=2,rejected=3,planned_rejections=3,
                source_rows=88,physical_exposures=800,new_independent_examples=0,
                intended_micro_per_rank=100,resume_skip_micro_per_rank=80,remaining_micro_per_rank=20,
                effective_batch=32,actual_native_processor=0,actual_Torch=0,actual_GPU=0,actual_updates=0)
