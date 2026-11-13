"""Exercise the staged SAGE scheduler in the target training container."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import torch

path=Path("/sage/hyvideo/schedulers/scheduling_flow_match_discrete.py")
pin="56f61a5b36275528392d04c259902642291bcfcb69c9f4f4f4f076d0c148b943"
assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
assert torch.version.hip and not torch.cuda.is_initialized()
assert all(os.environ[name]=="-1" for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"))
assert torch.cuda.device_count()==0
spec=importlib.util.spec_from_file_location("gencuda_sage_scheduler",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scheduler=module.FlowMatchDiscreteScheduler(shift=5.0,reverse=True,solver="euler")
scheduler.set_timesteps(4,device="cpu")
sigmas=scheduler.sigmas
assert torch.equal(scheduler.timesteps,torch.tensor([1000.,937.5,833.3333129882812,625.]))
assert torch.equal(sigmas,torch.tensor([1.,.9375,.8333333134651184,.625,0.]))
schedule_timesteps=sigmas[:-1]*scheduler.config.num_train_timesteps
weights=scheduler.compute_grad_balancing_factors(eta=.7,device="cpu",sde_type="sage_grpo",sde_solver=True,sigmas=sigmas)
assert len(weights)==4 and torch.isfinite(weights).all() and (weights>0).all()
sample=torch.linspace(-.375,.375,16).reshape(1,2,2,2,2)
previous=sample.flip(-1)*.5
rows=[]
for index,timestep in enumerate(schedule_timesteps):
    replay=module.FlowMatchDiscreteScheduler(shift=5.0,reverse=True,solver="euler")
    replay.num_inference_steps=4
    replay.sigmas=sigmas.clone()
    replay.timesteps=schedule_timesteps
    replay._step_index=index
    prediction=(sample*.25+0.1).requires_grad_()
    next_sample,_,log_prob,mean,std=replay.sde_step_with_logprob(
        prediction,sample,timestep,.7,prev_sample=previous,grpo=True,sde_solver=True,sde_type="sage_grpo")
    gradient,=torch.autograd.grad(log_prob.sum(),prediction)
    assert torch.equal(next_sample,previous) and replay.step_index==index+1
    assert log_prob.shape==(1,) and gradient.shape==sample.shape
    assert all(torch.isfinite(x).all() for x in (log_prob,mean,std,gradient)) and std.item()>0
    assert torch.count_nonzero(gradient)>0
    rows.append(dict(index=index,log_prob=log_prob.item(),mean_sum=mean.sum().item(),std=std.item(),
                     gradient=gradient.flatten().tolist()))
assert not torch.cuda.is_initialized()
result=dict(passed=True,owner=dict(step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],node=socket.gethostname()),
            scheduler_SHA256=pin,torch=torch.__version__,torch_hip=torch.version.hip,torch_path=torch.__file__,
            diffusers_path=sys.modules["diffusers"].__file__,timesteps=scheduler.timesteps.tolist(),
            sigmas=sigmas.tolist(),balancing=weights.tolist(),rows=rows,fixture="synthetic_cpu",GPU_cards=0)
with Path("/evidence/CPU-result.json").open("x") as out:json.dump(result,out,allow_nan=False)
