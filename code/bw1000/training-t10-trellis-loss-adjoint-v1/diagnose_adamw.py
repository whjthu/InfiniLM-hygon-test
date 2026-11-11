"""Attribute the original TRELLIS first-step update discrepancy on CPU."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from hygon_sdxl_compare_first_step import exact

PINS={
    "/reference/reference.pt":"c164b02f2bd4edea0d266daebf7d93adef17e753589d95fb530f81b1fba9e612",
    "/update/replay/observed.pt":"b1acdf0b074c3c66639b566ab082544d7c8bfac63a487d05b0f4edc1a8bcd767",
    "/update/replay/summary.json":"10bb10922391036cce9afd18ac31b7d50d59f965ee56ca70e76ad2b076ce5393",
}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else sha_stream(stream)

def sha_stream(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b""):
        digest.update(chunk)
    return digest.hexdigest()

def squares(a,b):
    a=a.double()
    b=b.double()
    return float((a-b).square().sum()),float(b.square().sum())

def relative(a,b):
    error,signal=squares(a,b)
    return math.sqrt(error/signal) if signal else (0.0 if not error else None)

def update(initial,state,gradients):
    named={name:torch.nn.Parameter(x.clone()) for name,x in initial.items()}
    group,=state["param_groups"]
    options={k:v for k,v in group.items() if k not in ("params","initial_lr","decoupled_weight_decay")}
    assert group["decoupled_weight_decay"] is True and group["foreach"] is False and group["fused"] is False
    assert list(named)==list(gradients) and state["state"]=={}
    optimizer=torch.optim.AdamW(list(named.values()),**options)
    optimizer.load_state_dict(state)
    for name,param in named.items():
        param.grad=gradients[name].clone()
    optimizer.step()
    return {n:x.detach() for n,x in named.items()},optimizer.state_dict()

def compare_state(a,b):
    exact(torch,a["param_groups"],b["param_groups"],"AdamW final groups")
    exact(torch,list(a["state"]),list(b["state"]),"AdamW final state IDs")
    sums={key:[0.,0.] for key in ("exp_avg","exp_avg_sq")}
    for i,fields in b["state"].items():
        exact(torch,list(a["state"][i]),list(fields),"AdamW final state fields")
        for key in fields:
            if key in sums:
                d,s=squares(a["state"][i][key],fields[key])
                sums[key][0]+=d
                sums[key][1]+=s
            else:
                exact(torch,a["state"][i][key],fields[key],key)
    return {key:math.sqrt(d/s) for key,(d,s) in sums.items()}

def delta_metrics(a,b,initial):
    total=error=0.
    for name,start in initial.items():
        d,s=squares(a[name].double()-start.double(),b[name].double()-start.double())
        total+=s
        error+=d
    return {"relative_l2":math.sqrt(error/total),"error_l2_squared":error,"reference_update_l2_squared":total}

def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert all(sha(path)==pin for path,pin in PINS.items())
    ref,hip=[torch.load(path,map_location="cpu",weights_only=True) for path in list(PINS)[:2]]
    receipt=json.loads(Path(list(PINS)[2]).read_bytes())
    assert receipt["observed_sha256"]==PINS["/update/replay/observed.pt"]
    assert receipt["stage_id"]=="trellis-shape" and receipt["runtime"]["accelerator"]=="hip"
    for key in ("initial_adapter","initial_optimizer","inputs","prediction_coords"):
        exact(torch,hip[key],ref[key],key)
    initial=ref["initial_adapter"]
    assert len(initial)==24 and sum(x.numel() for x in initial.values())==96256
    grad_ref,grad_hip=ref["gradients_after_clip"],hip["gradients_after_clip"]
    cpu_ref,state_ref=update(initial,ref["initial_optimizer"],grad_ref)
    cpu_hip,state_hip=update(initial,hip["initial_optimizer"],grad_hip)
    categories={k:{"elements":0,"update_error_l2_squared":0.0,"gradient_error_l2_squared":0.0}
                for k in ("both_zero","same_nonzero_sign","opposite_nonzero_sign","zero_vs_nonzero")}
    rows=[]
    for name in initial:
        left,right=grad_hip[name].double(),grad_ref[name].double()
        ge=(left-right).square()
        de=(hip["final_adapter"][name].double()-ref["final_adapter"][name].double()).square()
        a,b=left.sign(),right.sign()
        masks={"both_zero":(a==0)&(b==0),"same_nonzero_sign":(a==b)&(a!=0),
               "opposite_nonzero_sign":a*b==-1,"zero_vs_nonzero":(a==0)^(b==0)}
        assert sum(int(m.sum()) for m in masks.values())==left.numel()
        for key,mask in masks.items():
            categories[key]["elements"]+=int(mask.sum())
            categories[key]["update_error_l2_squared"]+=float(de[mask].sum())
            categories[key]["gradient_error_l2_squared"]+=float(ge[mask].sum())
        rows.append({"parameter":name,"gradient_error_l2_squared":float(ge.sum()),
                     "update_error_l2_squared":float(de.sum()),
                     "sign_flip_elements":int(masks["opposite_nonzero_sign"].sum())})
    original=delta_metrics(hip["final_adapter"],ref["final_adapter"],initial)
    assert math.isclose(original["relative_l2"],0.2582321163881623,rel_tol=1e-10)
    assert math.isclose(sum(x["update_error_l2_squared"] for x in categories.values()),
                        original["error_l2_squared"],rel_tol=1e-10)
    loss_derivative={}
    for name,payload in (("a100",ref),("hip",hip)):
        pred=payload["prediction"].clone().requires_grad_(True)
        (pred.float()-payload["inputs"]["target"].float()).square().mean().backward()
        loss_derivative[name]=pred.grad
    output=dict(step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],gpu_initialized=torch.cuda.is_initialized(),
                input_sha256=list(PINS.values()),original_cross_update=original,
                cpu_a100_vs_actual=delta_metrics(cpu_ref,ref["final_adapter"],initial),
                cpu_hip_vs_actual=delta_metrics(cpu_hip,hip["final_adapter"],initial),
                cpu_cross_update=delta_metrics(cpu_hip,cpu_ref,initial),
                cpu_a100_optimizer_vs_actual=compare_state(state_ref,ref["final_optimizer"]),
                cpu_hip_optimizer_vs_actual=compare_state(state_hip,hip["final_optimizer"]),
                clipping_changed_gradients={name:any(not torch.equal(x["gradients_before_clip"][k],x["gradients_after_clip"][k]) for k in initial)
                                            for name,x in (("a100",ref),("hip",hip))},
                loss_input_gradient_relative_l2=relative(loss_derivative["hip"],loss_derivative["a100"]),
                sign_categories=categories,parameter_contributions=rows)
    print(json.dumps(output,allow_nan=False))

if __name__=="__main__":
    main()
