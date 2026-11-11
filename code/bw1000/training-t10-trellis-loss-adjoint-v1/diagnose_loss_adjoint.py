"""Hold one TRELLIS forward fixed and substitute only the A100 loss gradient."""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"generation"/"src"))
import numpy as np
import torch
from gencuda.backends.trellis2_training import checked_memory, flow_prediction_and_loss, observe_sdpa
from gencuda.runtime import MEMORY_LIMIT_BYTES, inspect_runtime, validate_accelerator
from gencuda.trellis_training_contract import file_sha, require
from gencuda.trellis_training_resume_checkpoint import make_optimizer
from hygon_trellis_first_step import executed_source, inputs_and_reference, prepare_model, verify_shape_flow_source
from hygon_sdxl_compare_first_step import exact


HIP_OBSERVED = "b1acdf0b074c3c66639b566ab082544d7c8bfac63a487d05b0f4edc1a8bcd767"


def capture(named):
    values = {name: p.grad.detach().cpu().clone() for name,p in named.items()}
    require(all(torch.isfinite(g).all() for g in values.values()), "non-finite adapter gradient")
    return values


def delta_l2(a,b):
    assert list(a)==list(b)
    squared = sum(float((a[n].double()-x.double()).square().sum()) for n,x in b.items())
    signal = sum(float(x.double().square().sum()) for x in b.values())
    return (squared/signal)**0.5


def project_update(initial,state,gradients,config):
    named={name:torch.nn.Parameter(value.clone()) for name,value in initial.items()}
    opt=make_optimizer(list(named.values()),config)
    opt.load_state_dict(state)
    for name,p in named.items():
        p.grad=gradients[name].clone()
    opt.step()
    return {name:p.detach().double()-initial[name].double() for name,p in named.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("reference","weights","cache","source","source-inventory","original-hip","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    validate_accelerator(inspect_runtime(),"hip",1)
    require(file_sha(args.original_hip)==HIP_OBSERVED,"original HIP evidence differs")
    config,reference,_,_=inputs_and_reference(torch,args)
    original=torch.load(args.original_hip,map_location="cpu",weights_only=True)
    for key in ("initial_adapter","initial_optimizer","inputs","prediction_coords"):
        exact(torch,reference[key],original[key],key)
    source=verify_shape_flow_source(config,args.source_inventory)
    random.seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda",0)
    torch.cuda.set_device(device)
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0,MEMORY_LIMIT_BYTES/total),device)
    torch.cuda.reset_peak_memory_stats(device)
    model,sparse,named,optimizer,initial,state,_=prepare_model(torch,args,config,reference,device)
    values={name:t.to(device) for name,t in reference["inputs"].items()}
    a100_prediction=reference["prediction"].clone().requires_grad_(True)
    a100_loss=(a100_prediction.float()-reference["inputs"]["target"].float()).square().mean()
    a100_adjoint,=torch.autograd.grad(a100_loss,a100_prediction)
    require(a100_adjoint.dtype==torch.bfloat16 and torch.isfinite(a100_adjoint).all(),"invalid A100 loss gradient")
    optimizer.zero_grad(set_to_none=True)
    with observe_sdpa(torch) as sdpa:
        prediction,loss=flow_prediction_and_loss(torch,model,sparse,values["coords"],values["noised"],
            values["fraction"],values["condition"],values["target"])
        features=prediction.feats
        native_adjoint,=torch.autograd.grad(loss,features,retain_graph=True)
        loss.backward(retain_graph=True)
        control=capture(named)
        optimizer.zero_grad(set_to_none=True)
        features.backward(a100_adjoint.to(device))
        intervention=capture(named)
    torch.cuda.synchronize(device)
    require(sdpa["forward_and_recompute_calls"]==60,"unexpected number of native SDPA forward calls")
    require(all(torch.equal(p.detach().cpu(),initial[name]) for name,p in named.items()),"parameters changed during backwards")
    require(optimizer.state_dict()["state"]=={},"device optimizer state changed")
    modules=executed_source(config)
    require(verify_shape_flow_source(config,args.source_inventory)==source,"source changed during test")
    mem=checked_memory()
    ref_grads=reference["gradients_before_clip"]
    ref_update={name:x.double()-initial[name].double() for name,x in reference["final_adapter"].items()}
    hip_prediction=features.detach().cpu()
    measures={
        "control_gradient_relative_l2_to_a100":delta_l2(control,ref_grads),
        "intervention_gradient_relative_l2_to_a100":delta_l2(intervention,ref_grads),
        "control_gradient_relative_l2_to_original_hip":delta_l2(control,original["gradients_before_clip"]),
        "intervention_gradient_relative_l2_to_control":delta_l2(intervention,control),
        "control_cpu_projected_update_relative_l2_to_a100":delta_l2(project_update(initial,state,control,config),ref_update),
        "intervention_cpu_projected_update_relative_l2_to_a100":delta_l2(project_update(initial,state,intervention,config),ref_update),
        "native_loss_adjoint_relative_l2_to_a100":delta_l2({"grad":native_adjoint.cpu()},{"grad":a100_adjoint}),
        "prediction_relative_l2_to_original_hip":delta_l2({"prediction":hip_prediction},{"prediction":original["prediction"]}),
        "prediction_bitwise_to_original_hip":torch.equal(hip_prediction,original["prediction"]),
    }
    for name,gradients in (("control",control),("intervention",intervention)):
        result={k:float((gradients[k].double()-ref_grads[k].double()).square().sum()) for k in named}
        measures[name+"_gradient_error_l2_squared_by_parameter"]=result
        require(sum(float(x.double().square().sum()) for x in gradients.values())**0.5<config["max_grad_norm"],
                "clipping would change projected CPU update")
    result={"scope":"same device and forward; the loss gradient is the only changed input to backward",
            "gpu_backward_arms":2,"gpu_optimizer_updates":0,"weights_loaded":True,"sdpa_calls":sdpa,
            "measurements":measures,"source_binding":source,"executed_source":modules,
            "original_hip_SHA256":HIP_OBSERVED,"peak_reserved_bytes":mem["peak_reserved_bytes"]}
    args.output.mkdir()
    raw_path=args.output/"adjoint-gradients.pt"
    torch.save({"control":control,"a100_loss_gradient_intervention":intervention},raw_path)
    result["private_gradient_artifact_SHA256"]=file_sha(raw_path)
    with (args.output/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,allow_nan=False))

if __name__=="__main__":
    main()
