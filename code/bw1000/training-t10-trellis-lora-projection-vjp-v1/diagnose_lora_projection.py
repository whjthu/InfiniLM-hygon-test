"""Capture and replay the TRELLIS LoRA B weight gradient at four projections."""

import argparse
import json
from pathlib import Path
import random
import sys

tools_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(tools_dir))
sys.path.insert(0, str(tools_dir.parent / "generation" / "src"))
import numpy as np
import torch
from gencuda.backends.trellis2_training import checked_memory, flow_prediction_and_loss
from gencuda.runtime import MEMORY_LIMIT_BYTES, inspect_runtime, validate_accelerator
from gencuda.trellis_training_contract import file_sha, require
from hygon_trellis_first_step import executed_source, inputs_and_reference, prepare_model, verify_shape_flow_source
from hygon_sdxl_compare_first_step import exact
from lora_projection_reference import capture_lora, compare, measure, qualify_cpu_vjp
from gencuda.trellis_training_resume_checkpoint import make_optimizer


HIP_OBSERVED = "b1acdf0b074c3c66639b566ab082544d7c8bfac63a487d05b0f4edc1a8bcd767"


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
    qualify_cpu_vjp()
    optimizer.zero_grad(set_to_none=True)
    with capture_lora(model) as record:
        prediction, loss = flow_prediction_and_loss(torch,model,sparse,values["coords"],values["noised"],
            values["fraction"],values["condition"],values["target"])
        loss.backward()
    for name, actual in (("prediction",prediction.feats), ("loss",loss)):
        exact(torch,actual.detach().cpu(),original[name],name)
    control = {name:p.grad.detach().cpu() for name,p in named.items()}
    exact(torch,control,original["gradients_before_clip"],"full_model_gradients")
    require(all(torch.equal(p.detach().cpu(),initial[name]) for name,p in named.items()),"parameters changed")
    exact(torch,optimizer.state_dict(),state,"optimizer_state")
    a100_grad = reference["gradients_before_clip"]
    result, replacement = compare(record,control,a100_grad,args.output,file_sha)
    ref_update = {n:value.double()-initial[n].double() for n,value in reference["final_adapter"].items()}
    ref_update_flat = torch.cat([value.flatten() for value in ref_update.values()])
    for label, gradient in (("native",control), ("four_cpu_f64_rounded",replacement)):
        parameters = {n:torch.nn.Parameter(value.clone()) for n,value in initial.items()}
        projected = make_optimizer(list(parameters.values()),config)
        projected.load_state_dict(state)
        for name, param in parameters.items():
            param.grad = gradient[name].clone()
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config["max_grad_norm"])
        require(torch.isfinite(norm) and 0<float(norm)<config["max_grad_norm"],"CPU projection clipping differs")
        projected.step()
        update = torch.cat([(param.detach().double()-initial[n].double()).flatten()
                            for n,param in parameters.items()])
        result[label+"_cpu_projected_update_vs_a100"] = measure(update,ref_update_flat)
    torch.cuda.synchronize(device)
    require(verify_shape_flow_source(config,args.source_inventory)==source,"source changed during test")
    result.update({"model_gradients_bitwise_to_original_hip":True,"prediction_bitwise_to_original_hip":True,
                   "gpu_optimizer_updates":0,"scope":"captured native LoRA B backward at four projections; exact-operand DCU BF16/FP32 and CPU FP64 VJP; CPU-only AdamW projection",
                   "peak_reserved_bytes":checked_memory()["peak_reserved_bytes"],
                   "source_binding":source,"executed_source":executed_source(config),"original_hip_SHA256":HIP_OBSERVED})
    with (args.output/"summary.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main()
