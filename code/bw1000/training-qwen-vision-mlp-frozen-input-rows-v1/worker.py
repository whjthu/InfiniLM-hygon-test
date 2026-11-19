"""Resolve the first Qwen vision MLP from an actual, fixed rank-one call."""
import json
import math
from pathlib import Path
import statistics
import time

import torch

from vision_original import CAPTURE, ROOT, check, device_identity, load_model, sha, tensor_note, vision


WIDTH=128
ARMS=(False,True,True,False,True,False,False,True)


def archived_mlp_input(model,pin,deadline):
    rank=pin["ranks"]["1"]
    path=CAPTURE/"rank-01"/"forward-01-input.pt"
    sha(path.parent/"receipt.json",rank["receipt"],deadline)
    sha(path,rank["files"]["forward-01-input.pt"],deadline)
    archive=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    check(archive["global_step"]==0 and archive["micro_ordinal"]==1 and not archive["args"],"ORIGINAL_MICROBATCH")
    values=archive["kwargs"]
    check(values.get("pixel_values_videos") is None,"IMAGE_ONLY_SCOPE")
    check(not any(values.get(k) is not None for k in ("bilinear_indices","bilinear_weights","cu_seqlens")),
          "ORIGINAL_VISION_HELPER_SCOPE")
    pixels=values["pixel_values"].to(device="cuda:0",dtype=model.dtype).requires_grad_()
    grid=values["image_grid_thw"].to(device="cuda:0")
    check(list(pixels.shape)==[640,1536] and list(grid.shape)==[1,3] and grid.dtype==torch.long,
          "ACTUAL_RANK_ONE_IMAGE_GEOMETRY")
    rng=archive["RNG_before_forward"]
    torch.set_rng_state(rng["torch_CPU"])
    torch.cuda.set_rng_state(rng["torch_device"],0)
    captured=[]
    mlp=model.blocks[0].mlp
    def observe(_m,args):
        check(len(args)==1,"ONE_MLP_OPERAND")
        captured.append((args[0],args[0]._version))
    hook=mlp.register_forward_pre_hook(observe)
    try:
        with torch.enable_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            result=model(pixels,grid_thw=grid,return_dict=True)
        check(result.pooler_output.requires_grad,"IMAGE_GRAPH_CONNECTED")
    finally:
        hook.remove()
    (value,version),=captured
    check(value._version==version and value.requires_grad and value.dtype==torch.bfloat16 and
          list(value.shape)==[640,1152],"ACTUAL_BLOCK_ZERO_MLP_OPERAND")
    return value.detach(),rng


def apply(mlp,x,split):
    if split:
        return torch.cat([mlp(chunk) for chunk in x.split(WIDTH)],dim=0)
    return mlp(x)


def ref(mlp,x):
    with torch.no_grad():
        fc1=torch.nn.functional.linear(x.cpu().float(),
              mlp.linear_fc1.weight.cpu().float(),mlp.linear_fc1.bias.cpu().float()).to(torch.bfloat16)
        activation=mlp.act_fn(fc1)
        fc2=torch.nn.functional.linear(activation.float(),mlp.linear_fc2.weight.cpu().float(),
                                      mlp.linear_fc2.bias.cpu().float()).to(torch.bfloat16)
    return {"fc1":fc1,"activation":activation,"fc2":fc2}


def delta(a,b):
    check(a.shape==b.shape and a.dtype==b.dtype,"DIFFERING_COMPONENT_LAYOUT")
    h=a.detach().cpu().double()
    r=b.detach().cpu().double()
    x=h-r
    n=float(r.square().sum().item())
    e=float(x.square().sum().item())
    return dict(max_abs=float(x.abs().max().item()),different=int((x!=0).sum().item()),
                relative_L2=math.sqrt(e/n) if n else None)


def observe(mlp,source,rng,reference,deadline):
    notes=[]
    base=tensor_note(source)
    for split in ARMS:
        check(time.time()<deadline,"ORIGINAL_EPOCH")
        torch.set_rng_state(rng["torch_CPU"])
        torch.cuda.set_rng_state(rng["torch_device"],0)
        stages={key:[] for key in ("fc1","activation","fc2")}
        x=source.detach().requires_grad_()
        def hook(name):
            def take(_m,_a,y):
                stages[name].append((y,y._version))
            return take
        modules=dict(fc1=mlp.linear_fc1,activation=mlp.act_fn,fc2=mlp.linear_fc2)
        handles=[m.register_forward_hook(hook(name)) for name,m in modules.items()]
        try:
            with torch.enable_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                y=apply(mlp,x,split)
        finally:
            for h in handles:h.remove()
        check(y.requires_grad and y.shape==x.shape and tensor_note(source)==base,"FIXED_GRAPH_CONNECTED_MLP_OPERAND")
        result={}
        count=5 if split else 1
        for key,values in stages.items():
            check(len(values)==count and all(t._version==v for t,v in values),"EXACT_NATIVE_STAGE_CALLS")
            parts=[t.detach().cpu() for t,_ in values]
            tensor=torch.cat(parts,dim=0) if split else parts[0]
            result[key]=tensor_note(tensor)|dict(vs_float32_cpu_then_BF16=delta(tensor,reference[key]))
        check(result["fc2"]["SHA256"]==tensor_note(y)["SHA256"],"EXACT_OUTPUT_CAPTURE")
        notes.append(dict(split=split,stage_calls=count,stages=result))
    return notes


def timing(mlp,source,deadline):
    out=[]
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    x=source.detach().requires_grad_()
    for split in (False,True,True,False):
        rates=[]
        with torch.enable_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            for i in range(17):
                check(time.time()<deadline,"ORIGINAL_EPOCH")
                events[0].record()
                output=apply(mlp,x,split)
                events[1].record()
                events[1].synchronize()
                if i>=5:rates.append(events[0].elapsed_time(events[1]))
                del output
        check(all(math.isfinite(v) and v>0 for v in rates),"NATIVE_MLP_COMPONENT_TIMER")
        out.append(dict(split=split,measured=12,warm=5,event_ms=rates,median_event_ms=statistics.median(rates)))
    return out


def main():
    intent=json.loads((ROOT/"intent.json").read_bytes())
    deadline=intent["epoch"]+600
    pin=json.loads((ROOT/"INPUTS.json").read_bytes())
    original=ROOT/"vision_original.py"
    manifest=json.loads((ROOT/"CLAIM.json").read_bytes())
    import hashlib
    check(hashlib.sha256(original.read_bytes()).hexdigest()==manifest["native_loader_SHA256"],"REAL_VISION_LOADER")
    check(hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest()==pin["provider_SHA256"],"REAL_VISION_PROVIDER")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    identity=device_identity()
    t0=time.monotonic()
    model=load_model(pin,deadline)
    operand,rng=archived_mlp_input(model,pin,deadline)
    mlp=model.blocks[0].mlp
    check(not any(p.requires_grad for p in mlp.parameters()) and
          tuple(mlp.linear_fc1.weight.shape)==(4304,1152) and tuple(mlp.linear_fc2.weight.shape)==(1152,4304),
          "EXACT_ORIGINAL_MLP_LAYOUT")
    before={name:t._version for name,t in mlp.named_parameters()}
    reference=ref(mlp,operand)
    rows=observe(mlp,operand,rng,reference,deadline)
    measurements=timing(mlp,operand,deadline)
    check(before=={name:t._version for name,t in mlp.named_parameters()},"ORIGINAL_MLP_WEIGHT_MUTATION")
    result=dict(passed=True,identity=identity,original_visual_image_rank=1,original_full_vision_forwards=1,
                actual_mlp_input=tensor_note(operand),row_chunk=WIDTH,records=rows,timing=measurements,
                math_reference="float32 CPU linear rounded to BF16, native GELU BF16, float32 CPU linear rounded to BF16",
                elapsed_seconds=time.monotonic()-t0,actual_backward=0,actual_optimizer=0,formal_training=False)
    with (ROOT/"result.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,allow_nan=False,separators=(",",":"))


if __name__=="__main__":
    main()
