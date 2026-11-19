"""Compare Qwen SDPA VJPs on one physical DCU with fixed synthetic inputs."""
import contextlib
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations import sdpa_attention
from assigned_kfd import assigned_kfd

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "PLAN.json").read_bytes())


def require(condition, name):
    if not condition:
        raise ValueError(name)


def digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def device_identity():
    require(torch.cuda.device_count() == 1, "ONE_ASSIGNED_DCU")
    hip = ctypes.CDLL("libamdhip64.so")
    text = ctypes.create_string_buffer(32)
    require(hip.hipDeviceGetPCIBusId(text,len(text),0)==0,"RUNTIME_ASSIGNED_PCI")
    pci = text.value.decode().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",pci):
        pci = "0000:" + pci
    return assigned_kfd(pci)


def fresh_inputs(token_count, heads, kv_heads, width):
    gen = torch.Generator(device="cpu").manual_seed(3218283)
    def direct(h):
        return torch.randn((1,token_count,h,width),generator=gen).to(device="cuda",dtype=torch.bfloat16).transpose(1,2)
    return direct(heads), direct(kv_heads), direct(kv_heads), direct(heads).transpose(1,2).contiguous()


def one(base, mask, backend, heads, kv_heads):
    inputs = [x.detach().requires_grad_(True) for x in base[:3]]
    scope = (contextlib.nullcontext() if backend=="dispatch_default"
             else sdpa_kernel(SDPBackend.MATH))
    module = SimpleNamespace(num_key_value_groups=heads//kv_heads,is_causal=True)
    with scope:
        output, weights = sdpa_attention.sdpa_attention_forward(module,*inputs,mask,dropout=0.0,
                                  scaling=128**-0.5,is_causal=mask is None)
        require(weights is None and output.shape==base[3].shape,"SDPA_OUTPUT_DESCRIPTOR")
        grads = torch.autograd.grad(output,inputs,base[3])
    return (output.detach(), *grads)


def metrics(now, ref):
    result = []
    for a,b in zip(now,ref,strict=True):
        af,bf=a.float(),b.float()
        difference=af-bf
        result.append(dict(finite=bool(torch.isfinite(af).all()),
                           relative_L2=float(difference.norm()/bf.norm().clamp_min(1e-8)),
                           max_abs=float(difference.abs().max())))
    return result


def compare(base, mask, backend, h, kv, count):
    first = None
    rows = []
    for _ in range(count):
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = one(base,mask,backend,h,kv)
        torch.cuda.synchronize()
        wall = time.perf_counter()-started
        if first is None:
            first = values
        rows.append(dict(digests=[digest(x) for x in values],wall_seconds=wall,vs_first=metrics(values,first)))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        one(base,mask,backend,h,kv)
        torch.cuda.synchronize()
    route = sorted({evt.key for evt in prof.key_averages() if "scaled_dot_product" in evt.key or "flash_attention" in evt.key})
    return first, dict(calls=rows,profile_operator_keys=route,
                       bitwise_stable=all(row["digests"]==rows[0]["digests"] for row in rows))


def main():
    require(socket.gethostname().split(".")[0]==PLAN["node"] and os.environ["SLURM_JOB_ID"]==PLAN["parent"]
            and os.environ["SLURM_STEP_ID"].isdecimal(),"EXACT_SLURM_DEVICE_SCOPE")
    require(str(torch.__version__)=="2.10.0" and torch.version.hip=="6.3.26113","ORIGINAL_TORCH_AND_HIP")
    torch.set_num_threads(2)
    path=Path(inspect.getsourcefile(sdpa_attention)).resolve()
    raw=path.read_bytes()
    require(len(raw)==4851 and hashlib.sha256(raw).hexdigest()==PLAN["provider_sha"],"ORIGINAL_TRANSFORMERS_SDPA_SOURCE")
    device = device_identity()
    shape=PLAN["geometry"]; tokens=shape["tokens"];heads=shape["query_heads"];kv=shape["kv_heads"];dim=shape["head_dim"]
    base=fresh_inputs(tokens,heads,kv,dim)
    result = dict(request=PLAN["request"],step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  torch=str(torch.__version__),hip=torch.version.hip,device=device,
                  provider_sha=PLAN["provider_sha"],inputs=[dict(shape=list(x.shape),stride=list(x.stride()),sha=digest(x)) for x in base],
                  seed=3218283,model_loaded=False,actual_original_activations=False,optimizer_updates=0,
                  gradient_order=["output","dq","dk","dv"],cases={})
    masks=[("is_causal",None),("explicit_causal_boolean",torch.ones((tokens,tokens),device="cuda",dtype=torch.bool).tril().view(1,1,tokens,tokens))]
    for name,mask in masks:
        default,drow=compare(base,mask,"dispatch_default",heads,kv,PLAN["repeat"])
        math,mrow=compare(base,mask,"math_only",heads,kv,PLAN["repeat"])
        compare_result=metrics(default,math)
        require(all(x["finite"] for x in compare_result),"NONFINITE_RESULT")
        result["cases"][name]=dict(default=drow,math=mrow,default_vs_math=compare_result)
    result["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    with (ROOT/"component-result.json").open("x",encoding="utf-8") as out:
        json.dump(result,out,allow_nan=False,sort_keys=True)
    print(json.dumps(dict(request=PLAN["request"],step=result["step"],cases=list(result["cases"]),worker_completed=True)))


if __name__=="__main__":
    main()
