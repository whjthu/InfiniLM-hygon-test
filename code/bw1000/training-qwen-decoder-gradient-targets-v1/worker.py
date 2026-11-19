"""Compare native direct/checkpoint derivatives on two real isolated decoders."""
import ctypes
import hashlib
import importlib.metadata as modmeta
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from assigned_kfd import assigned_kfd
from weights import ROOT, MODEL, CHECKPOINT, metadata, load_capture, load_layer, need


D=Path(__file__).resolve().parent
SCHEDULE=(
    ('direct','input'),('direct','all'),('direct','backward'),
    ('checkpoint','input'),('checkpoint','all'),('checkpoint','backward'),
)
ORDER=SCHEDULE+tuple(reversed(SCHEDULE))


def observe(value):
    tensor=value.detach().cpu()
    need(bool(torch.isfinite(tensor).all()),'Nonfinite gradient or decoder output')
    raw=tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def diff(left,right):
    need((left.shape,left.dtype)==(right.shape,right.dtype),'Compared tensor metadata differs')
    a,b=left.double(),right.double()
    delta=a-b
    num=float(torch.linalg.vector_norm(delta))
    den=float(torch.linalg.vector_norm(a))
    need(math.isfinite(num) and math.isfinite(den),'Comparison norm is nonfinite')
    return dict(exact=torch.equal(left,right), max_abs=float(delta.abs().max()),
                relative_l2=num/den if den else None, changed=int(torch.count_nonzero(delta)))


def device_binding():
    need(torch.cuda.device_count()==1,'Expected one runtime-visible accelerator')
    hip=ctypes.CDLL('libamdhip64.so')
    fn=hip.hipDeviceGetPCIBusId
    fn.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_int)
    fn.restype=ctypes.c_int
    buffer=ctypes.create_string_buffer(32)
    need(fn(buffer,len(buffer),0)==0,'Assigned HIP device has no PCI mapping')
    pci=buffer.value.decode().lower()
    if pci.count(':')==1:pci='0000:'+pci
    return assigned_kfd(pci)


def trial(layer,params,capture,route,targets,device):
    rng_cpu,rng_gpu=capture['cpu_rng'],capture['gpu_rng']
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_gpu,device)
    args=capture['args']
    inp=args[0].detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    leaves=tuple(params.values())
    for p in leaves:
        p.grad=None
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda',
            enabled=capture['autocast_enabled'],dtype=capture['autocast_dtype']):
        if route=='checkpoint':
            out=layer(inp,*args[1:],**capture['kwargs'])
        else:
            out=torch.nn.Module.__call__(layer,inp,*args[1:],**capture['kwargs'])
        if targets=='input':
            dx,=torch.autograd.grad(out,inp,capture['grad_out'])
            weights=()
        elif targets=='all':
            dx,*weights=torch.autograd.grad(out,(inp,*leaves),capture['grad_out'])
        else:
            torch.autograd.backward(out,capture['grad_out'])
            dx,weights=inp.grad,[p.grad for p in leaves]
    need(torch.is_tensor(dx) and dx.dtype==torch.bfloat16 and len(weights)==(0 if targets=='input' else 14)
         and all(torch.is_tensor(w) and w.dtype==torch.float32 for w in weights),'Unexpected native gradient scope')
    need(targets=='backward' or all(p.grad is None for p in leaves),'autograd.grad accumulated a parameter gradient')
    torch.cuda.synchronize(device)
    left=out.detach().cpu()
    right=dx.detach().cpu()
    w={name:observe(x) for name,x in zip(params,weights)}
    for p in leaves:
        p.grad=None
    return left,right,w


def execute(plan,text,rank,device,deadline):
    spec=plan['cases'][str(rank)]
    layer,params=load_layer(plan,text,spec['layer'],device)
    capture=load_capture(plan,rank,device)
    hashes_before={name:observe(p)['SHA256'] for name,p in params.items()}
    groups={}
    runs=[]
    for route,targets in ORDER:
        need(time.time() < deadline,'Actual component worker exhausted its shared epoch')
        out,dx,w=trial(layer,params,capture,route,targets,device)
        out_info=observe(out)
        dx_info=observe(dx)
        need(out_info['SHA256']==spec['previous_output_sha']
             and out_info['SHA256']==observe(capture['output'])['SHA256'],'Rehydrated decoder failed original forward reproduction')
        if targets=='input':
            need(dx_info['SHA256']==spec['previous_dx_sha'],'Rehydrated input-only VJP differs from original local VJP')
        key=(route,targets)
        repeat=None
        if key in groups:
            first,first_weights=groups[key]
            repeat=dict(input=diff(first,dx),parameter_gradients_exact=first_weights==w)
        else:
            groups[key]=(dx,w)
        runs.append(dict(route=route,targets=targets,output=out_info,dx=dx_info,parameter_gradients=w,repeat=repeat,
                         vs_full_training_dx=diff(capture['grad_in'],dx)))
    need(hashes_before=={name:observe(p)['SHA256'] for name,p in params.items()},'LoRA parameters changed during diagnostic')
    compare={}
    for route in ('direct','checkpoint'):
        x,xm=groups[(route,'input')]
        a,am=groups[(route,'all')]
        b,bm=groups[(route,'backward')]
        compare[route]=dict(input_vs_all=diff(x,a),all_vs_backward=diff(a,b),input_vs_backward=diff(x,b),
                            parameter_hashes_all_vs_backward_equal=am==bm,len_parameter_gradients=len(am))
    compare['direct_vs_checkpoint']={t:diff(groups[('direct',t)][0],groups[('checkpoint',t)][0])
                                      for _,t in SCHEDULE[:3]}
    return dict(old_rank=rank,layer=spec['layer'],capture_bytes=spec['private_operand']['bytes'],
                inventory=dict(frozen=11,trainable=len(params)),comparisons=compare,runs=runs)


def main():
    context=json.loads((D/'dispatch-context.json').read_bytes())
    plan=json.loads((D/'PLAN.json').read_bytes())
    need(context['request']==plan['request'] and socket.gethostname().split('.')[0]==plan['node']
         and os.environ['SLURM_JOB_ID']==plan['parent'] and os.environ['SLURM_STEP_ID'].isdigit(),
         'Unexpected Slurm lease')
    runtime={p:modmeta.version(p) for p in ('transformers','safetensors','peft')}
    need(torch.version.hip=='6.3.26113' and str(torch.__version__)=='2.10.0' and runtime['transformers']=='5.12.1'
         and runtime['peft']=='0.19.1','Original PyTorch, HIP or model provider differs')
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    pci=device_binding()
    timing=time.monotonic()
    text=metadata(plan)
    results=[execute(plan,text,rank,device,context['deadline']-25) for rank in (1,4)]
    torch.cuda.synchronize(device)
    data=dict(request=plan['request'],state='ACTUAL_REHYDRATED_DECODER_GRADIENT_TARGETS_COMPLETE',
              step=plan['parent']+'.'+os.environ['SLURM_STEP_ID'],node=plan['node'],actual_gpu=1,
              assigned_KFD=pci,runtime={**runtime,'torch':str(torch.__version__),'hip':torch.version.hip},
              actual_calls=sum(len(x['runs']) for x in results),results=results,optimizer_updates=0,
              timing_seconds=time.monotonic()-timing,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    with (D/'result.json').open('x',encoding='utf-8') as f:json.dump(data,f,allow_nan=False,indent=2)
    print(json.dumps(dict(state=data['state'],actual_calls=data['actual_calls'],step=data['step'])),flush=True)


if __name__=='__main__':
    main()
