"""Compare interventions at the actual SDXL GroupNorm-to-SiLU-to-convolution boundary."""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

import torch
from safetensors import safe_open


FIRST='down_blocks.0.resnets.0'
MODULES=('conv_in',FIRST+'.norm1',FIRST+'.conv1')


def pin(tensor):
    v=tensor.detach().cpu().contiguous()
    return dict(shape=list(v.shape),dtype=str(v.dtype),
                SHA256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def difference(x,y):
    a,b=(t.detach().cpu() for t in (x,y))
    d=a.double()-b.double()
    denominator=b.double().norm().item()
    assert denominator>0 and torch.isfinite(d).all()
    return dict(elements=d.numel(),changed=int((a!=b).count_nonzero()),
                max_abs=float(d.abs().max()),relative_l2=float(d.norm()/denominator))


def bind_gpu(root,platform,claim):
    info=dict(host=socket.gethostname().split('.')[0],pid=os.getpid())
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==1
    if platform=='hip':
        from hygon_sdxl_assigned_kfd import assigned_kfd
        from snapshot import drm_inventory
        gate=json.loads((root/'lease-before.json').read_bytes())
        step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        assert gate['step']==step and gate['node']==info['host']==claim['hygon']['node']
        assert gate['kfd_holder_count']==0 and gate['pci']==drm_inventory()
        assert 0<=time.monotonic_ns()-gate['monotonic_ns']<45_000_000_000
        amp=ctypes.CDLL('libamdhip64.so')
        buffer=ctypes.create_string_buffer(32)
        assert amp.hipDeviceGetPCIBusId(buffer,len(buffer),0)==0
        pci=buffer.value.decode().lower()
        if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):
            pci='0000:'+pci
        assert pci in gate['pci']
        info.update(step=step,pci=pci,kfd=assigned_kfd(pci))
    else:
        assert os.environ['CUDA_VISIBLE_DEVICES']==claim['a100']['uuid']
        assert torch.cuda.get_device_name(0)=='NVIDIA A100-SXM4-40GB'
    return info


def original_prefix(weights, claim):
    from diffusers import UNet2DConditionModel
    config=(weights/'config.json').read_bytes()
    assert hashlib.sha256(config).hexdigest()==claim['model_config']
    with torch.device('meta'):
        model=UNet2DConditionModel.from_config(json.loads(config))
    assert all(p.is_meta for p in model.parameters())
    file=weights/'diffusion_pytorch_model.safetensors'
    assert file.stat().st_size==10270077736
    archive={}
    affine={}
    with safe_open(str(file),framework='pt',device='cpu') as src:
        for prefix in MODULES:
            layer=model.get_submodule(prefix)
            state={}
            for n,schema in layer.state_dict().items():
                name=prefix+'.'+n
                x=src.get_tensor(name)
                assert x.shape==schema.shape and x.dtype==torch.float32
                archive[name]=pin(x)
                old=claim['verified_weight_ranges'].get(name)
                if old:
                    assert archive[name]['SHA256']==old['sha256']
                state[n]=x
            assert len(state)==2
            missing,unexpected=layer.load_state_dict(state,strict=True,assign=True)
            assert not missing and not unexpected
            layer.requires_grad_(False).to(device='cuda',dtype=torch.bfloat16)
            for n,t in layer.state_dict().items():
                affine[prefix+'.'+n]=pin(t)
    assert set(affine)==set(archive) and len(archive)==6
    unit=model.get_submodule(FIRST)
    assert unit.output_scale_factor==1.0 and unit.norm1.num_groups==32 and unit.norm1.eps==1e-5
    return model.conv_in,unit,archive,affine


def run(platform,reference,weights):
    root=Path(__file__).resolve().parent
    claim=json.loads((root/'CLAIM.json').read_bytes())
    ctx=json.loads((root/'intent.json').read_bytes())
    assert ctx['request']==claim['request'] and 0<=time.time()-ctx['epoch']<600
    dep=json.loads((root/(platform+'-deps.json')).read_bytes())
    assert {k:importlib.metadata.version(k) for k in dep['versions']}==dep['versions']
    assert bool(torch.version.hip)==(platform=='hip')
    info=bind_gpu(root,platform,claim)
    torch.set_num_threads(4)
    recipe=dict(bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    assert recipe==claim[platform+'_policy']
    ref=Path(reference)
    assert ref.stat().st_size==74257307 and sha(ref)==claim['reference']
    src=torch.load(ref,map_location='cpu',weights_only=True)
    noisy=src['inputs']['noisy_model_input'].to('cuda')
    conv_in,unit,checkpoint,runtime=original_prefix(Path(weights)/'unet',claim)
    groups,eps=unit.norm1.num_groups,unit.norm1.eps
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=conv_in(noisy)
        native_norm=unit.norm1(hidden)

    x=hidden.detach().cpu().double()
    grouped=x.reshape(x.shape[0],groups,-1)
    zero_center=grouped-grouped.mean(-1,keepdim=True)
    affine=tuple(t.detach().cpu().double().reshape(1,-1,1,1) for t in (unit.norm1.weight,unit.norm1.bias))
    reference_norm=(zero_center*(zero_center.square().mean(-1,keepdim=True)+eps).rsqrt()).reshape_as(x)*affine[0]+affine[1]
    reference_norm=reference_norm.float().to('cuda')

    with torch.no_grad(), torch.autocast('cuda',enabled=False):
        f32=hidden.float().reshape(hidden.shape[0],groups,-1)
        variance,mean=torch.var_mean(f32,dim=-1,correction=0,keepdim=True)
        norm=((f32-mean)*(variance+eps).rsqrt()).reshape_as(hidden)
        gamma,beta=(t.float().reshape(1,-1,1,1) for t in (unit.norm1.weight,unit.norm1.bias))
        explicit_norm=torch.addcmul(beta,norm,gamma)
    arms={}
    for name,boundary in [('native',native_norm),('common_cpu_fp64_rounded',reference_norm),
                          ('gpu_var_mean_addcmul',explicit_norm)]:
        assert boundary.dtype==torch.float32
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            silu=unit.nonlinearity(boundary)
            out=unit.conv1(silu)
        assert silu.dtype==torch.float32 and out.dtype==torch.bfloat16
        arms[name]=dict(norm=boundary,silu_fp32=silu,silu_bf16=silu.to(torch.bfloat16),conv1=out)
    observed=dict(hidden=pin(hidden),norm=pin(arms['native']['norm']),
                  silu_fp32=pin(arms['native']['silu_fp32']),conv1=pin(arms['native']['conv1']))
    assert observed==claim['previous_observation_pins'][platform], 'ORIGINAL_CONV1_PREFIX_MISMATCH'
    identities={k:{name:pin(t) for name,t in row.items()} for k,row in arms.items()}
    diffs={k:{name:difference(t,arms['native'][name]) for name,t in row.items()}
           for k,row in arms.items() if k!='native'}
    result=dict(platform=platform,identity=info,checkpoint_pins=checkpoint,runtime_weight_pins=runtime,
                original_prefix_exact=True,native_observed=observed,arms=identities,differences_from_native=diffs,
                norm_groups=groups,norm_eps=eps,raw_tensor_transfer_bytes=0,full_model_calls=0,optimizer_updates=0,
                module_parameters=sum(t.numel() for m in (conv_in,unit.norm1,unit.conv1) for t in m.parameters()),
                reference_scope='common_cpu_fp64_rounded is a diagnostic replacement only')
    with (root/'result.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    platform,reference,_policy,weights,_previous=sys.argv[1:]
    run(platform,reference,weights)
