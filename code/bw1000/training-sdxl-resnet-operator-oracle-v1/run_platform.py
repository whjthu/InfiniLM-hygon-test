"""Replay two real SDXL operators from the verified native prefix without loading the network."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import importlib.metadata

import torch
import torch.nn.functional as F
from safetensors import safe_open


FIRST='down_blocks.0.resnets.0'
PREFIXES=('conv_in','time_embedding','add_embedding',FIRST+'.norm1',FIRST+'.time_emb_proj')


def pin(tensor):
    v=tensor.detach().cpu().contiguous()
    return dict(shape=list(v.shape),dtype=str(v.dtype),
                SHA256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def metrics(native, reference):
    val=native.detach().cpu()
    wanted=reference.to(val.dtype)
    felt=val.double()-reference
    rounded=val.double()-wanted.double()
    assert all(torch.isfinite(x).all() for x in (val,reference))
    return dict(elements=val.numel(),exact_correctly_rounded=int((val==wanted).count_nonzero()),
                relative_L2_to_FP64=float(felt.norm()/reference.norm()),
                relative_L2_to_correct_round=float(rounded.norm()/wanted.double().norm()),
                max_abs_to_FP64=float(felt.abs().max()),max_abs_to_correct_round=float(rounded.abs().max()),
                sign_differences_to_correct_round=int((val.sign()!=wanted.sign()).count_nonzero()),
                correctly_rounded_pin=pin(wanted))


def bind_gpu(root,platform,claim):
    identity=dict(host=socket.gethostname().split('.')[0],pid=os.getpid())
    assert not torch.cuda.is_initialized() and torch.cuda.device_count()==1
    if platform=='hip':
        from hygon_sdxl_assigned_kfd import assigned_kfd
        from snapshot import drm_inventory
        gate=json.loads((root/'lease-before.json').read_bytes())
        step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        assert gate['step']==step and gate['node']==identity['host']==claim['hygon']['node']
        assert gate['kfd_holder_count']==0 and gate['pci']==drm_inventory()
        assert 0<=time.monotonic_ns()-gate['monotonic_ns']<45_000_000_000
        amd=ctypes.CDLL('libamdhip64.so')
        buf=ctypes.create_string_buffer(32)
        assert amd.hipDeviceGetPCIBusId(buf,len(buf),0)==0
        pci=buf.value.decode().lower()
        if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):pci='0000:'+pci
        assert pci in gate['pci']
        identity.update(step=step,pci=pci,kfd=assigned_kfd(pci))
    else:
        assert os.environ['CUDA_VISIBLE_DEVICES']==claim['a100']['uuid']
        assert torch.cuda.get_device_name(0)=='NVIDIA A100-SXM4-40GB'
    return identity


def main(platform, reference, _policy, weights, _old):
    root=Path(__file__).resolve().parent
    claim=json.loads((root/'CLAIM.json').read_bytes())
    intent=json.loads((root/'intent.json').read_bytes())
    assert intent['request']==claim['request'] and 0<=time.time()-intent['epoch']<600
    dep=json.loads((root/(platform+'-deps.json')).read_bytes())
    assert {k:importlib.metadata.version(k) for k in dep['versions']}==dep['versions']
    identity=bind_gpu(root,platform,claim)
    torch.set_num_threads(4)
    assets=Path(weights)/'unet'
    config=(assets/'config.json').read_bytes()
    assert hashlib.sha256(config).hexdigest()==claim['model_config']
    ref_path=Path(reference)
    assert ref_path.stat().st_size==74257307 and sha(ref_path)==claim['reference']
    ref=torch.load(ref_path,map_location='cpu',weights_only=True)
    from diffusers import UNet2DConditionModel
    with torch.device('meta'):
        model=UNet2DConditionModel.from_config(json.loads(config))
    assert all(p.is_meta for p in model.parameters()) and model.class_embedding is None
    device=torch.device('cuda',0)
    archive=assets/'diffusion_pytorch_model.safetensors'
    assert archive.stat().st_size==10270077736
    runtime_pins={}
    archive_pins={}
    elements_loaded=0
    with safe_open(str(archive),framework='pt',device='cpu') as fp:
        for prefix in PREFIXES:
            module=model.get_submodule(prefix)
            values={}
            for n, schema in module.state_dict().items():
                name=prefix+'.'+n
                tensor=fp.get_tensor(name)
                assert tensor.shape==schema.shape and tensor.dtype==torch.float32
                expected=claim['verified_weight_ranges'].get(name)
                archive_pins[name]=pin(tensor)
                if expected:
                    assert archive_pins[name]['SHA256']==expected['sha256']
                values[n]=tensor
                elements_loaded+=tensor.numel()
            missing, unexpected=module.load_state_dict(values,strict=True,assign=True)
            assert not missing and not unexpected
            module.requires_grad_(False).to(device=device,dtype=torch.bfloat16)
            for n,v in module.state_dict().items():runtime_pins[prefix+'.'+n]=pin(v)
    assert elements_loaded<20_000_000
    unit=model.get_submodule(FIRST)
    assert unit.norm1.num_groups>0 and unit.time_embedding_norm=='default' and not unit.skip_time_act
    assert unit.conv_shortcut is None and unit.output_scale_factor==1.0
    recipe=dict(bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    assert recipe==claim[platform+'_policy']
    inputs=ref['inputs']
    noisy=inputs['noisy_model_input'].to(device)
    timesteps=inputs['timesteps'].to(device)
    prompt=inputs['prompt_embeds'].to(device)
    extra={k:v.to(device) for k,v in inputs['unet_added_conditions'].items()}
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        temb=model.time_embedding(model.get_time_embed(sample=noisy,timestep=timesteps),None)
        augmentation=model.get_aug_embed(emb=temb,encoder_hidden_states=prompt,added_cond_kwargs=extra)
        temb=temb+augmentation if augmentation is not None else temb
        if model.time_embed_act is not None:
            temb=model.time_embed_act(temb)
        hidden=model.conv_in(noisy)
        gn=unit.norm1(hidden)
        act=unit.nonlinearity(temb)
        linear=unit.time_emb_proj(act)
    truth=dict(hidden=pin(hidden),temb=pin(temb),linear_input=pin(act),norm1=pin(gn),linear=pin(linear))
    assert truth==claim['previous_observation_pins'][platform], 'ORIGINAL_NATIVE_PREFIX_MISMATCH'
    x=hidden.detach().cpu().double()
    gamma=unit.norm1.weight.detach().cpu().double().reshape(1,-1,1,1)
    beta=unit.norm1.bias.detach().cpu().double().reshape(1,-1,1,1)
    g=x.reshape(x.shape[0],unit.norm1.num_groups,-1)
    centered=g-g.mean(-1,keepdim=True)
    expected_norm=(centered*(centered.square().mean(-1,keepdim=True)+unit.norm1.eps).rsqrt()).reshape_as(x)*gamma+beta
    z=act.detach().cpu().double()
    w=unit.time_emb_proj.weight.detach().cpu().double()
    b=unit.time_emb_proj.bias.detach().cpu().double()
    expected_linear=z@w.T+b
    float32_norm=F.group_norm(x.float(),unit.norm1.num_groups,gamma.reshape(-1).float(),
                              beta.reshape(-1).float(),unit.norm1.eps).double()
    float32_linear=F.linear(z.float(),w.float(),b.float()).double()
    next_input_observed=F.silu(gn.detach().cpu()).to(torch.bfloat16)
    next_input_ref=F.silu(expected_norm.float()).to(torch.bfloat16)
    postact_round_differences=int((next_input_observed!=next_input_ref).count_nonzero())
    result=dict(platform=platform,identity=identity,operator_pins=truth,
         runtime_parameter_pins=runtime_pins,checkpoint_parameter_pins=archive_pins,
         exact_previous_native_prefix=True,full_UNet_loads=0,optimizer_updates=0,
         raw_tensor_transfer_bytes=0,cpu_oracle=dict(norm1=metrics(gn,expected_norm),
         time_emb_proj=metrics(linear,expected_linear)),
         cpu_fp32_native_reference=dict(norm1=metrics(gn,float32_norm),time_emb_proj=metrics(linear,float32_linear)),
         norm_post_silu_bf16_different_elements=postact_round_differences,
         operand_stride=dict(norm1=list(hidden.stride()),linear=list(act.stride())),groupnorm=dict(groups=unit.norm1.num_groups,eps=unit.norm1.eps),
         source_runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,cuda=torch.version.cuda),
         materialized_parameter_elements=elements_loaded)
    with (root/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main(*sys.argv[1:])
