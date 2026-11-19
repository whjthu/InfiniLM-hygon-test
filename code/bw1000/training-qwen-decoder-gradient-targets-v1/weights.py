"""Rehydrate only the original two decoder layers and their trained LoRA factors."""
from functools import partial
import hashlib
import json
from pathlib import Path
import stat

import torch
from safetensors import safe_open
from peft import LoraConfig, inject_adapter_in_model
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
from torch.utils.checkpoint import checkpoint


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
MODEL = ROOT/'training-qwen-original-public-model-v1/model'
CHECKPOINT = ROOT/'training-qwen-math-sdpa-recovery-pair-v1/continuous/train/checkpoint-20'
CAPTURES = ROOT/'training-qwen-local-decoder-vjp-v3/train/decoder-operands'
SHARD = 'model-00003-of-00004.safetensors'
PROJECTIONS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
OPERANDS = {'args','kwargs','cpu_rng','gpu_rng','autocast_enabled','autocast_dtype',
            'recompute_input','output','grad_in','grad_out'}


def need(value, message):
    if not value:
        raise ValueError(message)


def verify_file(path,pin):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size==pin['bytes'],'Frozen file size or regular-file identity differs')
    dig=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            dig.update(block)
    after=path.lstat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
         dig.hexdigest()==pin['SHA256'],'Frozen file changed from its verified SHA256')
    return dict(bytes=after.st_size,SHA256=dig.hexdigest())


def metadata(plan):
    a=json.loads((CHECKPOINT/'adapter_config.json').read_bytes())
    expected=dict(r=16,lora_alpha=32,lora_dropout=.05,bias='none',use_rslora=False,use_dora=False,
                  fan_in_fan_out=False,rank_pattern={},alpha_pattern={},modules_to_save=[],lora_bias=False)
    need(all(a.get(k)==v for k,v in expected.items()),'Original adapter options differ')
    need(a['target_modules']==plan['adapter_target_modules'],'Original adapter target pattern differs')
    return json.loads((MODEL/'config.json').read_bytes())['text_config']


def load_layer(plan,text,layer_id,device):
    detail=plan['layers'][str(layer_id)]
    prefix=f'model.language_model.layers.{layer_id}.'
    index=json.loads((MODEL/'model.safetensors.index.json').read_bytes())['weight_map']
    expected={prefix+n:SHARD for n in detail['base']}
    actual={n:file for n,file in index.items() if n.startswith(prefix)}
    need(actual==expected and len(actual)==11,'Original full decoder tensor inventory differs')
    config=Qwen3VLTextConfig.from_dict(text)
    config._attn_implementation='sdpa'
    config.use_cache=False
    with torch.device('meta'):
        layer=Qwen3VLTextDecoderLayer(config,layer_id)
    need(set(layer.state_dict())==set(detail['base']),'Original decoder provider state names differ')
    base={}
    with safe_open(str(MODEL/SHARD),framework='pt',device='cpu') as f:
        for name,pin in detail['base'].items():
            x=f.get_tensor(prefix+name)
            need(pin==['BF16',list(x.shape)] and x.dtype==torch.bfloat16,'Original frozen decoder tensor differs')
            base[name]=x
    layer.load_state_dict(base,strict=True,assign=True)
    layer.requires_grad_(False)
    layer.to(device)
    del base
    cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=list(PROJECTIONS),bias='none',init_lora_weights=False)
    need(inject_adapter_in_model(cfg,layer,adapter_name='default') is layer,'PEFT modified standalone layer identity')
    trained={}
    with safe_open(str(CHECKPOINT/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        key_prefix='base_model.model.'+prefix
        actual={k.removeprefix(key_prefix):k for k in f.keys() if k.startswith(key_prefix)}
        need(set(actual)==set(detail['adapter']) and len(actual)==14,'Original adapter named tensors differ')
        for n,k in actual.items():
            x=f.get_tensor(k)
            need(detail['adapter'][n]==['F32',list(x.shape)] and x.dtype==torch.float32,'Original trained LoRA tensor dtype or shape differs')
            stem,_,name=n.rpartition('.')
            qualified=stem+'.default.'+name
            trained[qualified]=x.to(device)
    need(set(n for n,p in layer.named_parameters() if p.requires_grad)==set(trained),'Expected only the original 14 trainable LoRA parameters')
    restore=layer.load_state_dict(trained,strict=False,assign=True)
    need(not restore.unexpected_keys and len(restore.missing_keys)==11,'PEFT checkpoint assignment changed frozen boundary')
    params={n:p for n,p in layer.named_parameters() if p.requires_grad}
    need(set(params)==set(trained) and all(p.dtype==torch.float32 and p.device==device for p in params.values()),
         'Native FP32 LoRA parameter bindings differ')
    need(all(p.dtype==torch.bfloat16 for p in layer.parameters() if not p.requires_grad),'Frozen decoder precision differs')
    layer.train()
    layer.gradient_checkpointing=True
    layer._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)
    return layer,params


def copy_to(value,device):
    if isinstance(value,torch.Tensor):
        out=value.to(device)
        need(out.stride()==value.stride(),'Captured decoder operand layout changed')
        return out
    if type(value)==dict:
        return {k:copy_to(v,device) for k,v in value.items()}
    if type(value) in (list,tuple):
        return type(value)(copy_to(v,device) for v in value)
    need(value is None or type(value) in (bool,int,float,str,torch.dtype),'Unexpected captured operand type')
    return value


def load_capture(plan,rank,device):
    proof=plan['cases'][str(rank)]
    path=CAPTURES/f'rank-{rank:02d}.pt'
    verify_file(path,proof['private_operand'])
    capture=torch.load(path,weights_only=True,map_location='cpu')
    need(set(capture)==OPERANDS and capture['autocast_dtype']==torch.bfloat16,'Original captured decoder envelope differs')
    need(capture['cpu_rng'].dtype==capture['gpu_rng'].dtype==torch.uint8,'Original captured RNG state kind differs')
    hidden=capture['args'][0]
    need((list(hidden.shape),list(hidden.stride()),hidden.dtype)==([1,283,4096],[1159168,4096,1],torch.bfloat16),
         'Original captured decoder boundary geometry differs')
    for field in ('args','kwargs','grad_out'):
        capture[field]=copy_to(capture[field],device)
    return capture
