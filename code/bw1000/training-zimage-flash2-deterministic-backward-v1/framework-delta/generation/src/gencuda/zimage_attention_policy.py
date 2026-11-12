"""Explicit Flash2 deterministic backward policy;default execution is unchanged."""
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path

POLICY_ID='flash2-deterministic-backward-v1'
FLASH_VERSION='2.7.4.post1'
FLASH_PROVIDER_SHA256='d8eaf989a5401fedd23d11a336a7f38e67596d0a4d5a279df14cbe2d7de2ed75'
NATIVE_PROVIDER_SHA256='1b55a91812b4e82b6b0ba9c7f46036231acca3234a0b6a105ed210954508e2e4'


def policy_for_config(config):
    if 'attention_backward_policy' not in config:return None
    if type(config['attention_backward_policy']) is not str or config['attention_backward_policy']!=POLICY_ID:
        raise ValueError('unsupported explicit attention backward policy')
    return dict(schema=POLICY_ID,provider='flash_attn.flash_attn_func',version=FLASH_VERSION,
        provider_SHA256=FLASH_PROVIDER_SHA256,deterministic_backward=True,grad_enabled_only=True)


def _resolve_provider():
    import importlib,torch,flash_attn
    import flash_attn.flash_attn_interface as interface
    native=importlib.import_module('diffsynth.core.attention.attention')
    original=flash_attn.flash_attn_func
    if flash_attn.__version__!=FLASH_VERSION or original is not interface.flash_attn_func:
        raise ValueError('actual Flash2 provider or version differs')
    provider=Path(inspect.getsourcefile(original));np=Path(native.__file__)
    if provider.is_symlink() or np.is_symlink() or hashlib.sha256(provider.read_bytes()).hexdigest()!=FLASH_PROVIDER_SHA256 or hashlib.sha256(np.read_bytes()).hexdigest()!=NATIVE_PROVIDER_SHA256:
        raise ValueError('actual Flash2/native provider bytes differ')
    if native.ATTENTION_IMPLEMENTATION!='flash_attention_2':raise ValueError('policy requires actual native Flash2 selection')
    sig=inspect.signature(original)
    if sig.parameters['deterministic'].default is not False:raise ValueError('Flash2 deterministic API default differs')
    return flash_attn,torch,original,sig


@contextmanager
def apply_training_attention_policy(policy,*,check=lambda:None):
    if policy is None:
        yield None;return
    if policy!=policy_for_config(dict(attention_backward_policy=POLICY_ID)):
        raise ValueError('attention backward policy identity differs')
    module,torch,original,signature=_resolve_provider();counts=Counter();shapes={}
    ledger=dict(policy=policy,forced_calls=0,unchanged_calls=0,shape_counts=[])
    @functools.wraps(original)
    def call(*args,**kwargs):
        check();bound=signature.bind_partial(*args,**kwargs)
        if 'deterministic' in bound.arguments:raise ValueError('caller already supplies a deterministic override')
        tensors=[bound.arguments[n] for n in ('q','k','v')]
        active=bool(torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
        if active:
            if any(t.device.type!='cuda' for t in tensors):raise ValueError('Flash2 policy requires actual CUDA tensors')
            kwargs=dict(kwargs,deterministic=True);ledger['forced_calls']+=1
            row=dict(q_shape=list(tensors[0].shape),q_stride=list(tensors[0].stride()),dtype=str(tensors[0].dtype))
            key=json.dumps(row,sort_keys=True)
            if key not in shapes and len(shapes)>=512:raise ValueError('attention policy shape bound exceeded')
            shapes[key]=row;counts[key]+=1;ledger['shape_counts']=[dict(shapes[k],calls=counts[k]) for k in sorted(counts)]
        else:ledger['unchanged_calls']+=1
        return original(*args,**kwargs)
    module.flash_attn_func=call
    try:yield ledger
    finally:
        if module.flash_attn_func is not call:raise RuntimeError('Flash2 policy wrapper ownership changed')
        module.flash_attn_func=original
