"""Capture and restore the mutable state used by the fixed native forward."""
import hashlib
import json
import math
import random

import numpy as np
import torch


def describe(value, *, version=False):
    if isinstance(value, torch.Tensor):
        shape=list(value.shape)
        n=value.numel()*value.element_size()
        if n>768*2**20:
            raise ValueError('TENSOR_OBSERVER_BYTE_BOUND')
        raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return dict(shape=shape,dtype=str(value.dtype),SHA256=hashlib.sha256(memoryview(raw)).hexdigest(),
                    **({'version':value._version} if version else {}))
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):
            raise TypeError('UNSUPPORTED_FORWARD_MAPPING_KEYS')
        return {k:describe(v,version=version) for k,v in sorted(value.items())}
    if isinstance(value,(tuple,list)):
        return {'kind':type(value).__name__,'items':[describe(x,version=version) for x in value]}
    if type(value) in (bool,int,str,type(None)) or type(value) is float and math.isfinite(value):
        return value
    raise TypeError('UNSUPPORTED_FORWARD_VALUE:'+type(value).__name__)


def digest(value, *, version=False):
    return hashlib.sha256(json.dumps(describe(value,version=version),sort_keys=True,
                                    allow_nan=False,separators=(',',':')).encode()).hexdigest()


def rng():
    s=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(algorithm=s[0],keys=s[1].tolist(),position=int(s[2]),
                has_gauss=int(s[3]),cached_gaussian=float(s[4])),
                torch_CPU=torch.get_rng_state(),torch_device=torch.cuda.get_rng_state(),
                device_index=torch.cuda.current_device())


def restore_rng(s):
    if torch.cuda.current_device()!=s['device_index']:
        raise ValueError('RNG_DEVICE_CHANGED')
    random.setstate(s['python'])
    n=s['numpy']
    np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],
                         n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(s['torch_CPU'])
    torch.cuda.set_rng_state(s['torch_device'])


ROTARY=('inv_freq','original_inv_freq','attention_scaling','max_seq_len_cached','original_max_seq_len')


def copy_value(x):
    return x.detach().clone() if isinstance(x,torch.Tensor) else x


class ForwardState:
    def __init__(self,model):
        self.model=model
        self.modules=tuple(model.named_modules())
        self.parameters={name:(p,p._version) for name,p in model.named_parameters()}
        self.modes={name:m.training for name,m in self.modules}
        self.context=self.runtime_context()
        self.buffers={}
        self.attrs={}
        total=0
        for name,m in self.modules:
            for key,x in m._buffers.items():
                if x is not None:
                    total+=x.numel()*x.element_size()
                self.buffers[name,key]=(x,copy_value(x))
            attrs=('rope_deltas',) if hasattr(m,'rope_deltas') else ()
            if 'RotaryEmbedding' in type(m).__name__:
                attrs+=ROTARY
            for key in attrs:
                if key in m.__dict__ and key not in m._buffers:
                    x=m.__dict__[key]
                    if isinstance(x,torch.Tensor):
                        total+=x.numel()*x.element_size()
                    self.attrs[name,key]=(x,copy_value(x))
        if total>64*2**20:
            raise ValueError('MUTABLE_STATE_BYTE_BOUND')
        self.state_hash=self.state_digest()
        self.bytes=total

    @staticmethod
    def runtime_context():
        return dict(grad=torch.is_grad_enabled(),inference=torch.is_inference_mode_enabled(),
                    autocast=torch.is_autocast_enabled('cuda'))
    def immutable(self):
        if tuple(self.model.named_modules())!=self.modules:
            raise ValueError('MODULE_TREE_CHANGED')
        if {n:(id(p),p._version) for n,p in self.model.named_parameters()}!={n:(id(p),v) for n,(p,v) in self.parameters.items()}:
            raise ValueError('MODEL_PARAMETER_CHANGED')
        if {n:m.training for n,m in self.modules}!=self.modes or self.runtime_context()!=self.context:
            raise ValueError('TRAINING_CONTEXT_CHANGED')
        if {(n,k) for n,m in self.modules for k in m._buffers}!=self.buffers.keys():
            raise ValueError('REGISTERED_BUFFER_INVENTORY_CHANGED')

    def state_digest(self):
        modules=dict(self.modules)
        return digest({'buffers':{f'{n}:{k}':modules[n]._buffers[k] for n,k in self.buffers},
                       'rotary':{f'{n}:{k}':modules[n].__dict__[k] for n,k in self.attrs}})

    def restore(self):
        self.immutable()
        modules=dict(self.modules)
        with torch.no_grad():
            for (n,k),(original,pin) in self.buffers.items():
                if isinstance(original,torch.Tensor) and not torch.equal(original,pin):
                    original.copy_(pin)
                modules[n]._buffers[k]=original
            for (n,k),(original,pin) in self.attrs.items():
                if isinstance(original,torch.Tensor) and not torch.equal(original,pin):
                    original.copy_(pin)
                modules[n].__dict__[k]=original
        if self.state_digest()!=self.state_hash:
            raise ValueError('MUTABLE_STATE_RESTORE_FAILED')
