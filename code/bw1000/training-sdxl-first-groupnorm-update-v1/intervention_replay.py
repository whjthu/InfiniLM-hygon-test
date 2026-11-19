"""Apply exactly one already qualified, frozen-prefix SDXL normalization intervention."""
from pathlib import Path
import argparse
import hashlib
import json

import torch

import hygon_sdxl_first_step as frozen_replay


TARGET='down_blocks.0.resnets.0'


def pin(v):
    t=v.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),
                SHA256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class ReplaceFirstNorm:
    def __init__(self, model_type, binding):
        self.model_type=model_type
        self.binding=binding
        self.handles=[]
        self.evidence={}
        self.model_calls=0

    def __enter__(self):
        self.handles.append(torch.nn.modules.module.register_module_forward_pre_hook(self.attach))
        return self

    def __exit__(self,*_):
        while self.handles:
            self.handles.pop().remove()

    def attach(self, model, _args):
        if not isinstance(model,self.model_type):
            return None
        self.model_calls+=1
        assert self.model_calls==1
        names=[name for name,p in model.named_parameters() if p.requires_grad]
        assert len(names)==1120 and not any(n.startswith(TARGET+'.') or n.startswith('conv_in.') for n in names)
        unit=model.get_submodule(TARGET)
        self.handles.extend((unit.norm1.register_forward_hook(self.replace),
                             unit.conv1.register_forward_hook(self.capture_conv)))
        return None

    def replace(self, layer, args, native):
        assert 'replaced_norm' not in self.evidence and len(args)==1
        hidden=args[0]
        assert not any(t.requires_grad for t in (hidden,native,layer.weight,layer.bias))
        assert native.dtype==torch.float32 and hidden.dtype==torch.bfloat16 and layer.num_groups==32 and layer.eps==1e-5
        assert pin(hidden)==self.binding['hidden'] and pin(native)==self.binding['native_norm']
        assert pin(layer.weight)==self.binding['weight'] and pin(layer.bias)==self.binding['bias']
        with torch.autocast('cuda',enabled=False):
            f32=hidden.float().reshape(hidden.shape[0],layer.num_groups,-1)
            variance,mean=torch.var_mean(f32,dim=-1,correction=0,keepdim=True)
            norm=((f32-mean)*(variance+layer.eps).rsqrt()).reshape_as(hidden)
            gamma,beta=(t.float().reshape(1,-1,1,1) for t in (layer.weight,layer.bias))
            replacement=torch.addcmul(beta,norm,gamma)
        self.evidence.update(native_norm=pin(native),replaced_norm=pin(replacement),frozen_prefix=True)
        assert self.evidence['replaced_norm']==self.binding['replacement_norm']
        return replacement

    def capture_conv(self,_layer,_args,value):
        assert 'conv1' not in self.evidence
        self.evidence['conv1']=pin(value)
        assert self.evidence['conv1']==self.binding['replacement_conv1']
        return None


def main():
    p=argparse.ArgumentParser()
    for n in ('reference','policy','weights','output'):
        p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--accelerator',choices=('hip',),required=True)
    a=p.parse_args()
    from diffusers import UNet2DConditionModel
    claim=json.loads(Path(__file__).with_name('CLAIM.json').read_bytes())
    binding=frozen_replay.verify_reference(a.reference,a.policy)
    a.output.mkdir(mode=0o700,exist_ok=False)
    with ReplaceFirstNorm(UNet2DConditionModel,claim['intervention']) as observer:
        frozen_replay.replay(a.reference,a.weights,a.output/'replay',binding,accelerator='hip')
    assert observer.model_calls==1 and set(observer.evidence)=={'native_norm','replaced_norm','frozen_prefix','conv1'}
    with (a.output/'intervention.json').open('x',encoding='utf-8') as f:
        json.dump(observer.evidence,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
