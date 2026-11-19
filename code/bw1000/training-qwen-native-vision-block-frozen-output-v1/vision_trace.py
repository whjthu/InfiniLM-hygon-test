"""Observe native vision internals and substitute only the two consumed output fields."""
from copy import copy
from functools import wraps

import torch

from detail_trace import DetailTrace
from replay_state import digest

EDGES=('norm1','qkv','attention','norm2','mlp','output')
BASE={'text_embed','vision_input','vision_output','get_image_features','get_placeholder_mask',
      'language_model_entry','delivered_image_features'}


def clone_frozen(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad:
            raise ValueError('VISION_OUTPUT_REQUIRES_GRAD')
        return value.detach().clone()
    if isinstance(value,(tuple,list)):
        return type(value)(clone_frozen(v) for v in value)
    raise TypeError('UNEXPECTED_FROZEN_VISION_FIELD:'+type(value).__name__)


def geometry(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad:
            raise ValueError('VISION_OUTPUT_REQUIRES_GRAD')
        return tuple(value.shape),value.dtype,value.device
    if isinstance(value,(list,tuple)):
        return type(value)(geometry(item) for item in value)
    raise TypeError('UNEXPECTED_VISION_FIELD_GEOMETRY')


class VisionTrace(DetailTrace):
    def __init__(self,core,rank):
        super().__init__(core,1,None)
        self.actual_rank=rank
        self.pin=None
        self.use_pin=False
        self.required=set(BASE)
        model=core.get_submodule('model')
        vision=model.visual
        if any(p.requires_grad for p in vision.parameters()):
            raise ValueError('VISION_BACKBONE_MUST_BE_FROZEN')
        if rank==1:
            self.add_blocks(vision)
        original=model.get_image_features
        prior=model.__dict__['get_image_features']

        @wraps(original)
        def wrapped(*args,**kwargs):
            raw=original(*args,**kwargs)
            value=self.vision(raw)
            if self.pin is None:
                if self.use_pin:raise ValueError('REFERENCE_IMAGE_OUTPUT_MISSING')
                self.pin={k:clone_frozen(v) for k,v in value.items()}
            if self.use_pin:
                raw=copy(raw)
                for key,val in self.pin.items():
                    if geometry(value[key])!=geometry(val):
                        raise ValueError('PINNED_VISION_FIELD_GEOMETRY_CHANGED')
                    setattr(raw,key,clone_frozen(val))
            self.mark('delivered_image_features',self.vision(raw))
            return raw

        model.get_image_features=wrapped
        self.patches.append((model,'get_image_features',True,prior))

    def add_blocks(self,vision):
        if len(vision.blocks)!=27 or len(vision.deepstack_merger_list)!=3:
            raise ValueError('ORIGINAL_VISION_BLOCK_AND_MERGER_COUNT')
        self.add(vision.patch_embed,'patch_embed')
        self.add(vision.pos_embed,'position_embedding_samples')
        self.add(vision.rotary_pos_emb,'vision_rotary')
        self.add(vision.blocks[0],'block00_post_position_input',entry=True,
                 choose=lambda a,k:dict(hidden=a[0],positions=k['position_embeddings'],cu_seqlens=k['cu_seqlens']))
        for index,block in enumerate(vision.blocks):
            for field,mod in zip(EDGES,(block.norm1,block.attn.qkv,block.attn,block.norm2,block.mlp,block)):
                self.add(mod,f'block{index:02d}.{field}')
        for i,merger in enumerate(vision.deepstack_merger_list):
            self.add(merger,f'deepstack_merger_{i}')
        self.add(vision.merger,'vision_merger')

    def add(self,module,name,**kwargs):
        self.required.add(name)
        self.module(module,name,**kwargs)

    def mark(self,name,value):
        if len(self.records)>=192:
            raise ValueError('BOUNDED_VISION_TRACE_CALL_COUNT')
        self.records.append((name,self.retain(value)))

    def finish(self):
        names={name for name,_ in self.records}
        if self.required-names or names-self.required-{'compute_3d_position_ids'}:
            raise ValueError('VISION_BLOCK_COVERAGE:'+','.join(sorted(self.required-names)))
        out=[dict(boundary=k,value=self.materialize(v)) for k,v in self.records]
        delivered=[p['value'] for p in out if p['boundary']=='delivered_image_features']
        if len(delivered)!=1 or not self.pin:
            raise ValueError('EXACT_IMAGE_FIELD_DELIVERY_COUNT')
        if 'compute_3d_position_ids' not in names:
            entry=[p['value']['position_ids'] for p in out if p['boundary']=='language_model_entry']
            if len(entry)!=1 or not isinstance(entry[0],dict) or 'SHA256' not in entry[0]:
                raise ValueError('MISSING_ACTUAL_POSITION_IDS')
        observed={'records':out,'pin_sha':digest(self.pin),'delivered':delivered[0]}
        self.clear()
        return observed
