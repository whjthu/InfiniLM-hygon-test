"""Observe the original attention graph while deferring device tensor records."""
import torch

from detail_trace import DetailTrace

EDGES=('norm1','qkv','attention','norm2','mlp','output')
BASE={'text_embed','vision_input','vision_output','get_image_features','get_placeholder_mask',
      'language_model_entry','image_feature_requires_grad'}


def grad_flags(value):
    if isinstance(value,torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value,(list,tuple)):
        return [grad_flags(item) for item in value]
    raise TypeError('UNEXPECTED_VISION_FIELD')


class VisionTrace(DetailTrace):
    def __init__(self,core,rank):
        super().__init__(core,1,None)
        self.actual_rank=rank
        self.required=set(BASE)
        model=core.get_submodule('model')
        vision=model.visual
        if any(p.requires_grad for p in vision.parameters()):
            raise ValueError('VISION_BACKBONE_MUST_BE_FROZEN')
        if rank==1:
            self.add_blocks(vision)
        self.observe_feature_grad_flags(model)

    def observe_feature_grad_flags(self,model):
        original=model.get_image_features
        prior=model.__dict__['get_image_features']
        def wrapped(*args,**kwargs):
            output=original(*args,**kwargs)
            self.mark('image_feature_requires_grad',
                      {key:grad_flags(value) for key,value in self.vision(output).items()})
            return output
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
        flags=[p['value'] for p in out if p['boundary']=='image_feature_requires_grad']
        if len(flags)!=1:
            raise ValueError('EXACT_IMAGE_FIELD_GRAD_FLAGS_COUNT')
        if 'compute_3d_position_ids' not in names:
            entry=[p['value']['position_ids'] for p in out if p['boundary']=='language_model_entry']
            if len(entry)!=1 or not isinstance(entry[0],dict) or 'SHA256' not in entry[0]:
                raise ValueError('MISSING_ACTUAL_POSITION_IDS')
        observed={'records':out,'grad_flags':flags[0]}
        self.clear()
        return observed
