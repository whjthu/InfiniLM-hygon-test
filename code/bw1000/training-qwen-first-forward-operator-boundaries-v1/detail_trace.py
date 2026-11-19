"""Record targeted native boundaries while deferring device-to-host copies."""
from functools import wraps

import torch

from replay_state import describe


MAX_RETAINED_LOGICAL_BYTES=512*2**20


class DetailTrace:
    def __init__(self,core,rank):
        self.handles=[]
        self.patches=[]
        self.clear()
        model=core.get_submodule('model')
        text=model.language_model
        self.rank=rank
        self.attention=None
        if rank==1:
            self.module(model.get_input_embeddings(),'text_embed')
            self.module(model.visual,'vision_input',entry=True,choose=lambda args,kw:args[0] if args else kw['hidden_states'])
            self.module(model.visual,'vision_output',choose=self.vision)
            self.method(model,'get_image_features',self.vision)
            self.method(model,'get_placeholder_mask',lambda value:value)
            self.method(model,'compute_3d_position_ids',lambda value:value)
            self.module(text,'language_model_entry',entry=True,choose=lambda args,kw:{
                name:kw.get(name) for name in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')})
        if rank==5:
            layer=text.layers[18]
            self.attention=layer.self_attn
            self.module(layer,'decoder18_entry',entry=True,choose=lambda args,kw:{
                'hidden':args[0] if args else kw['hidden_states'],
                'arguments':{n:kw.get(n) for n in ('position_ids','attention_mask','position_embeddings','cache_position')}})
            self.module(layer.input_layernorm,'input_norm')
            for name in ('q_proj','k_proj','v_proj','q_norm','k_norm','o_proj'):
                self.module(self.attention.get_submodule(name),name)
            self.module(self.attention.o_proj,'o_proj_input',entry=True,choose=lambda args,kw:args[0])
            self.module(self.attention,'attention_output',choose=lambda value:value[0])
            self.module(layer.post_attention_layernorm,'attention_plus_residual',entry=True,choose=lambda args,kw:args[0])
            self.module(layer.post_attention_layernorm,'post_attention_norm')
            for name in ('gate_proj','up_proj','down_proj'):
                self.module(layer.mlp.get_submodule(name),'mlp_'+name)
            self.module(layer.mlp,'mlp_output')
            self.module(layer,'decoder18_output')

    @staticmethod
    def vision(value):
        if not hasattr(value,'pooler_output'):
            raise ValueError('ACTUAL_VISION_OUTPUT_SCHEMA')
        return {name:getattr(value,name,None) for name in ('pooler_output','deepstack_features')}

    def clear(self):
        self.records=[]
        self.bytes=0

    def retain(self,value):
        if isinstance(value,torch.Tensor):
            self.bytes+=value.numel()*value.element_size()
            if self.bytes>MAX_RETAINED_LOGICAL_BYTES:
                raise ValueError('TARGETED_TRACE_RETAINED_BYTES_CAP')
            return (value.detach(),value._version)
        if isinstance(value,dict):
            return {k:self.retain(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [self.retain(v) for v in value]
        if value is None or isinstance(value,(int,float,bool,str)):
            return value
        raise TypeError('UNSUPPORTED_NATIVE_TRACE_ELEMENT:'+type(value).__name__)

    def materialize(self,value):
        if isinstance(value,tuple):
            t,version=value
            if t._version!=version:
                raise ValueError('TRACE_TENSOR_MUTATED_AFTER_BOUNDARY')
            return describe(t)|dict(original_stride=list(t.stride()),storage_offset=t.storage_offset())
        if isinstance(value,dict):
            return {k:self.materialize(v) for k,v in value.items()}
        if isinstance(value,list):
            return [self.materialize(x) for x in value]
        return value

    def mark(self,name,value):
        if len(self.records)>=64:
            raise ValueError('TOO_MANY_TARGETED_NATIVE_CALLS')
        self.records.append((name,self.retain(value)))

    def module(self,module,name,*,entry=False,choose=lambda v:v):
        if entry:
            def hook(_m,args,kw):
                self.mark(name,choose(args,kw))
            h=module.register_forward_pre_hook(hook,with_kwargs=True)
        else:
            def hook(_m,_args,result):
                self.mark(name,choose(result))
            h=module.register_forward_hook(hook)
        self.handles.append(h)

    def method(self,owner,name,choose):
        original=getattr(owner,name)
        had=name in owner.__dict__
        prior=owner.__dict__.get(name)
        @wraps(original)
        def wrapper(*a,**kw):
            result=original(*a,**kw)
            self.mark(name,choose(result))
            return result
        setattr(owner,name,wrapper)
        self.patches.append((owner,name,had,prior))

    def observe_sdpa_input(self,module,query,key,value,mask):
        if module is self.attention:
            self.mark('math_sdpa_inputs',dict(query=query,key=key,value=value,mask=mask))

    def observe_sdpa_output(self,module,result):
        if module is self.attention:
            self.mark('math_sdpa_output',result[0])

    def finish(self):
        names=[name for name,_ in self.records]
        need=('text_embed','vision_input','vision_output','get_image_features','get_placeholder_mask',
              'compute_3d_position_ids','language_model_entry') if self.rank==1 else (
              'decoder18_entry','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm',
              'math_sdpa_inputs','math_sdpa_output','o_proj_input','o_proj','attention_output',
              'attention_plus_residual','post_attention_norm','mlp_gate_proj','mlp_up_proj',
              'mlp_down_proj','mlp_output','decoder18_output')
        if set(names)!=set(need):
            raise ValueError('NATIVE_TARGET_TRACE_COVERAGE:'+','.join(sorted(set(need)-set(names))))
        result=[dict(boundary=k,value=self.materialize(v)) for k,v in self.records]
        self.clear()
        return result

    def close(self):
        self.clear()
        for h in self.handles:h.remove()
        for owner,name,had,prior in reversed(self.patches):
            if had:setattr(owner,name,prior)
            else:delattr(owner,name)
