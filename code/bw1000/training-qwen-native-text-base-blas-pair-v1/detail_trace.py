"""Record targeted native boundaries while deferring device-to-host copies."""
from functools import wraps

import torch

from replay_state import describe


MAX_RETAINED_LOGICAL_BYTES=512*2**20


def first_input(args, kw):
    return args[0] if args else kw['hidden_states']


def first_output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class DetailTrace:
    def __init__(self,core,rank,focus_layers,branches):
        self.handles=[]
        self.patches=[]
        self.clear()
        model=core.get_submodule('model')
        text=model.language_model
        self.rank=rank
        self.attention = {}
        self.focus_layers = tuple(focus_layers)
        self.branch_names=[]
        if rank==1:
            self.module(model.get_input_embeddings(),'text_embed')
            self.module(model.visual,'vision_input',entry=True,choose=lambda args,kw:args[0] if args else kw['hidden_states'])
            self.module(model.visual,'vision_output',choose=self.vision)
            self.method(model,'get_image_features',self.vision)
            self.method(model,'get_placeholder_mask',lambda value:value)
            self.method(model,'compute_3d_position_ids',lambda value:value)
            self.module(text,'language_model_entry',entry=True,choose=lambda args,kw:{
                name:kw.get(name) for name in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')})
        for index in self.focus_layers:
            layer=text.layers[index]
            prefix=f'decoder_{index:02d}.'
            self.attention[layer.self_attn]=prefix

            def layer_entry(args,kw):
                return {'hidden':args[0] if args else kw['hidden_states'],
                        'arguments':{n:kw.get(n) for n in
                                     ('position_ids','attention_mask','position_embeddings','cache_position')}}

            self.module(layer,prefix+'entry',entry=True,choose=layer_entry)
            self.module(layer.input_layernorm,prefix+'input_norm')
            for name in ('q_proj','k_proj','v_proj','q_norm','k_norm','o_proj'):
                self.module(layer.self_attn.get_submodule(name),prefix+name)
            self.module(layer.self_attn.o_proj,prefix+'o_proj_input',entry=True,choose=first_input)
            self.module(layer.self_attn,prefix+'attention_output',choose=first_output)
            self.module(layer.post_attention_layernorm,prefix+'attention_plus_residual',entry=True,choose=first_input)
            self.module(layer.post_attention_layernorm,prefix+'post_attention_norm')
            self.module(layer.mlp,prefix+'mlp_input',entry=True,choose=first_input)
            for name in ('gate_proj','up_proj','down_proj'):
                self.module(layer.mlp.get_submodule(name),prefix+'mlp_'+name)
            self.module(layer.mlp.down_proj,prefix+'mlp_down_proj_input',entry=True,choose=first_input)
            self.module(layer.mlp,prefix+'mlp_output')
            self.module(layer,prefix+'output',choose=first_output)

        for site, triplet in branches.items():
            for name, module in zip(('base','lora_A','lora_B'),triplet,strict=True):
                entry=site+'.'+name+'.input'
                output=site+'.'+name+'.output'
                self.module(module,entry,entry=True,choose=first_input)
                self.module(module,output)
                self.branch_names.extend((entry,output))

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
        prefix=self.attention.get(module)
        if prefix is not None:
            self.mark(prefix+'math_sdpa_inputs',dict(query=query,key=key,value=value,mask=mask))

    def observe_sdpa_output(self,module,result):
        prefix=self.attention.get(module)
        if prefix is not None:
            self.mark(prefix+'math_sdpa_output',first_output(result))

    def finish(self):
        names=[name for name,_ in self.records]
        need=list(('text_embed','vision_input','vision_output','get_image_features','get_placeholder_mask',
                   'language_model_entry') if self.rank==1 else ())
        boundaries=('entry','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm',
                    'math_sdpa_inputs','math_sdpa_output','o_proj_input','o_proj','attention_output',
                    'attention_plus_residual','post_attention_norm','mlp_input','mlp_gate_proj','mlp_up_proj',
                    'mlp_down_proj_input','mlp_down_proj','mlp_output','output')
        need.extend(f'decoder_{i:02d}.{k}' for i in self.focus_layers for k in boundaries)
        need.extend(self.branch_names)
        difference=set(names)-set(need)
        if self.rank==1 and 'compute_3d_position_ids' in difference:
            difference.remove('compute_3d_position_ids')
        if difference or set(need)-set(names) or len(names)!=len(set(names)):
            raise ValueError('NATIVE_TARGET_TRACE_COVERAGE:'+','.join(sorted(set(need)-set(names))))
        result=[dict(boundary=k,value=self.materialize(v)) for k,v in self.records]
        if self.rank==1 and 'compute_3d_position_ids' not in names:
            supplied=[row['value']['position_ids'] for row in result if row['boundary']=='language_model_entry']
            if len(supplied)!=1 or not isinstance(supplied[0],dict) or 'SHA256' not in supplied[0]:
                raise ValueError('UNOBSERVED_PROVIDER_INPUT_POSITION_IDS')
        logical=self.bytes
        self.clear()
        return result, logical

    def close(self):
        self.clear()
        for h in self.handles:h.remove()
        for owner,name,had,prior in reversed(self.patches):
            if had:setattr(owner,name,prior)
            else:delattr(owner,name)
