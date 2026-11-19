"""Record SDXL's original embedding paths and first ResNet operator boundaries."""
import hashlib

import torch


FIRST = 'down_blocks.0.resnets.0'
TARGETS = {
    'time_proj': 1,
    'time_embedding.linear_1': 1,
    'time_embedding.act': 1,
    'time_embedding.linear_2': 1,
    'time_embedding': 1,
    'add_time_proj': 1,
    'add_embedding.linear_1': 1,
    'add_embedding.act': 1,
    'add_embedding.linear_2': 1,
    'add_embedding': 1,
    'conv_in': 1,
    'down_blocks.0': 1,
    FIRST: 1,
    FIRST+'.norm1': 1,
    FIRST+'.nonlinearity': 3,
    FIRST+'.conv1': 1,
    FIRST+'.time_emb_proj': 1,
    FIRST+'.norm2': 1,
    FIRST+'.conv2': 1,
}


class EmbeddingResnetProbe:
    """Snapshot forward inputs and outputs without replacing them."""

    def __init__(self, model_type, targets=None, *, cap=64*1024**2):
        self.model_type=model_type
        self.targets=dict(TARGETS if targets is None else targets)
        self.cap=cap
        self.bytes=0
        self.handles=[]
        self.calls={}
        self.events=[]
        self.data={}
        self.attached=False
        self.first_config=None

    def __enter__(self):
        self.handles.append(torch.nn.modules.module.register_module_forward_pre_hook(self._attach))
        return self

    def __exit__(self,*_):
        while self.handles:
            self.handles.pop().remove()
        self.data.clear()

    def _take(self,name,value):
        if name in self.data or not isinstance(value,torch.Tensor) or not value.numel():
            raise ValueError('TRACE_TENSOR_IDENTITY')
        if value.dtype not in (torch.bfloat16,torch.float32,torch.int32,torch.int64):
            raise ValueError('TRACE_TENSOR_DTYPE')
        size=value.numel()*value.element_size()
        if self.bytes+size>self.cap:
            raise ValueError('TRACE_STORAGE_LIMIT')
        self.data[name]=(value.detach().clone(),list(value.stride()))
        self.bytes+=size
        self.events.append(name)

    @staticmethod
    def _argument(args,kwargs,index,names):
        matching=[kwargs[key] for key in names if key in kwargs]
        if len(args)>index:
            if matching:
                raise ValueError('AMBIGUOUS_MODULE_ARGUMENT')
            return args[index]
        if len(matching)!=1:
            raise ValueError('MISSING_MODULE_ARGUMENT')
        return matching[0]

    def _attach(self,model,_args):
        if not isinstance(model,self.model_type):
            return None
        if self.attached:
            raise ValueError('REPEATED_NATIVE_UNET')
        self.attached=True
        if FIRST in self.targets:
            unit=model.get_submodule(FIRST)
            self.first_config=dict(time_embedding_norm=unit.time_embedding_norm,
               skip_time_act=unit.skip_time_act, output_scale_factor=unit.output_scale_factor,
               shortcut=unit.conv_shortcut is not None)
            if self.first_config!=dict(time_embedding_norm='default',skip_time_act=False,
                                       output_scale_factor=1.0,shortcut=False):
                raise ValueError('FIRST_RESNET_RECIPE_CHANGED')
        for path,limit in self.targets.items():
            layer=model.get_submodule(path)

            def enter(_module,args,kwargs,path=path,limit=limit):
                step=self.calls.get(path,0)+1
                if step>limit:
                    raise ValueError('REPEATED_TARGET_ABOVE_EXPECTED_COUNT')
                self.calls[path]=step
                base=f'{path}#{step}'
                x=self._argument(args,kwargs,0,('hidden_states','input_tensor','sample','input'))
                self._take(base+'.in',x)
                if path in ('down_blocks.0',FIRST):
                    self._take(base+'.temb',self._argument(args,kwargs,1,('temb',)))
                return None

            def leave(_module,_args,_kwargs,value,path=path):
                base=f'{path}#{self.calls[path]}'
                if path=='down_blocks.0':
                    if not isinstance(value,tuple) or len(value)!=2 or not value[1]:
                        raise ValueError('NATIVE_DOWN_BLOCK_RESIDUAL_SCHEMA')
                    self._take(base+'.out',value[0])
                    self._take(base+'.first_residual',value[1][0])
                else:
                    self._take(base+'.out',value)
                return None

            self.handles.append(layer.register_forward_pre_hook(enter,with_kwargs=True))
            self.handles.append(layer.register_forward_hook(leave,with_kwargs=True))
        return None

    def report(self):
        if not self.attached or self.calls!=self.targets:
            raise ValueError('INCOMPLETE_EMBEDDING_RESNET_TRACE')
        result={}
        for name,(raw,stride) in self.data.items():
            host=raw.cpu().contiguous()
            if not bool(torch.isfinite(host).all()):
                raise ValueError('NONFINITE_NATIVE_BOUNDARY')
            digest=hashlib.sha256(host.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            floating=host.double()
            result[name]=dict(shape=list(host.shape),stride=stride,dtype=str(host.dtype),sha256=digest,
                              l2=float(floating.norm()),max_abs=float(floating.abs().max()))
        return dict(events=list(self.events),edges=result,call_counts=dict(self.calls),
                    first_resnet_config=self.first_config,device_snapshot_bytes=self.bytes,
                    target_count=len(self.targets),trace_scope='native_forward_only_all_archived_backward_and_optimizer_fields_verified')
