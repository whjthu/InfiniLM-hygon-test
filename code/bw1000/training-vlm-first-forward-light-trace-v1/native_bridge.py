"""Lazy bridge to already loaded native model; no Torch import at module import."""
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys

from light_trace import install_if_enabled,need

CONTEXT_PIN=dict(bytes=3002,SHA256='df412bb2982914fb47705a7a90b4a0bf340e6e5551595e754caa474eee6eed76')
RESUME_PIN=dict(bytes=20976,SHA256='4454f738caa07554caa5054234414941e07431416ac8ca3d6b5a3368a14bb02c')


def install(config,*,model=None,trainer=None,rank=None,check=None):
    def factory():return NativeBackend(config,model,trainer,rank,check)
    trace=install_if_enabled(config,rank=rank,backend_factory=factory)
    return trace.attach(model) if trace is not None else None


class NativeBackend:
    def __init__(self,config,model,trainer,rank,check):
        need(callable(check) and model is not None and trainer is not None,'already admitted caller required')
        need('torch' in sys.modules,'observer does not initialize/import Torch')
        self.torch=sys.modules['torch'];self.model,self.trainer,self.rank,self.check=model,trainer,rank,check
        need(config.get('device_type')=='cuda' and config.get('already_admitted_native_device') is True,
             'explicit already-admitted HIP/CUDA context required')
        need(self.torch.cuda.is_initialized(),'observer must not initialize a GPU context')
        need(os.environ.get('WORLD_SIZE')=='8' and os.environ.get('RANK')==os.environ.get('LOCAL_RANK')==str(rank)
             and self.torch.cuda.current_device()==rank,'already admitted eight-rank/local device differs')
        self.root=Path(config['framework_root']).resolve(strict=True)
        self.context=self.provider('vlm.runtime.forward_context_state',CONTEXT_PIN)
        self.rng=self.provider('vlm.runtime.resume_witness',RESUME_PIN)
        self.provider_pins=dict(forward_context=CONTEXT_PIN,resume_rng=RESUME_PIN)

    def provider(self,name,pin):
        # Read exact known source before importing it; don't load an unchecked alternate module.
        relative=Path(*name.split('.')).with_suffix('.py')
        p=self.root/relative
        need(p.resolve(strict=True)==p and not p.is_symlink() and p.stat().st_size==pin['bytes'],
             'native observer provider path/size differs')
        raw=p.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==pin['SHA256'],'native observer provider SHA differs')
        module=importlib.import_module(name)
        need(Path(module.__file__).resolve(strict=True)==p,'loaded observer provider differs')
        return module

    def rng_snapshot(self):
        # Only the already initialized local device, not get_rng_state_all():
        # no new contexts on the other seven devices are requested by this observer.
        value=self.rng.rng_values(lambda:[self.torch.cuda.get_rng_state(self.rank)])
        combined=self.rng.rng_fingerprint(value)
        need(combined['finite'] is True,'nonfinite RNG observation')
        n=value['numpy']
        numpy=dict(algorithm=n[0],keys=n[1].tolist(),position=int(n[2]),has_gauss=int(n[3]),cached_gaussian=float(n[4]))
        components={name:self.rng.fingerprint(self.rng.cpu(state))['SHA256'] for name,state in
            [('python',value['python']),('numpy',numpy),('CPU',value['cpu']),('device_local',value['cuda'][0])]}
        return dict(scope='python_numpy_CPU_and_current_rank_device_only',
                    local_rank=self.rank,SHA256=combined['SHA256'],components_SHA256=components)

    def tensor_header(self,value):
        if not isinstance(value,self.torch.Tensor):return None
        return dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()),
            storage_offset=value.storage_offset(),device=str(value.device),layout=str(value.layout),
            contiguous=value.is_contiguous(),requires_grad=bool(value.requires_grad),
            data_mod_256=value.data_ptr()%256,storage_mod_256=value.untyped_storage().data_ptr()%256)

    def scalar_or_header(self,value):
        if value is None or type(value) in (bool,int,str):return value
        if type(value)is float:return value if math.isfinite(value) else 'NONFINITE_METADATA'
        header=self.tensor_header(value)
        return dict(tensor_header=header,value_not_read=True) if header is not None else dict(type=type(value).__name__,value_not_read=True)

    def metadata(self,module,args,kwargs,output):
        state=self.trainer.state;training_args=self.trainer.args
        normal=dict(global_step=self.scalar_or_header(state.global_step),trainer_epoch=self.scalar_or_header(state.epoch),
            configured_accumulation=self.scalar_or_header(training_args.gradient_accumulation_steps),
            actual_current_accumulation=self.scalar_or_header(getattr(self.trainer,'current_gradient_accumulation_steps',None)),
            model_accepts_loss_kwargs=self.scalar_or_header(getattr(self.trainer,'model_accepts_loss_kwargs',None)),
            average_tokens_across_devices=self.scalar_or_header(getattr(training_args,'average_tokens_across_devices',None)))
        normal['kwargs']={k:self.scalar_or_header(kwargs[k]) for k in
            ('num_items_in_batch','loss_scale','logits_to_keep','use_cache') if k in kwargs}
        scaler=getattr(getattr(self.trainer,'accelerator',None),'scaler',None)
        scale=dict(present=scaler is not None)
        if scaler is not None:
            scale.update(type=type(scaler).__name__,
                initialized_scale=self.scalar_or_header(getattr(scaler,'_scale',None)),
                initial_configured_scale=self.scalar_or_header(getattr(scaler,'_init_scale',None)),
                get_scale_not_called=True)
        # output tensors are headers only: no .item/.cpu/.numpy/activation hash.
        loss=output.get('loss') if isinstance(output,dict) else getattr(output,'loss',None)
        layouts=self.context.input_layouts(args,kwargs)
        alignment={}
        def walk(value,path):
            header=self.tensor_header(value)
            if header is not None:alignment[path]=header;return
            if isinstance(value,dict):
                for key in sorted(value):walk(value[key],path+'/'+key)
            elif isinstance(value,(tuple,list)):
                for index,item in enumerate(value):walk(item,path+'/'+str(index))
        walk(args,'args');walk(kwargs,'kwargs')
        return dict(context=self.context.context_snapshot(module,device_type='cuda'),input_layouts=layouts,
            headers_and_mod256_alignment=alignment,normalization=normal,grad_scaler=scale,
            loss_header=self.tensor_header(loss),output_type=type(output).__name__,
            native_exception_active=sys.exc_info()[0].__name__ if sys.exc_info()[0] is not None else None,
            provider_pins=self.provider_pins,hook_placement='after_existing_pre_and_post_hooks',
            forbidden_payload_reads='.item/.cpu/.numpy not used on model inputs, outputs, weights or activations')
