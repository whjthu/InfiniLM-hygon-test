"""Observe real training-forward context without changing math or device state."""
import hashlib
import json
import torch


def context_snapshot(model, *, device_type='cuda'):
    modes = {name: bool(module.training) for name, module in model.named_modules()}
    value = dict(device_type=device_type, grad_enabled=torch.is_grad_enabled(),
                 inference_mode=torch.is_inference_mode_enabled(),
                 autocast_enabled=torch.is_autocast_enabled(device_type),
                 autocast_dtype=str(torch.get_autocast_dtype(device_type)),
                 module_training=modes, training_modules=sum(modes.values()),
                 total_modules=len(modes), float32_matmul_precision=torch.get_float32_matmul_precision(),
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if device_type == 'cuda':
        value['backend_flags'] = dict(
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled())
        if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled'):
            value['backend_flags']['cudnn_sdp'] = torch.backends.cuda.cudnn_sdp_enabled()
    value['identity_SHA256'] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return value


def training_mode_changes(before, after):
    a, b = before['module_training'], after['module_training']
    if set(a) != set(b):
        raise ValueError('module inventory changed between observations')
    return [dict(name=name,before=a[name],after=b[name]) for name in a if a[name] != b[name]]


def input_layouts(args,kwargs):
    """Tensor headers only: no scalar/token values or tensor payload reads."""
    result={}
    def walk(value,path):
        if isinstance(value,torch.Tensor):
            if len(result)>=512:raise ValueError('input tensor inventory exceeds512')
            result[path]=dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()),
                              storage_offset=value.storage_offset(),contiguous=value.is_contiguous(),device=str(value.device))
        elif isinstance(value,dict):
            if not all(isinstance(k,str) for k in value):raise ValueError('input keys must be strings')
            for key in sorted(value):walk(value[key],path+'/'+key)
        elif isinstance(value,(tuple,list)):
            for index,item in enumerate(value):walk(item,path+'/'+str(index))
        elif value is not None and not isinstance(value,(str,bool,int,float)):
            raise ValueError('unsupported input layout object')
    walk(args,'args');walk(kwargs,'kwargs');return result
