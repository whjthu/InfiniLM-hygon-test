"""Bounded diagnostic profiling of the exact native update and actual dispatch.

No input values or stacks are recorded. CPU parent/device times overlap;none
of these profiled windows is formal throughput or fleet/SM utilization.
"""
import ast
from collections import Counter,defaultdict
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

ATTENTION_SHA256='1b55a91812b4e82b6b0ba9c7f46036231acca3234a0b6a105ed210954508e2e4'


def instrument_profiled_update(original,record_function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('one original update required')
    baseline=ast.dump(tree,include_attributes=False);body=[];labels=[]
    for node in tree.body[0].body:
        value=node.value if isinstance(node,(ast.Expr,ast.Assign)) else None;label=None
        if isinstance(value,ast.Call):
            fn=value.func
            if isinstance(fn,ast.Name) and fn.id=='model':label='native.forward'
            elif isinstance(fn,ast.Attribute) and fn.attr=='backward':label='native.backward'
            elif isinstance(fn,ast.Attribute) and fn.attr=='clip_grad_norm_':label='native.clip_grad'
            elif isinstance(fn,ast.Attribute) and fn.attr=='step' and isinstance(fn.value,ast.Name) and fn.value.id=='optimizer':label='native.optimizer'
            elif isinstance(fn,ast.Name) and fn.id=='synchronize':label='native.synchronize'
        if label:
            context=ast.Call(func=ast.Name(id='_native_record_function',ctx=ast.Load()),args=[ast.Constant(label)],keywords=[])
            body.append(ast.With(items=[ast.withitem(context_expr=context,optional_vars=None)],body=[node]));labels.append(label)
        else:body.append(node)
    if labels!=['native.forward','native.backward','native.clip_grad','native.optimizer','native.synchronize']:
        raise ValueError('original update phase boundaries differ')
    tree.body[0].body=body;ast.fix_missing_locations(tree);stripped=ast.parse(ast.unparse(tree));unwrapped=[]
    for node in stripped.body[0].body:
        if isinstance(node,ast.With) and isinstance(node.items[0].context_expr,ast.Call) and isinstance(node.items[0].context_expr.func,ast.Name) and node.items[0].context_expr.func.id=='_native_record_function':unwrapped.extend(node.body)
        else:unwrapped.append(node)
    stripped.body[0].body=unwrapped
    if ast.dump(stripped,include_attributes=False)!=baseline:raise ValueError('profile annotations changed original update math AST')
    scope=dict(original.__globals__);scope['_native_record_function']=record_function;ast.fix_missing_locations(tree)
    exec(compile(tree,inspect.getsourcefile(original) or '<native_profile>','exec'),scope)
    result=scope[tree.body[0].name];result.__module__=original.__module__;return result


def tensor_meta(t):
    if t is None:return None
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),requires_grad=bool(t.requires_grad))


@contextmanager
def capture_attention_routes(module,torch,check):
    """Wrap real provider globals and Torch SDPA;restore even on an exception."""
    replacements=[];counts=Counter();metadata={}
    def wrap(owner,key,label):
        old=getattr(owner,key)
        @functools.wraps(old)
        def value(*args,**kwargs):
            check()
            row=dict(route=label,q=tensor_meta(args[0]),k=tensor_meta(args[1]),v=tensor_meta(args[2]),grad_enabled=bool(torch.is_grad_enabled()))
            mask=kwargs.get('attn_mask',args[3] if label=='torch.functional.sdpa' and len(args)>3 else None)
            row.update(mask=tensor_meta(mask),scale=kwargs.get('scale'),is_causal=bool(kwargs.get('is_causal',args[5] if label=='torch.functional.sdpa' and len(args)>5 else False)))
            signature=json.dumps(row,sort_keys=True,separators=(',',':'))
            if signature not in metadata and len(metadata)>=512:raise ValueError('attention metadata bound exceeded')
            metadata[signature]=row;counts[signature]+=1
            with torch.profiler.record_function('dispatch.'+label):return old(*args,**kwargs)
        replacements.append((owner,key,old,value));setattr(owner,key,value)
    try:
        if module is not None:
            for key in ('torch_sdpa','flash_attention_3','flash_attention_2','sage_attention','xformers_attention'):wrap(module,key,'native.attention.'+key)
        wrap(torch.nn.functional,'scaled_dot_product_attention','torch.functional.sdpa')
        yield lambda:[dict(metadata[s],calls=counts[s]) for s in sorted(counts)]
    finally:
        conflict=False
        for owner,key,old,value in reversed(replacements):
            if getattr(owner,key) is value:setattr(owner,key,old)
            else:conflict=True
        if conflict:raise RuntimeError('attention observer ownership changed')


def summarize_events(profiler,output,*,event_limit=200000,byte_limit=64*1024**2):
    import torch
    from torch.autograd.profiler_util import DeviceType
    output=Path(output);events=profiler.events()
    if not 0<len(events)<=event_limit:raise ValueError('profiler event denominator missing or exceeded')
    cuda=defaultdict(lambda:dict(calls=0,duration_us=0.));cpu=defaultdict(lambda:dict(calls=0,self_cpu_us=0.,inclusive_device_us=0.));device_events=0;total_bytes=0;digest=hashlib.sha256();intervals=[]
    path=output/'events.private.jsonl'
    with path.open('xb') as stream:
        for e in events:
            name=str(e.name)
            if len(name)>8192:raise ValueError('unbounded profiler event name')
            is_cuda=e.device_type==DeviceType.CUDA
            row=dict(name=name,device_type=str(e.device_type),device_index=e.device_index,start_us=e.time_range.start,end_us=e.time_range.end,
                duration_us=e.time_range.elapsed_us(),self_cpu_us=e.self_cpu_time_total,inclusive_device_us=e.device_time_total,input_shapes=e.input_shapes)
            kernels=list(e.kernels)
            if len(kernels)>128:raise ValueError('profiler kernel-owner bound exceeded')
            row['kernel_links']=[dict(name=k.name,device=k.device,duration_us=k.duration) for k in kernels]
            parent=e.cpu_parent;ancestry=[]
            while parent is not None and len(ancestry)<32:
                if str(parent.name).startswith(('native.','dispatch.')):ancestry.append(str(parent.name))
                parent=parent.cpu_parent
            row['phase_ancestry']=ancestry
            raw=(json.dumps(row,separators=(',',':'))+'\n').encode()
            if len(raw)>128*1024 or total_bytes+len(raw)>byte_limit:raise ValueError('profiler byte budget exceeded;partial denominator retained')
            stream.write(raw);digest.update(raw);total_bytes+=len(raw)
            if is_cuda:
                device_events+=1;cuda[name]['calls']+=1;cuda[name]['duration_us']+=float(e.time_range.elapsed_us());intervals.append((float(e.time_range.start),float(e.time_range.end)))
            else:
                cpu[name]['calls']+=1;cpu[name]['self_cpu_us']+=float(e.self_cpu_time_total);cpu[name]['inclusive_device_us']+=float(e.device_time_total)
    if device_events==0:raise RuntimeError('No real CUDA device events;do not load a model on profiler fallback')
    union=0.;end=None
    for left,right in sorted(intervals):
        if end is None or left>end:union+=right-left;end=right
        elif right>end:union+=right-end;end=right
    return dict(events=len(events),CUDA_device_events=device_events,raw_event_bytes=total_bytes,raw_event_SHA256=digest.hexdigest(),
        top_CUDA_events=[dict(name=n,**v) for n,v in sorted(cuda.items(),key=lambda p:-p[1]['duration_us'])[:150]],
        top_CPU_ops=[dict(name=n,**v) for n,v in sorted(cpu.items(),key=lambda p:-p[1]['inclusive_device_us'])[:150]],
        device_activity_union_us=union,includes_memcpy_memset_and_possible_overlap=True,inclusive_times_not_additive=True,formal_performance=False,fleet_busy_qualified=False)


def profile_native_update(original,model,optimizer,data,seed,max_grad_norm,output,*,check):
    import importlib,torch
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    module=importlib.import_module('diffsynth.core.attention.attention');provider=Path(module.__file__)
    raw=provider.read_bytes()
    if provider.is_symlink() or len(raw)!=5756 or hashlib.sha256(raw).hexdigest()!=ATTENTION_SHA256:raise ValueError('actual native attention source differs')
    check();started=time.monotonic();update=instrument_profiled_update(original,torch.profiler.record_function)
    with capture_attention_routes(module,torch,check) as rows:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=False,with_stack=False) as prof:
            value=update(model,optimizer,data,seed,max_grad_norm)
        routes=rows()
    result=summarize_events(prof,output)
    if not routes:raise RuntimeError('No attention dispatch observed;do not infer a backend')
    result.update(passed=True,actual_optimizer_updates=1,routes=routes,attention_source_SHA256=ATTENTION_SHA256,
        configured_native_implementation=str(module.ATTENTION_IMPLEMENTATION),diagnostic_wall_seconds=time.monotonic()-started,
        original_update_math_AST_preserved=True,no_tensor_values_recorded=True,private_values_exported=False,Model_GO=False)
    with (output/'profile-summary.json').open('x') as f:json.dump(result,f,indent=2)
    if provider.read_bytes()!=raw:raise ValueError('native attention source bytes changed')
    return value,result


def smoke(output):
    import torch
    from .backends import zimage_training as backend
    instrument_profiled_update(backend._train_step,torch.profiler.record_function)
    if torch.cuda.device_count()!=1:raise ValueError('one owned CUDA device required')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    x=torch.ones((64,64),device='cuda',requires_grad=True);torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=False,with_stack=False) as p:
        y=(x@x).sum();y.backward();torch.cuda.synchronize()
    result=summarize_events(p,output);result.update(passed=True,native_models_loaded=0,actual_native_optimizer_updates=0,tiny_qualification_only=True)
    with (output/'profile-summary.json').open('x') as f:json.dump(result,f,indent=2)
    return result
