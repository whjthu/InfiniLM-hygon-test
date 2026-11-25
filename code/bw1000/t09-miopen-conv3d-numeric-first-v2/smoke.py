"""Exact-source projection and bounded observed ATen MIOpen Conv3d evidence."""
import ast
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time

from scope import LIBRARY, LIBRARY_BYTES, LIBRARY_SHA, R, require

NATIVE=R/'results/training-hunyuanvideo-original-source-v1/source'
EMBED='hyvideo/models/transformers/modules/embed_layers.py'
EMBED_PIN={'bytes':7227,'SHA256':'7e025d178c2e67d58cd3c58436ce795c35a5d2aed2a14fb8d7acac97c4d46fd9'}
TRANSFORMER='hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py'
TRANSFORMER_PIN={'bytes':39361,'SHA256':'d1a702891a92e04c80cdc072bc06cffe6e631fb962dae702d3c9244d347302d0'}
CONFIG=R/'results/training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5/transformer/480p_t2v/config.json'
CONFIG_PIN={'bytes':968,'SHA256':'689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae'}
SYNTHETIC={'input':[1,65,1,2,2],'weight':[2048,65,1,1,1],'stride':[1,1,1],
           'padding':[0,0,0],'bias':True,'groups':1}
MARK='T09_SOLE_SYNTHETIC_CONV3D'
KNOWN={'aten::conv3d','aten::convolution','aten::_convolution','aten::miopen_convolution',
       'aten::miopen_convolution_transpose','aten::slow_conv3d_forward','aten::slow_conv_dilated3d',
       'aten::cudnn_convolution','aten::mkldnn_convolution'}
MIOPEN_ROUTE='aten::miopen_convolution'


def source_bytes(path,pin,read_file,deadline):
    require(read_file(path,pin['bytes'],deadline)==pin,'EXACT_ORIGINAL_METADATA_SHA')
    with path.open('rb') as stream:
        raw=stream.read(pin['bytes']+1)
    require(len(raw)==pin['bytes'] and sha256(raw).hexdigest()==pin['SHA256'],'ORIGINAL_METADATA_READBACK_SHA')
    return raw


def invoked_name(expr):
    if isinstance(expr,ast.Attribute):
        return expr.attr
    return expr.id if isinstance(expr,ast.Name) else None


def origins(reader,deadline):
    embed=source_bytes(NATIVE/EMBED,EMBED_PIN,reader,deadline)
    transform=source_bytes(NATIVE/TRANSFORMER,TRANSFORMER_PIN,reader,deadline)
    cfg=json.loads(source_bytes(CONFIG,CONFIG_PIN,reader,deadline))
    tree=ast.parse(embed)
    conv=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and invoked_name(node.func)=='Conv3d']
    blocks=[node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and
            any(isinstance(call,ast.Call) and invoked_name(call.func)=='Conv3d' for call in ast.walk(node))]
    require(len(conv)==len(blocks)==1,'EXACT_ORIGINAL_PATCH_CONV3D_CLASS')
    lines=[(node.lineno,invoked_name(node.func)) for node in ast.walk(blocks[0]) if isinstance(node,ast.Call)
           and node.lineno<=83<=node.end_lineno]
    require(bool(lines),'ORIGINAL_ACTUAL_FORWARD_STACK_LINE83')
    transformer_calls=[node.lineno for node in ast.walk(ast.parse(transform))
        if isinstance(node,ast.Call) and invoked_name(node.func)==blocks[0].name]
    require(bool(transformer_calls),'ORIGINAL_TRANSFORMER_PATCH_EMBED_CALLSITE')
    selected={key:cfg[key] for key in ('in_channels','hidden_size','patch_size','concat_condition',
              'is_reshape_temporal_channels')}
    require(selected==dict(in_channels=32,hidden_size=2048,patch_size=[1,1,1],
             concat_condition=True,is_reshape_temporal_channels=False),
             'ACTUAL_ROOT_VERIFIED_65_CHANNEL_CONFIG_BRANCH')
    return dict(embed=EMBED_PIN,transformer=TRANSFORMER_PIN,configuration=CONFIG_PIN,
                embed_class=blocks[0].name,conv3d_ctor_line=conv[0].lineno,
                actual_stack_call_at_line83=sorted(lines),transformer_ctor_lines=transformer_calls,
                original_config_selected_fields=selected,selected_channels=2*selected['in_channels']+1,
                selected_shape=SYNTHETIC,prototype_is_synthetic=True,
                actual_original_sample_shape_proven=False,source_payloads_are_metadata_only=True)


def descendant(child,parent):
    current=child.cpu_parent
    for _ in range(32):
        if current is parent:
            return True
        if current is None:
            break
        current=current.cpu_parent
    return False


def dispatch_trace(events):
    roots=[event for event in events if event.name==MARK]
    scoped=[event for event in events if event.name in KNOWN and
            len(roots)==1 and descendant(event,roots[0])]
    counts=Counter(event.name for event in scoped)
    convolution=[item for item in scoped if item.name=='aten::conv3d']
    backend=[item for item in scoped if item.name==MIOPEN_ROUTE]
    actual=bool(len(convolution)==len(backend)==1 and descendant(backend[0],convolution[0]))
    shapes={event.name:[list(shape) for shape in event.input_shapes]
            for event in convolution+backend}
    accurate_shapes=all(shapes.get(name,[])[:2]==[SYNTHETIC['input'],SYNTHETIC['weight']]
                        for name in ('aten::conv3d',MIOPEN_ROUTE))
    return dict(marker_count=len(roots),scoped_operator_counts=dict(sorted(counts.items())),
                correct_conv3d_nested_miopen=actual,scoped_operator_shapes=shapes,
                same_structural_shapes=accurate_shapes,ATen_route_passed=actual and accurate_shapes,
                vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED')


def numeric(output,torch):
    shape=list(output.shape)
    count=int(output.numel())
    finite_count=int(torch.count_nonzero(torch.isfinite(output)).item())
    max_delta=float(torch.max(torch.abs(output.float()-65.0)).item()) if count>0 and count==finite_count else None
    return dict(actual_output_shape=shape,expected_output_shape=[1,2048,1,2,2],output_elements=count,
                expected_output_elements=8192,finite_elements=finite_count,reference_value=65.0,
                max_absolute_error=max_delta,tolerance_abs=0.0,dtype=str(output.dtype),
                numeric_pass=shape==[1,2048,1,2,2] and count==finite_count==8192
                and max_delta==0.0 and output.dtype==torch.bfloat16,
                synchronized_single_conv3d=True)


def provider(deadline,authorized_aliases):
    from functools import partial
    from maps_probe import MAX_MAPS,capture,read_candidate
    with Path('/proc/self/maps').open('rb') as stream:
        raw=stream.read(MAX_MAPS+1)
    reader=partial(read_candidate,authorized_aliases=frozenset(authorized_aliases))
    info=capture(raw,time.monotonic()+max(0,deadline-time.time()),reader=reader)
    return dict(info,mapped_provider_qualified=info['selection']['state']=='ONE_EXECUTABLE_EXACT_MIOPEN_GROUP',
                exact_pin_policy='AUTHORIZED_PATH_SAME_MAPPED_DEV_INODE_SAME_FD_SHA_ONLY',
                map_files_permission_probe_attempted=False,
                vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED',native_open_write_syscalls='UNKNOWN_NOT_CAPTURED')


def conv(deadline,report,shape,sealer,authorized_aliases):
    require(time.time()+30<deadline,'ENOUGH_SINGLE_GPU_SCOPE_BUDGET')
    import torch
    require(torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113'
            and torch.version.cuda is None and torch.cuda.device_count()==1,'EXACT_ORIGINAL_HIP_TORCH_AND_SINGLE_DEVICE')
    require(shape==SYNTHETIC,'EXACT_FROZEN_MODEL_FREE_65_CHANNEL_SHAPE')
    bias=torch.zeros(2048,dtype=torch.bfloat16,device='cuda')
    x=torch.ones(shape['input'],dtype=torch.bfloat16,device='cuda')
    w=torch.ones(shape['weight'],dtype=torch.bfloat16,device='cuda')
    torch.cuda.synchronize()
    report['issued_conv3d_calls']=1
    numeric_pin=None
    profiler_exit_error=None
    try:
        with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,profile_memory=False,with_stack=False) as observed:
            with torch.profiler.record_function(MARK):
                y=torch.nn.functional.conv3d(x,w,bias,stride=shape['stride'],padding=shape['padding'])
                torch.cuda.synchronize()
            report['completed_conv3d_calls']=1
            result=numeric(y,torch)
            report['numeric']=result
            report['seals']={}
            numeric_pin=report['seals']['numeric']=sealer(0,None,result)
    except (OSError,ValueError,RuntimeError,TypeError,AttributeError) as error:
        if numeric_pin is None:
            raise
        profiler_exit_error=type(error).__name__

    try:
        trace=(dispatch_trace(observed.events()) if profiler_exit_error is None else
               dict(ATen_route_passed=False,profiler_exit_error=profiler_exit_error,
                    vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED'))
    except (OSError,ValueError,RuntimeError,TypeError,AttributeError) as error:
        trace=dict(ATen_route_passed=False,error_type=type(error).__name__,
                   vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED')
    report['dispatch_trace']=trace
    trace_pin=report['seals']['ATen']=sealer(1,numeric_pin,trace)

    try:
        mapped=provider(deadline,authorized_aliases)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError) as error:
        mapped=dict(mapped_provider_qualified=False,error_type=type(error).__name__,
                    errno=getattr(error,'errno',None),vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED',
                    native_open_write_syscalls='UNKNOWN_NOT_CAPTURED')
    report['provider_mapping']=mapped
    report['seals']['provider']=sealer(2,trace_pin,mapped)
    report['separate_outcomes']=dict(numeric_pass=result['numeric_pass'],ATen_pass=trace['ATen_route_passed'],
                                    exact_mapped_provider_pass=mapped['mapped_provider_qualified'])
    require(result['numeric_pass'] and trace['ATen_route_passed'] and mapped['mapped_provider_qualified'],
            'INCOMPLETE_NUMERIC_OR_ATEN_OR_EXACT_MAPPED_PROVIDER')
