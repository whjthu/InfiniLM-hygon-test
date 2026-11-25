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


def mapped_provider():
    with Path('/proc/self/maps').open('rb') as file:
        raw=file.read((2<<20)+1)
    require(len(raw)<=2<<20,'BOUNDED_CURRENT_PROCESS_MAPS')
    expected=LIBRARY.stat()
    require(LIBRARY.resolve(strict=True)==LIBRARY,'CANONICAL_MIOPEN_PATH')
    selected=[]
    for line in raw.decode('utf8').splitlines():
        cols=line.split(maxsplit=5)
        if len(cols)!=6 or 'libMIOpen.so' not in cols[-1]:
            continue
        _,perms,_,dev,inode,name=cols
        require(name==str(LIBRARY) and int(inode)==expected.st_ino and
                tuple(int(item,16) for item in dev.split(':'))==(os.major(expected.st_dev),os.minor(expected.st_dev)),
                'UNEXPECTED_OR_CHANGED_MAPPED_MIOPEN_OBJECT')
        selected.append(perms)
    require(selected and any('x' in mode for mode in selected),'ACTUAL_SELECTED_MIOPEN_EXECUTABLE_MAPPING')
    return dict(path=str(LIBRARY),dev_inode=[expected.st_dev,expected.st_ino],segments=len(selected),mapped=True)


def dispatch_trace(events):
    scoped=[]
    visited_roots=0
    for event in events:
        if event.name==MARK:
            visited_roots+=1
        if event.name not in KNOWN:
            continue
        parent=event.cpu_parent
        for _ in range(32):
            if parent is None:
                break
            if parent.name==MARK:
                scoped.append((event.name,[list(shape) for shape in event.input_shapes]))
                break
            parent=parent.cpu_parent
    names=Counter(name for name,_ in scoped)
    route=visited_roots==1 and names['aten::conv3d']==1 and names[MIOPEN_ROUTE]==1
    return dict(marker_count=visited_roots,scoped_operator_counts=dict(sorted(names.items())),
                observed_dispatch_event_shapes={name:shape for name,shape in scoped if name in ('aten::conv3d',MIOPEN_ROUTE)},
                scoped_event_SHA256=sha256(json.dumps(scoped,sort_keys=True).encode()).hexdigest(),
                actual_miopen_backend_dispatch_traced=route,vendor_C_ABI_symbol_entry='UNKNOWN_NOT_DIRECTLY_TRACED')


def conv(deadline,report,shape):
    require(time.time()+20<deadline,'ENOUGH_SINGLE_GPU_SCOPE_BUDGET')
    import torch
    require(torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113'
            and torch.version.cuda is None and torch.cuda.device_count()==1,'EXACT_ORIGINAL_HIP_TORCH_AND_SINGLE_DEVICE')
    xshape=tuple(shape['input'])
    wshape=tuple(shape['weight'])
    stride=tuple(shape['stride'])
    padding=tuple(shape.get('padding',[0,0,0]))
    require(len(xshape)==len(wshape)==5 and len(stride)==len(padding)==3 and
            all(type(value) is int and value>0 for value in xshape+wshape+stride)
            and all(type(value) is int and value==0 for value in padding)
            and xshape[0]==1 and xshape[1]==wshape[1],'BOUNDED_MATCHED_SYNTHETIC_SHAPE')
    from math import prod
    require(prod(xshape)<=1<<20 and prod(wshape)<=1<<21,'SMOKE_BOUNDED_ELEMENT_COUNT')
    out_shape=(1,wshape[0],*((size-kernel)//hop+1 for size,kernel,hop in zip(xshape[2:],wshape[2:],stride)))
    require(all(value>0 for value in out_shape) and prod(out_shape)<=1<<22,'SMOKE_OUTPUT_DENOMINATOR')
    bias=torch.zeros(wshape[0],dtype=torch.bfloat16,device='cuda') if shape.get('bias',True) else None
    x=torch.ones(xshape,dtype=torch.bfloat16,device='cuda')
    w=torch.ones(wshape,dtype=torch.bfloat16,device='cuda')
    torch.cuda.synchronize()
    report['issued_conv3d_calls']=1
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,profile_memory=False,with_stack=False) as observed:
        with torch.profiler.record_function(MARK):
            y=torch.nn.functional.conv3d(x,w,bias,stride=stride,padding=padding)
            torch.cuda.synchronize()
    report['completed_conv3d_calls']=1
    report['dispatch_trace']=dispatch_trace(observed.events())
    observed_shapes=report['dispatch_trace']['observed_dispatch_event_shapes']
    report['dispatch_trace']['same_structural_shapes']=all(observed_shapes.get(name,[])[:2]==[list(xshape),list(wshape)]
                         for name in ('aten::conv3d',MIOPEN_ROUTE))
    report['provider_mapping']=mapped_provider()
    value=wshape[1]*prod(wshape[2:])
    wanted=torch.tensor(value,dtype=torch.bfloat16).item()
    report['actual_output_shape']=list(y.shape)
    report['numeric_pass']=tuple(y.shape)==out_shape and bool(torch.isfinite(y).all()) and bool((y==wanted).all())
    require(report['numeric_pass'],'SYNTHETIC_CONV3D_EXACT_NUMERIC_RESULT')
    require(report['dispatch_trace']['actual_miopen_backend_dispatch_traced']
            and report['dispatch_trace']['same_structural_shapes'],'ACTUAL_MIOPEN_ATEN_TRACE_OR_SHAPE_MISSING')
