"""Run the exact native model constructor and activation-checkpoint wrapping on meta."""
import argparse
import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

EXPECTED={'double_blocks':54,'single_blocks':0}
CONFIG_SHA='689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae'
LAUNCH_SHA='3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f'
MODEL_SHA='d1a702891a92e04c80cdc072bc06cffe6e631fb962dae702d3c9244d347302d0'
BACKEND_SHA='5b3542994c47b4a190981058f0e8f531532515033e2251dd54a3143c2031560a'
OBSERVER_SHA='0043b4319d2a682e2d0a427e36c966b6ef5a7402419bab828000887d5c6c66f3'
MODEL='HunyuanVideo_1_5_DiffusionTransformer'
MODEL_FILE=Path('hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py')


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def pinned(path,want,maximum=65536):
    need(path.resolve(strict=True)==path and 0<path.stat().st_size<=maximum,'EXACT_NATIVE_MEMBER')
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==want,'NATIVE_MEMBER_DIGEST')
    return raw


def constructor(raw):
    scopes=[item for item in ast.parse(raw).body if isinstance(item,ast.ClassDef) and item.name==MODEL]
    need(len(scopes)==1,'ONE_DEPLOYED_TRANSFORMER_CLASS')
    init,=(node for node in scopes[0].body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    args={p.arg for p in (*init.args.posonlyargs,*init.args.args,*init.args.kwonlyargs)}
    result={}
    for family,count in EXPECTED.items():
        parameter='mm_'+family+'_depth'
        sites=[]
        for node in ast.walk(init):
            paths=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else ()
            if any(ast.unparse(path)=='self.'+family for path in paths):
                sites.append(node)
        need(len(sites)==1 and parameter in args,'ONE_ORIGINAL_BLOCK_FAMILY_CONSTRUCTOR_ASSIGNMENT')
        value=sites[0].value
        calls=[n for n in ast.walk(value) if isinstance(n,ast.Call)]
        modulelists=[n for n in calls if ast.unparse(n.func).split('.')[-1]=='ModuleList']
        ranges=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='range' and len(n.args)==1]
        need(len(modulelists)==len(ranges)==1 and ast.unparse(ranges[0].args[0]) in (parameter,'self.'+parameter),
             'ORIGINAL_MODULELIST_RANGE_USES_OWN_CONFIG_DEPTH')
        if ast.unparse(ranges[0].args[0])=='self.'+parameter:
            bindings=[n for n in ast.walk(init) if isinstance(n,ast.Assign) and
                      any(ast.unparse(t)=='self.'+parameter for t in n.targets)]
            need(len(bindings)==1 and ast.unparse(bindings[0].value)==parameter and bindings[0].lineno<sites[0].lineno,
                 'ORIGINAL_STORED_DEPTH_BINDS_CONSTRUCTOR_ARGUMENT')
        result[family]=dict(config_field=parameter,expected=count,assignment_line=sites[0].lineno,
                            range_line=ranges[0].lineno,constructor_line=init.lineno)
    return result


def run(a):
    need(all(os.environ.get(k)=='' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
         and not os.environ.get('SLURM_STEP_GPUS'),'ONLY_ZERO_GRES_CPU_TOPOLOGY_PREFLIGHT')
    source=a.native_root/MODEL_FILE
    layout=constructor(pinned(source,MODEL_SHA))
    config=json.loads(pinned(a.config,CONFIG_SHA))
    recipe=json.loads(pinned(a.launch,LAUNCH_SHA))['config']
    observer_raw=pinned(a.observer,OBSERVER_SHA)
    backend_raw=pinned(a.backend,BACKEND_SHA)
    sys.path.insert(0,str(a.native_root))
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
    from peft import LoraConfig
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer as Transformer
    need(Path(inspect.getsourcefile(Transformer)).resolve(strict=True)==source
         and torch.__version__=='2.10.0' and torch.version.hip=='6.3.26113'
         and not torch.cuda.is_initialized(),'ORIGINAL_NATIVE_PROVIDER_WITH_NO_DEVICE_INITIALIZATION')
    with torch.device('meta'):
        model=Transformer.from_config(config).to(dtype=torch.bfloat16)
        model.set_attn_mode('torch')
        model.add_adapter(LoraConfig(r=recipe['lora_rank'],lora_alpha=recipe['lora_alpha'],
            target_modules=recipe['lora_target_modules'],lora_dropout=recipe['lora_dropout'],
            bias='none',task_type='FEATURE_EXTRACTION'),adapter_name='default')
    module=ModuleType('gencuda.backends.t09_backward67_observer')
    exec(compile(observer_raw,str(a.observer),'exec'),module.__dict__)
    sys.modules[module.__name__]=module
    tracker=module.CheckpointRanges()
    orig_check,=(n for n in ast.parse(backend_raw).body if isinstance(n,ast.FunctionDef) and
                n.name=='_apply_gradient_checkpointing')
    ns={'__name__':'gencuda.backends.topology_check','__package__':'gencuda.backends','Any':Any}
    exec(compile(ast.Module(body=[orig_check],type_ignores=[]),str(a.backend),'exec'),ns)
    existing={k:len(getattr(model,k)) for k in EXPECTED}
    need(existing==EXPECTED and all(isinstance(getattr(model,k),torch.nn.ModuleList) for k in EXPECTED),
         'ACTUAL_META_NATIVE_FAMILY_COUNTS_AND_PROVIDER')
    ns[orig_check.name](model,tracker)
    wrapped={k:sum(isinstance(block,CheckpointWrapper) for block in getattr(model,k)) for k in EXPECTED}
    need(wrapped==existing and len(tracker.blocks)==54 and all(p.is_meta for p in model.parameters())
         and not torch.cuda.is_initialized(),'EVERY_ACTUAL_NATIVE_POPULATED_BLOCK_WRAPPED_ON_META')
    block_table=list(tracker.blocks.values())
    return dict(passed=True,original_constructor_layout=layout,actual_family_counts=existing,actual_wrapped=wrapped,
        block_count=len(block_table),block_table_SHA256=sha256(json.dumps(block_table,sort_keys=True).encode()).hexdigest(),
        first_populated_path=block_table[0]['path'],last_populated_path=block_table[-1]['path'],
        source_SHA256=MODEL_SHA,config_SHA256=CONFIG_SHA,observer_SHA256=OBSERVER_SHA,
        allocation_device='meta',weight_payloads_opened=0,forward_calls=0,optimizer_updates=0,GPU_calls=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('native-root','config','launch','observer','backend'):
        cli.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(run(cli.parse_args()),sort_keys=True),flush=True)
