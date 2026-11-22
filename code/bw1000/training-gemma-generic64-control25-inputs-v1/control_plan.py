"""Exact source-span derivation; never executes a model or alters the common trainer."""
import ast
import copy
import json
import re
from pathlib import PurePosixPath

from control_contract import need,sha,DOMAIN,PORT

PLAN_SHA='12d67618d386b04ac973e59e4580f32e8cb938e77d7322fb2bd5e4c31bf64de7'
CALLBACK_SHA='167cd49754992a7a0d71c0f46a5013d59254a7485af458bf4a5191b4c428afea'
KEY='gemma_generic_continuous_witness'


def source_functions(raw):
    tree=ast.parse(raw)
    cls,=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}


def derive_callback(raw,view_sha):
    need(sha(raw)==CALLBACK_SHA and re.fullmatch('[a-f0-9]{64}',view_sha),'exact callback source/view identity required')
    source=raw.decode('utf8')
    need(source.count('KEY = "gemma_domain_continuous_witness"')==1,'callback registration anchor differs')
    source=source.replace('KEY = "gemma_domain_continuous_witness"','KEY = '+repr(KEY),1)
    old='DATASET_SHA = "74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc"'
    need(source.count(old)==1,'domain input identity anchor differs')
    source=source.replace(old,'DATASET_SHA = '+repr(view_sha),1)
    tree=ast.parse(source);cls,=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    method,=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture_fresh_initial']
    lines=source.splitlines(keepends=True)
    lines[method.lineno-1:method.end_lineno]=[
        '    def capture_fresh_initial(self, config, trainables, local):\n'
        '        from control_initial import restore_target_initial\n'
        '        return restore_target_initial(config, trainables, local, self)\n']
    source=''.join(lines)
    old='            full_input=fingerprint(cpu(dict(args=argv, kwargs=kwargs))),\n'
    need(source.count(old)==1,'native input witness anchor differs')
    source=source.replace(old,old+'            tokens=control_tokens(kwargs),\n',1)
    source+='''

def control_tokens(kwargs):
    ids,labels=kwargs.get('input_ids'),kwargs.get('labels')
    if not isinstance(ids,torch.Tensor) or not isinstance(labels,torch.Tensor) or labels.ndim!=2:
        raise ValueError('actual full native input/labels required')
    return dict(padded_input_tokens=ids.numel(),shifted_supervised_tokens=int((labels[...,1:]!=-100).sum().item()))
'''
    before,after=source_functions(raw),source_functions(source)
    need(before.keys()==after.keys(),'new source unexpectedly changes callback methods')
    need(all(before[k]==after[k] for k in before if k not in {'capture_fresh_initial','before'}),
         'native update/loss/checkpoint/GC lifecycle changed')
    ast.parse(source,feature_version=(3,10))
    return source.encode('utf8')


def make_plan(raw,root):
    need(sha(raw)==PLAN_SHA,'original successful Gemma command plan differs')
    root=PurePosixPath(root)
    need(root.is_absolute() and root.anchor=='/' and '..' not in root.parts,'canonical POSIX target root required')
    plan=copy.deepcopy(json.loads(raw))
    def move(value):
        if isinstance(value,str):
            old=DOMAIN.as_posix()
            if value==old or value.startswith(old+'/'):return str(root)+value[len(old):]
            if value=='gemma_domain_continuous_witness':return KEY
        if isinstance(value,list):return [move(x) for x in value]
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        return value
    plan=move(plan);argv=plan['command'];value=lambda k:argv[argv.index(k)+1]
    expected={'--model_type':'gemma3_vision','--template':'gemma3_vision','--max_length':'512',
        '--max_pixels':'262144','--max_steps':'25','--per_device_train_batch_size':'1',
        '--gradient_accumulation_steps':'4','--gradient_checkpointing':'false','--attn_impl':'sdpa',
        '--lora_rank':'16','--lora_alpha':'32','--learning_rate':'1e-4','--optim':'adamw_torch'}
    for key,wanted in expected.items():need(argv.count(key)==1 and value(key)==wanted,'frozen native option changed: '+key)
    need(value('--dataset')==str(root/'view/train800-swift.jsonl') and
         value('--output_dir')==str(root/'continuous/train') and KEY in argv and
         '--resume_from_checkpoint' not in argv,'new control is not an isolated fresh continuous25')
    plan['environment_overrides']['MASTER_PORT']=str(PORT)
    need(plan['environment_overrides']['NPROC_PER_NODE']=='8','original eight-rank topology changed')
    plan.update(experiment='generic64-control25',source_unique_rows=64,physical_exposures=800,
        caption_target_format='original unmodified; differs from domain FINAL answers',
        same_update_count_not_equal_token_count=True,formal17=False)
    return plan
