"""Pure source AST/MF and metadata arithmetic; never imports native/GPU framework."""
import argparse,ast,json
from pathlib import Path
from common import D,P,need,obj,pinned,verify_source,sha

def verify(prepared=False,check=lambda:None):
    m=verify_source();c=obj(D/'CLAIM.json');q=obj(D/'native144-public.json');need(q['producer_passed'] is q['independent_metadata_verified'] is True and q['native']['completed']==dict(train=72,dev=36,blind=36),'ACTUAL_NATIVE144_DEPENDENCY')
    need(sha((D/'native144-public.json').read_bytes())==P['native144_public']['SHA256']=='4668214929a0a7f8539e5480bc61938a482f8758e0f6931f7dc02ebf2ef9d723','NATIVE144_PUBLIC_PIN')
    for k in ['root_GO','GPU_GO','READ_GO']:need(type(P[k])is bool and P[k] is c[k],'BOOLEAN_MATCHED_GO')
    if prepared:need(P['root_GO'] is P['GPU_GO'] is P['READ_GO'] is False,'PREPARED_ONLY')
    else:need(P['root_GO'] is P['GPU_GO'] is P['READ_GO'] is True,'ROOT_ADOPTED_SOURCE')
    need(P['Model_GO'] is P['Hygon_GPU_GO'] is P['formal17'] is False and c['Model_GO'] is c['Hygon_GPU_GO'] is c['formal17'] is False,'NO_TARGET_OR_MODEL_PROMOTION')
    need(P['whole_seconds']==900 and P['world_size']==1 and P['reference_world_size']==8 and P['planned_updates']==sum(x['updates'] for x in P['phases'])==32 and P['planned_microbatches']==sum(x['microbatches'] for x in P['phases'])==128,'COMPLETE32_128')
    need([(x['name'],x['start'],x['end']) for x in P['phases']]==[('continuous',0,25),('replay2',0,2),('resume',20,25)] and P['rows']==72 and P['context']==4096 and P['accumulation']==4,'EXACT_NATIVE_PHASES')
    need(P['CPU_affinity']==[6,7,8,9] and P['expected_GPU']['index']==5 and P['expected_GPU']['minimum_free_MiB']==38000 and P['expected_GPU']['UUID']=='GPU-19c8d012-5264-d041-a10c-48aa94938099','FIXED_RESOURCE')
    binding=obj(D/'NATIVE_BINDING.json');need(binding['spec']==P['native_case']['native_spec'] and binding['spec']['training']['gradient_checkpointing'] is False and P['attention']=='eager','FROZEN_ORIGINAL_RECIPE')
    need(len(obj(D/'FRAMEWORK_BINDING.json')['files'])==404 and len(P['providers'])==15,'EXACT_FRAMEWORK_PROVIDER_DENOMINATORS');reuse=obj(D/'SOURCE_REUSE.json')
    for n,pin in reuse['exact_copied'].items():pinned(D/n,pin)
    trees={}
    for f in D.glob('*.py'):check();trees[f.name]=ast.parse(f.read_bytes(),filename=f.name)
    def calls(name,attr):return [x for x in ast.walk(trees[name]) if isinstance(x,ast.Call) and ((isinstance(x.func,ast.Name) and x.func.id==attr) or (isinstance(x.func,ast.Attribute) and x.func.attr==attr))]
    need(len(calls('native_entry.py','build_launch'))==1 and len(calls('native_plugin.py','training_step'))==0,'NATIVE_ENTRY_NO_REPLACEMENT_LOOP')
    need(len(calls('native_plugin.py','original'))==3,'ONLY_ORIGINAL_TRAINER_ENTRY_TRAIN_STEP_CLIP_WRAPPERS')
    need(len(calls('native_plugin.py','register_forward_pre_hook'))==len(calls('native_plugin.py','register_forward_hook'))==1,'ACTUAL_FULL_FORWARD_OBSERVATION')
    need('dev/labels-private' not in (D/'common.py').read_text() and 'blind/labels-private' not in (D/'common.py').read_text(),'NO_EVAL_GOLD_HANDLES')
    return dict(passed=True,source_members=len(m),AST_files=len(trees),prepared=prepared,source_READY=P['source_READY'],actual_remote=0,actual_native=0,actual_GPU=0,actual_fixtures=0,actual_weights_or_tensors=0,planned_updates=32,planned_microbatches=128,MF_SHA256=sha((D/'SOURCE_FILES.json').read_bytes()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',action='store_true');a=p.parse_args();print(json.dumps(verify(a.prepared)))
