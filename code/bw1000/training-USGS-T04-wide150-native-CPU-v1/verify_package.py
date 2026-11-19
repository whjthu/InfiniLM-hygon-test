"""Source-only sealing check. --prepared admits false GO; never executes native code."""
import argparse,ast,json
from pathlib import Path
from prepare_inputs import D,need,pinned,verify_source
def verify(prepared=False,check=lambda:None):
    verify_source(check);plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    for k in ('root_GO','CPU_GO','READ_GO','GPU_GO','training_GO','Model_GO','Dataset_GO','formal17_accepted'):
        need(type(claim[k]) is bool,'BOOLEAN_GO')
    if prepared:need(not any(claim[k] for k in ('root_GO','CPU_GO','READ_GO')),'PREPARED_ALL_EXECUTION_FALSE')
    else:need(claim['root_GO'] and claim['CPU_GO'],'ROOT_CPU_ADOPTED')
    need(not any(claim[k] for k in ('GPU_GO','training_GO','Model_GO','Dataset_GO','formal17_accepted')),'NO_DOWNSTREAM_QUALIFICATION')
    need(plan['planned']==dict(train=50,dev=50,blind=50) and plan['CPU']==2 and plan['GPU']==0 and plan['whole_seconds']==300 and plan['work_seconds']==260 and plan['RSS_soft_bytes']==8*1024**3 and plan['RSS_hard_limit'] is False,'EXACT_NEW_RESOURCE_DENOMINATOR')
    fm=json.loads((D/'FRAMEWORK_BINDING.json').read_bytes());need(len(fm['files'])==plan['framework_members']==404,'ACTUAL404_BINDING')
    binding=plan['cases']['T04.sft'];native=json.loads(pinned(D/'NATIVE_BINDING.json',binding['native_binding_pin']))
    need(native['spec']==binding['native_spec'] and binding['resolved_template']=='llava_onevision1_5' and binding['model_type']=='llava_onevision1_5' and binding['attention']==binding['native_spec']['attention']=='eager' and len(binding['model_metadata']['files'])==14 and len(binding['Swift_sources'])==7,'ORIGINAL_T04_RECIPE')
    policy=json.loads((D/'WRAPPER_SOURCES.json').read_bytes());need(len(policy['extra_sources'])==len(policy['chain'])==3,'THREE_REAL_WRAPPERS')
    proof=json.loads((D/'SOURCE_PROOF.json').read_bytes())
    for name,pin in proof['exact_copied_members'].items():pinned(D/name,pin,2*1024**2,check)
    old=json.loads(pinned(D/'T04_FRAMEWORK400_REFERENCE.json',proof['old400_source']))['files'];current=fm['files']
    same=[name for name in old if name in current and all(old[name][k]==current[name][k] for k in ('bytes','SHA256'))]
    need(len(old)==400 and len(same)==399 and set(old)-set(same)=={'tools/hygon_training_recipe.py'},'EXACT399_OLD_MEMBERS')
    need(sorted(set(current)-set(old))==proof['framework_comparison']['added_T03_optin_members'] and len(set(current)-set(old))==4,'FOUR_T03_ONLY_ADDITIONS')
    before=ast.parse(pinned(D/'T04_dispatcher400.reference.txt',old['tools/hygon_training_recipe.py']))
    after=ast.parse(pinned(D/'COMMON_dispatcher404.reference.txt',current['tools/hygon_training_recipe.py']))
    main=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    branch=main.body.pop(1)
    expected=ast.parse("if args and args[0] == 'T03.sft' and '--usgs-t03-config' in args:\n    from usgs_t03_launch import main as t03_main\n    return t03_main(args)\n").body[0]
    need(ast.dump(branch)==ast.dump(expected) and ast.dump(before)==ast.dump(after),'ONLY_EXPLICIT_T03_OPTIN_DISPATCH_DELTA')
    need('tools/hygon_vlm_recipe.py' in same and 'vlm/runtime/template_inputs.py' in same and 'vlm/runtime/first_update_audit.py' in same,'T04_CPU_DEPENDENCIES_UNCHANGED')
    need(len(plan['media'])==30 and sum(x['bytes'] for x in plan['media'].values())==919823 and plan['data_counts']['actual_inputs']==150,'ACTUAL_NEW_MEDIA_PINS')
    parsed={}
    for p in D.glob('*.py'):check();parsed[p.name]=ast.parse(p.read_text(),filename=p.name,feature_version=(3,10))
    calls=lambda tree,name:[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
    process=calls(parsed['native_template_CPU.py'],'get_model_processor');need(len(process)==1,'ONE_REAL_PROCESSOR_CALLSITE')
    kw={x.arg:ast.literal_eval(x.value) for x in process[0].keywords if isinstance(x.value,ast.Constant)}
    need(kw['load_model'] is False and kw['download_model'] is False and kw['device_map']=='cpu','PROCESSOR_WITHOUT_MODEL')
    need('dev/labels-private.jsonl' not in (D/'prepare_inputs.py').read_text() and 'blind/labels-private.jsonl' not in (D/'prepare_inputs.py').read_text(),'NO_HELDOUT_GOLD_PATHS')
    return dict(source_members=len(json.loads((D/'SOURCE_FILES.json').read_bytes())['files']),python_AST=len(parsed),framework404=True,metadata14=True,Swift7=True,wrappers3=True,planned150=True,prepared=prepared,executed_native=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',action='store_true');a=p.parse_args();print(json.dumps(verify(a.prepared)))
