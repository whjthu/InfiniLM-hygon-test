"""Check launch recipe preservation and reject malformed denominator/deadline claims."""
import ast,copy,hashlib,importlib.util,json,math
from pathlib import Path
from deadline_contract import schedule,remaining
from reference_schema import expected_files,validate_capture
from owned_processes import native_command
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_reference',BASE/'run_reference_once.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
native.ROOT=BASE;native.SRC=BASE/'src'
native.SRC.mkdir(exist_ok=True)
(native.SRC/'MASK_CONFIG.json').write_bytes((BASE/'MASK_CONFIG.json').read_bytes())
original=json.loads((BASE/'original-launch.json').read_bytes())
argv,env=native.build(original)
assert original==json.loads((BASE/'original-launch.json').read_bytes())
for key in ('--max_steps','--warmup_steps','--max_length','--lora_rank','--lora_alpha','--lora_dropout','--gradient_accumulation_steps','--per_device_train_batch_size','--gradient_checkpointing','--freeze_vit','--freeze_aligner','--attn_impl','--model','--dataset','--target_modules','--seed','--data_seed','--optim'):
    assert argv[argv.index(key)+1]==original['command'][original['command'].index(key)+1],key
assert argv[argv.index('--callbacks')+1:]==['nvidia_baseline','original_first_updates','native_lora_dropout_alignment']
assert env['NATIVE_LORA_DROPOUT_CONFIG']==str(native.SRC/'MASK_CONFIG.json')
assert env['NATIVE_LORA_DROPOUT_CONFIG_SHA256']==hashlib.sha256((BASE/'MASK_CONFIG.json').read_bytes()).hexdigest()
assert env['NPROC_PER_NODE']=='8' and env['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
assert str(native.RUNTIME_PREFIX/'triton-v1/site') in env['PYTHONPATH']
assert env['PATH'].startswith(str(native.RUNTIME_PREFIX/'bin')+native.os.pathsep)
receipt={'state':'TWO_NATIVE_OPTIMIZER_UPDATES_CAPTURED','rank':0,'world_size':8,'completed_global_step':2,'micro_forwards':8,'optimizer_updates':2,'clip_calls':2,'files':[{'name':n,'bytes':1,'SHA256':'a'*64} for n in sorted(expected_files())]}
validate_capture(receipt,0);rejected=0
for key,bad in [('world_size',1),('completed_global_step',1),('micro_forwards',7),('optimizer_updates',1),('clip_calls',0),('rank',1)]:
    changed=copy.deepcopy(receipt);changed[key]=bad
    try:validate_capture(changed,0)
    except AssertionError:rejected+=1
    else:raise AssertionError(key)
for kind in ('missing','duplicate','bad-hash','too-large'):
    changed=copy.deepcopy(receipt)
    if kind=='missing':changed['files'].pop()
    if kind=='duplicate':changed['files'][0]=changed['files'][1].copy()
    if kind=='bad-hash':changed['files'][0]['SHA256']='z'*64
    if kind=='too-large':changed['files'][0]['bytes']=2*1024**3
    try:validate_capture(changed,0)
    except AssertionError:rejected+=1
    else:raise AssertionError(kind)
clock=schedule(1000,1000);assert clock['deadline']==1900 and clock['native_deadline']==1820
for triplet in ((0,900,1),(1000,1000,1000),(1000,1901,1000),(1000,1900,1900),(1000,1900,999),(math.nan,1900,1100)):
    try:remaining(*triplet)
    except ValueError:rejected+=1
    else:raise AssertionError(triplet)
cmd=[x.encode() for x in argv];assert native_command(cmd,native.ROOT)
foreign=cmd.copy();foreign[foreign.index(b'--output_dir')+1]=b'/unrelated/train';assert not native_command(foreign,native.ROOT)
for p in BASE.glob('*.py'):ast.parse(p.read_bytes())
result={'state':'LOCAL_CPU_RECIPE_AND_REJECT_CONTRACTS_PASS','rejected_malformed_claims':rejected,'source_files_AST':len(list(BASE.glob('*.py'))),'original_recipe_flags_preserved':True,'GPU_execution':False,'training_qualification':False}
with (BASE/'local-contract-checks.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
