"""Independently consume the original CPU terminal, complete manifest and owners."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['remote_outcome']=='ORIGINAL_RETURNED' and original['value']['passed']
value=original['value']
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',original['claim']+':coordination/tasks/training-qwen-checkpointing-lifecycle-CPU-v1.json']))
installed={remote:dict(SHA256=task['source_SHA256'][source]) for remote,source in task['remote_files'].items()}
expected=dict(claim=original['claim'],request_id=original['request_id'],epoch=value['epoch'],deadline=value['deadline'],CPU_result_SHA256=value['CPU_result_SHA256'])
remote=('EXPECTED='+repr(expected)+'\nINSTALLED='+repr(installed)+'\n').encode()+b'''
from pathlib import Path
import ast,datetime,hashlib,json,math,os,re,sys,time
start=time.monotonic();W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-checkpointing-lifecycle-CPU-v1')
def read(name):
 p=W/name;a=p.stat();assert 0<a.st_size<=2*1024**2;raw=p.read_bytes();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
intent,ip=read('intent.json');outcome,op=read('outcome.json');cpu,cp=read('CPU-result.json');binding,bp=read('QWEN_NATIVE_BINDING.json');providers,pp=read('providers.json')
assert cp['SHA256']==EXPECTED['CPU_result_SHA256']==outcome['CPU_result_SHA256']
for name in ('claim','request_id','epoch','deadline'):assert intent[name]==outcome[name]==EXPECTED[name]
assert outcome['returncode']==0 and outcome['passed'] and outcome['owned_group_reaped'] and outcome['stop_reason'] is None
assert math.isclose(outcome['deadline']-outcome['epoch'],300,abs_tol=1e-5)
assert cpu['epoch']==outcome['epoch'] and math.isclose(cpu['deadline']-cpu['epoch'],260,abs_tol=1e-5)
assert cpu['passed'] and cpu['normal_contract']==1 and cpu['shared_tests']==13 and len(set(cpu['rejected_boundaries']))==12
assert cpu['GPU_cards']==0 and cpu['GPU_initialized'] is False and cpu['training_forward_backward'] is False and cpu['weight_file_open_attempts']==[]
assert cpu['gradient_checkpointing'] is True and cpu['real_Swift_prepare_adapter'] and cpu['real_PEFT'] and cpu['callback_registered']
assert cpu['binding_SHA256']==bp['SHA256'] and cpu['providers_SHA256']==pp['SHA256']

lifecycle,lp=read('checkpointing-lifecycle.json')
assert lifecycle==cpu['checkpointing_lifecycle']==outcome['checkpointing-lifecycle.json']['value']
assert outcome['checkpointing-lifecycle.json']['SHA256']==lp['SHA256']
assert cpu['actual_native_checkpointing_preparation'] is True and cpu['full_Trainer_constructed'] is False
assert cpu['original_failure_reproduced']=='native Trainer argument differs: gradient_checkpointing'
assert lifecycle['original_requested'] is True and lifecycle['native_Trainer_flag'] is False
assert lifecycle['language_checkpointing'] is True and lifecycle['language_checkpointed_layers']==36
assert lifecycle['language_use_cache'] is False and lifecycle['vision_checkpointing'] is False
assert lifecycle['vision_checkpointing_flags']>=1
provider=lifecycle['provider'];p=Path('/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages/swift/trainers/mixin.py')
raw=p.read_bytes();lines=raw.decode().splitlines(keepends=True);tree=ast.parse(raw)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SwiftMixin')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_gradient_checkpointing')
method_raw=''.join(lines[method.lineno-1:method.end_lineno]).encode()
assert provider==dict(module='swift.trainers.mixin',qualname='SwiftMixin._prepare_gradient_checkpointing',source_file_bytes=len(raw),source_file_SHA256=hashlib.sha256(raw).hexdigest(),method_source_SHA256=hashlib.sha256(method_raw).hexdigest())
assert provider['source_file_SHA256']=='6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9'
assert binding['native_checkpointing_lifecycle']['provider']==provider
assert set(cpu['rejected_boundaries'])=={'Trainer_flag_not_lowered','one_language_layer_disabled','language_cache_enabled','vision_checkpointing_enabled','source_file_SHA256','method_source_SHA256','missing_lifecycle','missing_actual_provider','other_recipe_learning_rate','visual_unfrozen','truncated_manifest','original_recipe_disabled'}
for name,pin in INSTALLED.items():
 raw=(W/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['SHA256']
runtime,_=read('src/RUNTIME_SOURCE_PINS.json')
for pin in runtime['files'].values():
 p=Path(pin['path']);raw=p.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
prior,_=read('src/ORIGINAL_STRUCTURAL_BINDING.json')
assert binding['spec']==prior['spec'] and binding['trainable_manifest']==prior['trainable_manifest']

assert outcome['QWEN_NATIVE_BINDING.json']['SHA256']==bp['SHA256'] and outcome['QWEN_NATIVE_BINDING.json']['value']==binding
assert outcome['providers.json']['SHA256']==pp['SHA256'] and outcome['providers.json']['value']==providers
fm,_=read('src/FRAMEWORK_FILES.json');assert fm['framework_commit']=='e306d8e2e8f8d2bfd21421f1943e6102c673d02e' and len(fm['files'])==9
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
original,_=read('src/ORIGINAL_SOURCE_METADATA.json')
for tag,pin in original['files'].items():
 p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
config=json.loads(Path(original['files']['config']['path']).read_bytes());text=config['text_config']
assert config['model_type']=='qwen3_vl' and text['num_hidden_layers']==36
width=text['hidden_size'];head=text['num_key_value_heads']*text['head_dim'];intermediate=text['intermediate_size'];rank=16
shapes={'self_attn.q_proj':(width,width),'self_attn.k_proj':(head,width),'self_attn.v_proj':(head,width),'self_attn.o_proj':(width,width),
        'mlp.gate_proj':(intermediate,width),'mlp.up_proj':(intermediate,width),'mlp.down_proj':(width,intermediate)}
derived=[]
for layer in range(text['num_hidden_layers']):
 for target,(out_size,in_size) in shapes.items():
  prefix='base_model.model.model.language_model.layers.%d.%s.'%(layer,target)
  for suffix,shape in [('lora_A.default.weight',[rank,in_size]),('lora_B.default.weight',[out_size,rank])]:
   derived.append(dict(name=prefix+suffix,shape=shape,dtype='torch.float32',numel=shape[0]*shape[1]))
assert sorted(derived,key=lambda x:x['name'])==binding['trainable_manifest']
assert len(derived)==cpu['native_trainable_tensors']==504 and sum(p['numel'] for p in derived)==cpu['native_trainable_parameters']==43646976
assert binding['spec']['case']=='T01.sft' and binding['spec']['training']['gradient_checkpointing'] is True
assert binding['spec']['frozen_paths']==['model.visual'] and binding['spec']['language_path']=='model.language_model'
assert binding['initial_trainables_SHA256'] is None and binding['initial_values_qualified'] is False
assert cpu['frozen']=={'model.visual':{'tensors':351,'parameters':576388336}}
runtime_files=cpu['actual_sources'];assert runtime_files
for name,pin in runtime_files.items():
 p=Path(name);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
assert providers['files'] and all(p['SHA256']==original['files']['model_source']['SHA256'] for p in providers['files'].values())
assert providers['classes'] and all(p['type'].startswith('transformers.models.qwen3_vl.') for p in providers['classes'])
owners={}
for label,pin in [('supervisor',outcome['owner']),('child',outcome['child']),('CPU',cpu['owner'])]:
 p=Path('/proc/%d/stat'%pin['PID'])
 try:
  fields=p.read_text().split(') ',1)[1].split();current=int(fields[19]);matched=current==pin['startticks']
 except FileNotFoundError:current=None;matched=False
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=matched)
 assert not matched
group=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:fields=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(fields[2])==outcome['child']['PGID'] and fields[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
tests=(W/'tests.txt').read_bytes();assert b'Ran 13 tests' in tests and b'OK' in tests
result=dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=outcome['claim'],request_id=outcome['request_id'],
 original_files={'intent':ip,'outcome':op,'CPU_result':cp,'binding':bp,'providers':pp},epoch=outcome['epoch'],deadline=outcome['deadline'],
 source_framework_files=9,original_source_metadata_files=5,runtime_source_files=len(runtime_files),provider_classes=len(providers['classes']),
 actual_model_class=cpu['actual_model_class'],meta_parameter_count=cpu['meta_parameter_count'],
 native_trainable_tensors=504,native_trainable_parameters=43646976,independently_derived_complete_manifest=True,
 frozen=cpu['frozen'],gradient_checkpointing=True,shared_tests=13,actual_invalid_boundaries=12,
 wall_seconds=outcome['wall_seconds'],CPU_wall_seconds=cpu['wall_seconds'],peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_group_RSS_bytes=outcome['peak_group_RSS_bytes'],
 owners=owners,owned_group_empty=True,versions=cpu['versions'],weight_values_read=False,GPU_initialized=False,GPU_cards=0,
 actual_meta_Swift_PEFT_qualified=True,initial_values_qualified=False,capture_size_qualified=False,numerics_qualified=False,GPU_GO=False,Model_GO=False,
 reader_wall_seconds=time.monotonic()-start,actual_native_checkpointing_preparation=True,full_Trainer_constructed=False,original_failure_reproduced=cpu['original_failure_reproduced'],checkpointing_lifecycle=lifecycle,lifecycle_file=lp,installed_package_files=len(INSTALLED),installed_package_SHA256_verified=True,runtime_dependency_pins=len(runtime['files']))
print(json.dumps(result))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
with (D/'root-reader-intent.json').open('x',encoding='utf8') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),claim=original['claim'],request_id=original['request_id'],read_only=True),f,indent=2)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
with (D/'root-reader.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'root-reader.stderr.private.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),independently_derived_manifest=v.get('independently_derived_complete_manifest'),runtime_source_files=v.get('runtime_source_files'),owned_group_empty=v.get('owned_group_empty'),GPU_cards=0)))
