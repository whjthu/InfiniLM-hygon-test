"""Independent original bytes, config-derived LoRA manifest and owned cleanup."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());value=original['value'];assert original['transport_rc']==0 and value['passed']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',original['claim']+':coordination/tasks/training-internvl-A100-native-contract-CPU-v1.json']))
expected=dict(claim=original['claim'],request_id=original['request_id'],epoch=value['epoch'],deadline=value['deadline'],CPU_result_SHA256=value['CPU_result_SHA256'],installed={remote:task['source_SHA256'][name] for remote,name in task['remote_files'].items()})
remote=('EXPECTED='+repr(expected)+'\n').encode()+b'''
from pathlib import Path
import ast,datetime,hashlib,json,math,os,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-native-update-contract-CPU-v1');start=time.monotonic()
def read(name):
 p=W/name;a=p.stat();assert 0<a.st_size<=2*1024**2;raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
intent,ip=read('intent.json');outcome,op=read('outcome.json');cpu,cp=read('CPU-result.json');binding,bp=read('INTERNVL_NATIVE_BINDING.json');providers,pp=read('providers.json');lifecycle,lp=read('checkpointing-lifecycle.json')
for k in ('claim','request_id','epoch','deadline'):assert intent[k]==outcome[k]==EXPECTED[k]
assert outcome['passed'] and outcome['returncode']==0 and outcome['stop_reason'] is None and outcome['owned_group_reaped']
assert outcome['deadline']-outcome['epoch']==300 and cpu['deadline']-cpu['epoch']==260
assert cpu['epoch']==outcome['epoch'] and cpu['passed'] and cpu['shared_tests']==20 and len(set(cpu['rejected_boundaries']))==16
assert cpu['GPU_cards']==0 and cpu['GPU_initialized'] is False and cpu['weight_file_open_attempts']==[] and cpu['training_forward_backward'] is False and cpu['initial_values_qualified'] is False
assert cp['SHA256']==EXPECTED['CPU_result_SHA256']==outcome['CPU_result_SHA256'] and bp['SHA256']==cpu['binding_SHA256'] and pp['SHA256']==cpu['providers_SHA256']
for name,pin,document in [('INTERNVL_NATIVE_BINDING.json',bp,binding),('providers.json',pp,providers),('checkpointing-lifecycle.json',lp,lifecycle)]:
 assert outcome[name]['SHA256']==pin['SHA256'] and outcome[name]['value']==document
for name,pin in EXPECTED['installed'].items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==pin
fm,_=read('src/FRAMEWORK_FILES.json');assert fm['framework_commit']=='bbf3c6fb03b4696fdc0e71ebd6ca4e79c0ee7564' and len(fm['files'])==12
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
sources,_=read('src/ORIGINAL_SOURCES.json')
for pin in sources['files'].values():
 p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
meta,_=read('src/MODEL_METADATA.json');root=Path(meta['model_root'])
for name,pin in meta['files'].items():
 p=root/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
config=json.loads((root/'config.json').read_bytes());text=config['text_config'];assert config['model_type']=='internvl' and text['model_type']=='qwen3' and text['num_hidden_layers']==36 and config['vision_config']['num_hidden_layers']==24
width=text['hidden_size'];kv=text['num_key_value_heads']*text['head_dim'];inter=text['intermediate_size'];rank=16
shapes={'self_attn.q_proj':(width,width),'self_attn.k_proj':(kv,width),'self_attn.v_proj':(kv,width),'self_attn.o_proj':(width,width),'mlp.gate_proj':(inter,width),'mlp.up_proj':(inter,width),'mlp.down_proj':(width,inter)}
derived=[]
for layer in range(36):
 for name,(out_size,in_size) in shapes.items():
  prefix='base_model.model.model.language_model.layers.%d.%s.'%(layer,name)
  for suffix,shape in [('lora_A.default.weight',[rank,in_size]),('lora_B.default.weight',[out_size,rank])]:derived.append(dict(name=prefix+suffix,shape=shape,dtype='torch.float32',numel=math.prod(shape)))
assert sorted(derived,key=lambda r:r['name'])==binding['trainable_manifest']
assert len(derived)==cpu['native_trainable_tensors']==504 and sum(x['numel'] for x in derived)==cpu['native_trainable_parameters']==43646976
index=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
for path in ('model.vision_tower','model.multi_modal_projector'):
 assert cpu['frozen'][path]['tensors']==sum(name.startswith(path+'.') for name in index) and cpu['frozen'][path]['parameters']>0
assert binding['spec']['case']=='T03.sft' and binding['resolved_template_type']=='internvl_hf' and binding['spec']['training']['gradient_checkpointing'] is True
assert binding['spec']['frozen_paths']==['model.vision_tower','model.multi_modal_projector']
assert cpu['actual_model_class']=='InternVLForConditionalGeneration' and cpu['actual_language_class']=='Qwen3Model'
assert cpu['resolved_template_type']=='internvl_hf' and cpu['actual_template_class']=='swift.template.templates.internvl.InternvlhfTemplate'
assert cpu['actual_native_checkpointing_preparation'] and cpu['real_Swift_prepare_adapter'] and cpu['real_PEFT'] and cpu['callback_registered'] and cpu['full_Trainer_constructed'] is False
assert lifecycle==cpu['checkpointing_lifecycle']
assert lifecycle['original_requested'] is True and lifecycle['native_Trainer_flag'] is False and lifecycle['language_checkpointing'] is True and lifecycle['language_checkpointed_layers']==36
assert lifecycle['vision_checkpointing'] is False and lifecycle['vision_checkpointing_flags']>=24 and lifecycle['language_use_cache'] is False
p=Path(sources['files']['swift/trainers/mixin.py']['path']);raw=p.read_bytes();lines=raw.decode().splitlines(keepends=True);tree=ast.parse(raw)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SwiftMixin');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_gradient_checkpointing')
provider=dict(module='swift.trainers.mixin',qualname='SwiftMixin._prepare_gradient_checkpointing',source_file_bytes=len(raw),source_file_SHA256=hashlib.sha256(raw).hexdigest(),method_source_SHA256=hashlib.sha256(''.join(lines[method.lineno-1:method.end_lineno]).encode()).hexdigest())
assert provider==lifecycle['provider']==binding['native_checkpointing_lifecycle']['provider']
families={'transformers.models.internvl.':sources['files']['transformers/models/internvl/modeling_internvl.py']['SHA256'],'transformers.models.qwen3.':sources['files']['transformers/models/qwen3/modeling_qwen3.py']['SHA256']}
assert binding['model_source_SHA256']==families and set(pin['SHA256'] for pin in providers['files'].values())==set(families.values())
seen=set()
for row in providers['classes']:
 matches=[prefix for prefix in families if row['type'].startswith(prefix)];assert len(matches)==1;seen.update(matches)
 assert providers['files'][row['file']]['SHA256']==families[matches[0]]
assert seen==set(families)
for path,pin in cpu['actual_sources'].items():
 p=Path(path);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
owners={}
for name,saved in [('supervisor',outcome['owner']),('child',outcome['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%saved['PID']).read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:current=None
 assert current!=saved['startticks'];owners[name]=dict(PID=saved['PID'],original_startticks=saved['startticks'],current_startticks=current,original_identity_present=False)
group=[]
for name in os.listdir('/proc'):
 if not name.isdigit():continue
 try:fields=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
 except (FileNotFoundError,ProcessLookupError):continue
 if int(fields[2])==outcome['child']['PGID'] and fields[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
tests=(W/'tests.txt').read_bytes();assert b'Ran 20 tests' in tests and b'OK' in tests
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=outcome['claim'],request_id=outcome['request_id'],epoch=outcome['epoch'],deadline=outcome['deadline'],
 original_files={'intent':ip,'outcome':op,'CPU_result':cp,'binding':bp,'providers':pp,'lifecycle':lp},installed_files=len(EXPECTED['installed']),framework_subset_files=12,actual_runtime_source_files=len(cpu['actual_sources']),
 native_trainable_tensors=504,native_trainable_parameters=43646976,independently_derived_complete_manifest=True,frozen=cpu['frozen'],frozen_tensor_counts_match_original_index=True,
 resolved_template_type=cpu['resolved_template_type'],actual_template_class=cpu['actual_template_class'],checkpointing_lifecycle=lifecycle,provider_families=list(families),provider_classes=len(providers['classes']),
 shared_tests=20,actual_invalid_boundaries=16,wall_seconds=outcome['wall_seconds'],CPU_wall_seconds=cpu['wall_seconds'],peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_group_RSS_bytes=outcome['peak_group_RSS_bytes'],owners=owners,owned_group_empty=True,
 metadata_files=13,weight_values_read=False,GPU_initialized=False,GPU_cards=0,full_Trainer_constructed=False,initial_values_qualified=False,numerics_qualified=False,formal_performance=False,Model_GO=False,reader_wall_seconds=time.monotonic()-start)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
with (D/'root-reader-intent.json').open('x',encoding='utf8') as f:json.dump(dict(claim=original['claim'],request_id=original['request_id'],payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True),f,indent=2)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),independently_derived_manifest=v.get('independently_derived_complete_manifest'),runtime_files=v.get('actual_runtime_source_files'),provider_classes=v.get('provider_classes'),owned_group_empty=v.get('owned_group_empty'),GPU_cards=0)))
