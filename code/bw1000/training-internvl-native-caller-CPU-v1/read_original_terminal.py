"""Independent stdlib consumer of all64 native caller records and byte plan."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_exec_acknowledged'] and original['value']['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
remote=('TASK='+repr(task)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(original['request_id'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,os,sys,time
start=time.monotonic();W=Path(TASK['original_namespace']);pins={}
def read(name):
 p=W/name;raw=p.read_bytes();assert 0<len(raw)<=2*1024**2;pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return json.loads(raw)
intent=read('intent.json');out=read('outcome.json');cpu=read('CPU-result.json');child=read('CPU-child-owner.json');owner=read('CPU-owner.json')
assert out['claim']==intent['claim']==CLAIM and out['request_id']==intent['request_id']==REQUEST
assert out['passed'] and out['returncode']==0 and out['stop_reason'] is None and out['owned_group_reaped']
assert out['CPU_result']==cpu and out['CPU_result_SHA256']==pins['CPU-result.json']['SHA256']
assert child==out['child'] and all(child[k]==owner[k]==cpu['owner'][k] for k in ('PID','PGID','startticks'))
assert intent['epoch']==out['epoch']==cpu['epoch'] and out['deadline']==intent['deadline']==cpu['epoch']+300 and cpu['deadline']==cpu['epoch']+260
assert cpu['passed'] and cpu['framework_commit']==TASK['framework_commit'] and cpu['framework_subset_files']==4 and out['wall_seconds']<300 and cpu['wall_seconds']<260
assert len(out['CPU_affinity'])==4 and out['peak_group_RSS_bytes']<=8*1024**3
assert cpu['GPU_cards']==0 and not cpu['GPU_initialized'] and not cpu['weight_file_open_attempts'] and not cpu['actual_model_loaded'] and not cpu['full_Trainer_constructed']
assert not cpu['training_forward_backward'] and not cpu['actual_image_embedding_forward'] and not cpu['raw_encoded_values_saved'] and not cpu['actual_GPU_serialized_artifact_bytes_qualified']
for remote_name,source_name in TASK['remote_files'].items():
 raw=(W/remote_name).read_bytes();assert hashlib.sha256(raw).hexdigest()==TASK['source_SHA256'][source_name]
rows=read('native-caller-rows.json');providers=read('native-caller-providers.json');plan=read('capture-budget-plan.json');conditions=read('capture-budget-conditions.json');progress=read('caller-progress.json')
assert cpu['rows_SHA256']==pins['native-caller-rows.json']['SHA256'] and cpu['budget_SHA256']==pins['capture-budget-plan.json']['SHA256'] and cpu['capture_budget']==plan
assert progress==dict(planned_rows=64,completed_rows=64,failed=False) and len(rows)==cpu['completed_rows']==cpu['planned_rows']==64
reference=json.loads((W/'src/INPUT_REFERENCE.json').read_bytes());assert len(reference['identities'])==64
assert pins['src/INPUT_REFERENCE.json']['SHA256'] if 'src/INPUT_REFERENCE.json' in pins else True
assert hashlib.sha256((W/'src/INPUT_REFERENCE.json').read_bytes()).hexdigest()==cpu['original_pre_model_reference_SHA256']
assert cpu['native_template']=='internvl_hf' and cpu['native_logits_policy'] is True
assert conditions==dict(case='T03.sft',hidden_size=4096,vocab_size=151936,language_cache=False,hidden_state_stacks=False,attention_maps=False)
assert cpu['capture_conditions']==dict(passed=True,**conditions,actual_model='InternVLForConditionalGeneration')
inputs=[];outputs=[]
for ordinal,r in enumerate(rows,1):
 assert r['ordinal']==ordinal and r['sequence_length']==reference['identities'][ordinal-1]['sequence_length']
 assert r['original_encoded_fields_exact'] and r['complete_input_sequence_unchanged'] and r['non_label_fields_unchanged'] and r['label_and_logit_mask_native_semantics_checked']
 length=r['sequence_length'];selected=r['selected_logit_rows'];assert 0<selected==r['supervised_tokens']+1<=length+1 and 0<r['image_tokens']<=length<=4096
 before=r['before_fields'];after=r['after_fields'];assert {'input_ids','labels','pixel_values'}<=set(before) and set(after)==set(before)|{'logits_to_keep'}
 assert before['labels']['elements']==length and after['labels']['elements']==selected and after['logits_to_keep']['elements']==length
 assert all(before[n]==after[n] for n in set(before)-{'labels'})
 for doc in (before,after):
  for value in doc.values():assert value['finite'] and len(value['SHA256'])==64 and all(c in '0123456789abcdef' for c in value['SHA256'])
 assert r['collated_serialized_bytes']>0
 inputs.append(r['collated_serialized_bytes']+length*4096*4)
 outputs.append(selected*151936*4+r['image_tokens']*4096*4+4)
P=43646976*4;reserve=8*1024**2
assert plan['rows']==64 and plan['parameter_tensors']==504 and plan['parameter_elements']==43646976 and plan['parameter_bytes']==P
assert plan['full_parameter_and_AdamW_payload_bytes']==15*P and plan['max_input_payload_upper_bytes']==max(inputs) and plan['max_output_payload_upper_bytes']==max(outputs)
assert plan['metadata_RNG_scalar_reserve_per_file']==reserve and plan['planned_files_per_rank']==25 and plan['planned_micro_forwards_per_rank']==8 and plan['planned_optimizer_updates']==2
assert plan['maximum_artifact_upper_bytes']==max(3*P,max(inputs),max(outputs))+reserve<plan['per_artifact_limit']==2*1024**3-1
assert plan['per_rank_total_upper_bytes']==15*P+8*max(inputs)+8*max(outputs)+25*reserve<plan['per_rank_total_limit']==8*1024**3
assert plan['analytical_sufficiency'] and not plan['actual_GPU_serialized_artifact_bytes_qualified'] and not plan['input_truncation_allowed'] and not plan['output_truncation_allowed']
assert min(r['sequence_length'] for r in rows)==cpu['min_full_sequence']==283 and max(r['sequence_length'] for r in rows)==cpu['max_full_sequence']==3363
assert min(r['selected_logit_rows'] for r in rows)==cpu['min_selected_logit_rows']==11 and max(r['selected_logit_rows'] for r in rows)==cpu['max_selected_logit_rows']==21
sources=json.loads((W/'src/ORIGINAL_SOURCES.json').read_bytes())['files'];assert len(providers)==6 and providers==cpu['native_method_providers']
for name,pin in providers.items():
 p=Path(pin['path']);raw=p.read_bytes();expected=next(v for v in sources.values() if Path(v['path']).resolve()==p.resolve())
 assert len(raw)==pin['bytes']==expected['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']==expected['SHA256']
 assert len(pin['method_source_SHA256'])==64
original=json.loads((W/'src/original-launch.json').read_bytes())['command'];assert '--use_logits_to_keep' not in original and '--template' not in original
owners={}
for label,pin in [('supervisor',out['owner']),('child',out['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%pin['PID']).read_text().split(') ',1)[1].split()[19])
 except (FileNotFoundError,ProcessLookupError):current=None
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks']);assert current!=pin['startticks']
group=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(f[2])==out['child']['PGID'] and f[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,original_files=pins,
 installed_files_verified=len(TASK['remote_files']),rows=64,full_sequence_range=[283,3363],selected_logit_rows_range=[11,21],actual_native_method_providers=6,
 providers=providers,capture_budget=plan,capture_budget_conditions=conditions,owners=owners,owned_group_empty=True,framework_commit=cpu['framework_commit'],actual_runtime_subset=4,
 original_inside_seconds=cpu['wall_seconds'],original_whole_seconds=out['wall_seconds'],peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_sampled_group_RSS_bytes=out['peak_group_RSS_bytes'],
 full_Trainer_constructed=False,actual_image_embedding_forward=False,actual_GPU_serialized_artifact_bytes_qualified=False,Torch_imported=False,GPU_cards=0,
 actual_model_loaded=False,first_update_numerics_qualified=False,Model_GO=False,reader_wall_seconds=time.monotonic()-start)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_reader_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
else:receipt['stderr_tail']=r.stderr.decode(errors='replace')[-900:]
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),rows=v.get('rows'),owned_group_empty=v.get('owned_group_empty'),error=receipt.get('stderr_tail'),GPU_cards=0)))
