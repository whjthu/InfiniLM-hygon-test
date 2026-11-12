"""Independent stdlib arithmetic and original owner/source reader; no re-encoding."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_exec_acknowledged'] and original['value']['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)];task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
remote=('TASK='+repr(task)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(original['request_id'])+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,os,sys
W=Path(TASK['original_namespace']);pins={}
def read(name):
 p=W/name;assert p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=2*1024**2
 raw=p.read_bytes();pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return json.loads(raw)
intent=read('intent.json');out=read('outcome.json');cpu=read('CPU-result.json');child=read('CPU-child-owner.json');owner=read('CPU-owner.json')
assert out['claim']==intent['claim']==CLAIM and out['request_id']==intent['request_id']==REQUEST
assert out['passed'] and out['returncode']==0 and out['stop_reason'] is None and out['owned_group_reaped'] and out['CPU_result']==cpu and out['CPU_result_SHA256']==pins['CPU-result.json']['SHA256']
assert child==out['child'] and all(child[k]==owner[k]==cpu['owner'][k] for k in ('PID','PGID','startticks'))
assert out['epoch']==intent['epoch']==cpu['epoch'] and out['deadline']==intent['deadline']==cpu['epoch']+300 and cpu['deadline']==cpu['epoch']+260
assert cpu['passed'] and cpu['framework_commit']==TASK['framework_commit'] and cpu['framework_subset_files']==5 and out['wall_seconds']<300 and cpu['wall_seconds']<260 and len(out['CPU_affinity'])==4 and out['peak_group_RSS_bytes']<=8*1024**3
assert cpu['GPU_cards']==0 and not cpu['GPU_initialized'] and not cpu['weight_file_open_attempts'] and not cpu['actual_model_loaded'] and not cpu['full_Trainer_constructed'] and not cpu['actual_image_embedding_forward'] and not cpu['raw_encoded_values_saved'] and not cpu['actual_GPU_output_dtype_or_serialization_qualified']
assert cpu['actual_native_class_signature_only'] and not cpu['model_constructor_called'] and cpu['native_template']=='llava_onevision1_5' and cpu['native_logits_policy'] is False and cpu['logit_selection_called'] is False and 'logits_to_keep' not in cpu['forward_signature']
for remote_name,source_name in TASK['remote_files'].items():
 raw=(W/remote_name).read_bytes();assert hashlib.sha256(raw).hexdigest()==TASK['source_SHA256'][source_name]
rows=read('native-caller-rows.json');providers=read('native-caller-providers.json');plan=read('capture-budget-description.json');progress=read('caller-progress.json')
assert cpu['rows_SHA256']==pins['native-caller-rows.json']['SHA256'] and cpu['budget_SHA256']==pins['capture-budget-description.json']['SHA256'] and cpu['capture_budget_description']==plan
assert progress==dict(planned_rows=64,completed_rows=64,failed=False) and len(rows)==cpu['completed_rows']==cpu['planned_rows']==64
inputs=[];outputs=[]
for ordinal,r in enumerate(rows,1):
 assert r['ordinal']==ordinal and r['selected_logit_rows']==r['sequence_length'] and r['complete_sequence_and_labels_retained'] and r['native_logits_policy'] is False
 length=r['sequence_length'];assert 0<r['supervised_tokens']<=length<=4096 and 0<r['image_tokens']<=length and r['collated_serialized_bytes']>0
 encoded=r['encoded_fields'];collated=r['collated_fields'];assert {'input_ids','labels','pixel_values'}<=set(encoded) and {'input_ids','labels','pixel_values'}<=set(collated) and 'logits_to_keep' not in collated
 assert all(collated[n]['tensors']==1 and collated[n]['elements']==length and collated[n]['floating_elements']==0 for n in ('input_ids','labels'))
 for doc in (encoded,collated):
  for v in doc.values():assert v['finite'] and len(v['SHA256'])==64 and all(c in '0123456789abcdef' for c in v['SHA256'])
 inputs.append(r['collated_serialized_bytes']+length*4096*4);outputs.append(length*151936*4+r['image_tokens']*4096*4+4)
P=43646976*4;reserve=8*1024**2;maximum=max(3*P,max(inputs),max(outputs))+reserve;total=15*P+8*max(inputs)+8*max(outputs)+25*reserve
assert plan['rows']==64 and plan['parameter_tensors']==504 and plan['parameter_elements']==43646976 and plan['parameter_bytes']==P and plan['full_parameter_and_AdamW_payload_bytes']==15*P
assert plan['max_input_payload_upper_bytes']==max(inputs) and plan['max_output_payload_upper_bytes']==max(outputs) and plan['metadata_RNG_scalar_reserve_per_file']==reserve and plan['planned_files_per_rank']==25 and plan['planned_micro_forwards_per_rank']==8 and plan['planned_optimizer_updates']==2
assert plan['maximum_artifact_upper_bytes']==maximum and plan['per_rank_total_upper_bytes']==total and plan['per_artifact_limit']==2*1024**3-1 and plan['per_rank_total_limit']==8*1024**3
fits=maximum<=plan['per_artifact_limit'] and total<=plan['per_rank_total_limit'];assert plan['within_recorder_limits']==plan['analytical_sufficiency']==fits and plan['per_artifact_excess_upper_bytes']==max(0,maximum-plan['per_artifact_limit']) and plan['per_rank_excess_upper_bytes']==max(0,total-plan['per_rank_total_limit'])
assert not plan['input_truncation_allowed'] and not plan['output_truncation_allowed'] and not plan['actual_GPU_serialized_artifact_bytes_qualified']
assert min(r['sequence_length'] for r in rows)==cpu['min_full_sequence']==253 and max(r['sequence_length'] for r in rows)==cpu['max_full_sequence']==382
assert cpu['capture_admission_state']==('CONSERVATIVE_BOUND_FITS' if fits else 'CONSERVATIVE_BOUND_EXCEEDS_RECORDER')
sources=json.loads((W/'src/ORIGINAL_SOURCES.json').read_bytes())['files'];assert len(sources)==7 and len(providers)==6 and providers==cpu['native_method_providers']
for name,pin in providers.items():
 p=Path(pin['path']);assert p.is_file() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and len(pin['method_source_SHA256'])==64
 assert any(pin['SHA256']==v['SHA256'] and pin['bytes']==v['bytes'] for v in sources.values())
original=json.loads((W/'src/original-launch.json').read_bytes())['command'];assert '--use_logits_to_keep' not in original and '--template' not in original and original[original.index('--max_length')+1]=='4096'
def proc(pid):
 try:
  f=Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split();return dict(state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
for pin in (out['owner'],out['child'],cpu['owner']):
 now=proc(pin['PID']);assert now is None or now['startticks']!=pin['startticks'] or now['state']=='Z'
live=[n for n in os.listdir('/proc') if n.isdigit() and (p:=proc(int(n))) and p['PGID']==out['child']['PGID'] and p['state']!='Z'];assert not live
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,original_outcome=out,rows=rows,providers=providers,plan=plan,artifacts=pins,owners_absent=True,owned_group_empty=True,reader_Torch_imported=False,GPU_GO=False,Model_GO=False)))
'''
with (D/'root-reader-v2-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=original['request_id'],read_only=True,payload_SHA256=hashlib.sha256(remote).hexdigest()),f)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=90)
for name,raw in [('root-reader-v2.stdout.private.json',r.stdout),('root-reader-v2.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,claim=claim,request_id=original['request_id'],stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/'root-reader-v2-result.json').open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,independent_reader_passed=v.get('passed'),owners_absent=v.get('owners_absent'),rows=len(v.get('rows',[])),per_rank_upper_bytes=v.get('plan',{}).get('per_rank_total_upper_bytes'),GPU_GO=False)))
