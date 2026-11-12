"""Independent stdlib consumer of the one original CPU terminal and all metadata."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit']
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['remote_exec_acknowledged'] and original['value']['passed']
remote=('TASK='+repr(task)+'\nREQUEST='+repr(original['request_id'])+'\nCLAIM='+repr(claim)+'\n').encode()+b'''
from pathlib import Path
import collections,datetime,hashlib,json,math,os,sys
W=Path(TASK['original_namespace']);pins={};values={}
def load(name):
 p=W/name;raw=p.read_bytes();assert len(raw)<=2*1024**2
 pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());v=json.loads(raw);values[name]=v;return v
out=load('outcome.json');intent=load('intent.json');owner=load('CPU-owner.json');child=load('CPU-child-owner.json');cpu=load('CPU-result.json')
assert out['claim']==intent['claim']==CLAIM and out['request_id']==intent['request_id']==REQUEST
assert out['passed'] and out['returncode']==0 and out['stop_reason'] is None and out['owned_group_reaped']
assert out['CPU_result']==cpu and pins['CPU-result.json']['SHA256']==out['CPU_result_SHA256']
assert child['PID']==owner['PID'] and child['PGID']==owner['PGID'] and child['startticks']==owner['startticks']
assert out['child']==child and out['owner']==intent['owner']
assert intent['epoch']==out['epoch']==cpu['epoch'] and intent['deadline']==out['deadline']==cpu['epoch']+300 and cpu['deadline']==cpu['epoch']+260
assert out['wall_seconds']<300 and cpu['wall_seconds']<260 and len(out['CPU_affinity'])==4 and out['peak_group_RSS_bytes']<=8*1024**3
assert cpu['passed'] and cpu['framework_commit']==TASK['framework_commit'] and cpu['framework_subset_files']==4
assert cpu['GPU_cards']==0 and not cpu['GPU_initialized'] and not cpu['training_forward_backward'] and not cpu['weight_file_open_attempts']
assert not cpu['actual_weight_loading_qualified'] and not cpu['full_checkpoint_SHA256_qualified'] and not cpu['model_initialized_values']
def process(pid):
 try:
  fields=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,PGID=int(fields[2]),startticks=int(fields[19]),state=fields[0])
 except (FileNotFoundError,ProcessLookupError):return None
old=process(child['PID']);assert old is None or old['startticks']!=child['startticks']
host=process(out['owner']['PID']);assert host is None or host['startticks']!=out['owner']['startticks']
members=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  row=process(int(name))
  if row and row['PGID']==child['PGID'] and row['state']!='Z':members.append(row)
assert not members
for remote_name,source_name in TASK['remote_files'].items():
 raw=(W/remote_name).read_bytes();assert hashlib.sha256(raw).hexdigest()==TASK['source_SHA256'][source_name]
rows=load('encoded/input-identities.json');encoded=load('encoded/result.json');progress=load('encoded/progress.json')
assert encoded==cpu['capture'] and encoded['passed'] and encoded['rows']==64 and not encoded['CUDA_initialized'] and not encoded['model_loaded']
assert not encoded['raw_encoded_payload_saved'] and encoded['actual_template_type']=='internvl_hf'
assert encoded['settings']['case']=='T03.sft' and encoded['settings']['template_type'] is None and encoded['settings']['model_type']=='internvl' and encoded['settings']['max_length']==4096
assert encoded['identities_SHA256']==pins['encoded/input-identities.json']['SHA256'] and progress==dict(planned_rows=64,completed_rows=64,failed=False)
assert len(rows)==64 and sum(len(r['fields']) for r in rows)==cpu['encoded_fields']==192
for ordinal,row in enumerate(rows,1):
 assert row['row_ordinal']==ordinal and 0<row['sequence_length']<=4096 and {'input_ids','labels','pixel_values'}==set(row['fields'])
 for field in row['fields'].values():
  assert field['finite'] and len(field['SHA256'])==64 and all(c in '0123456789abcdef' for c in field['SHA256'])
lengths=[r['sequence_length'] for r in rows];assert min(lengths)==cpu['sequence_length_min'] and max(lengths)==cpu['sequence_length_max']
headers=load('checkpoint-headers.json');native=load('native-meta-schema.json');audit=load('native-checkpoint-key-schema.json')
assert len(headers)==4 and len(native)==len(audit['records'])==audit['keys']==audit['meta_state_keys']==cpu['schema_keys']==841
assert audit['passed'] and audit['model_class']=='InternVLForConditionalGeneration' and audit['changed_names']==cpu['renamed_keys']==841
index=json.loads(Path('/mnt/data/nvidia-baseline-20261010/inputs/models/InternVL3_5-8B-HF/model.safetensors.index.json').read_bytes())
seen=set();targets=set();groups=collections.Counter();payload=0;read=0
for shard,header in headers.items():
 assert header['payload_bytes_read']==0 and header['full_file_SHA256'] is None
 assert header['bytes_read']==8+header['header_bytes'];read+=header['bytes_read']
 cursor=0
 for record in sorted(header['records'].values(),key=lambda r:r['offsets']):
  first,last=record['offsets'];assert first==cursor and record['bytes']==last-first and math.prod(record['shape'])*2==last-first and record['dtype']=='BF16';cursor=last
 assert cursor+header['bytes_read']==header['file_bytes']
 payload+=cursor
for row in audit['records']:
 name=row['original'];target=row['target'];record=headers[row['shard']]['records'][name]
 assert name not in seen and target not in targets and index['weight_map'][name]==row['shard'];seen.add(name);targets.add(target)
 assert all(row[k]==record[k] for k in ('shape','dtype','bytes','offsets'))
 assert native[target]==dict(shape=row['shape'],dtype='torch.bfloat16',device='meta')
 groups[name.split('.',1)[0]]+=1
 # Independent namespace correspondence, not the runtime conversion implementation.
 if name.startswith('language_model.lm_head.'):expected=name.removeprefix('language_model.')
 elif name.startswith('language_model.model.'):expected='model.language_model.'+name.removeprefix('language_model.model.')
 else:expected='model.'+name
 assert expected==target
assert seen==set(index['weight_map']) and targets==set(native)
assert dict(groups)==dict(language_model=399,vision_tower=436,multi_modal_projector=6)
assert payload==audit['tensor_bytes']==index['metadata']['total_size']==17056636928
assert read==cpu['header_bytes_read']==107128 and cpu['header_payload_bytes_read']==audit['payload_bytes_read']==0
assert not audit['actual_weight_loading_qualified']
assert len(audit['transforms'])==9 and all(t['type']=='WeightRenaming' for t in audit['transforms'])
assert any(t['source']==['^language_model.model'] and t['scope']=='model' for t in audit['transforms'])
sources=json.loads((W/'src/ORIGINAL_SOURCES.json').read_bytes())['files']
for name,pin in audit['providers'].items():
 expected=sources['transformers/'+name];raw=Path(pin['path']).read_bytes()
 assert len(raw)==pin['bytes']==expected['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']==expected['SHA256']
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=CLAIM,request_id=REQUEST,files=pins,
 original_owner_absent=True,original_supervisor_absent=True,owned_group_members=members,installed_files_verified=len(TASK['remote_files']),
 native_input_rows=64,encoded_fields=192,min_length=min(lengths),max_length=max(lengths),native_template='internvl_hf',implicit_original_template=True,
 complete_schema_keys=841,renamed_keys=841,key_groups=dict(groups),tensor_bytes=payload,header_bytes_read=read,payload_bytes_read=0,
 native_provider_files=2,native_renamings=9,peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_sampled_group_RSS_bytes=out['peak_group_RSS_bytes'],
 original_inside_seconds=cpu['wall_seconds'],original_whole_seconds=out['wall_seconds'],Torch_imported=False,GPU_cards=0,
 actual_weight_loading_qualified=False,full_weight_SHA256_qualified=False,first_update_numerics_qualified=False,Model_GO=False)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -']
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
for name,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
receipt=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_reader_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
else:receipt['stderr_tail']=r.stderr.decode(errors='replace')[-900:]
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),rows=v.get('native_input_rows'),schema_keys=v.get('complete_schema_keys'),owners_absent=v.get('original_owner_absent'),error=receipt.get('stderr_tail'),GPU_cards=0)))
