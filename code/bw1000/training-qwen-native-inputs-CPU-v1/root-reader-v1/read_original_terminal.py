"""Independently audit the original 64-row CPU terminal; do not re-encode."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());assert original['transport_rc']==0 and original['value']['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/training-qwen-native-inputs-CPU-v1.json']))
expected=dict(claim=claim,request_id=original['request_id'],epoch=original['value']['epoch'],deadline=original['value']['deadline'],CPU_result_SHA256=original['value']['CPU_result_SHA256'])
pins={remote:task['source_SHA256'][source] for remote,source in task['remote_files'].items()}
remote=('EXPECTED='+repr(expected)+'\nPINS='+repr(pins)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,math,os,sys,time
start=time.monotonic();W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-qwen-native-inputs-CPU-v1')
def read(name):
 p=W/name;a=p.stat();assert 0<a.st_size<=2*1024**2;raw=p.read_bytes();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
intent,ip=read('intent.json');outcome,op=read('outcome.json');cpu,cp=read('CPU-result.json')
capture,cap=read('encoded/result.json');rows,rp=read('encoded/input-identities.json');size,sp=read('pre-template-capture-size.json');progress,_=read('encoded/progress.json')
for name in ('claim','request_id','epoch','deadline'):assert intent[name]==outcome[name]==EXPECTED[name]
assert cp['SHA256']==outcome['CPU_result_SHA256']==EXPECTED['CPU_result_SHA256']
assert outcome['passed'] and outcome['returncode']==0 and outcome['owned_group_reaped'] and outcome['stop_reason'] is None
assert math.isclose(outcome['deadline']-outcome['epoch'],300,abs_tol=1e-5) and cpu['epoch']==outcome['epoch']
assert math.isclose(cpu['deadline']-cpu['epoch'],260,abs_tol=1e-5)
assert cpu['passed'] and cpu['shared_tests']==5 and not cpu['weight_file_open_attempts'] and cpu['GPU_cards']==0
assert not cpu['model_loaded'] and not cpu['first_update_numerics_qualified'] and not cpu['capture_size_qualified']
assert capture['passed'] and capture['rows']==64 and not capture['model_loaded'] and not capture['CUDA_initialized'] and not capture['raw_encoded_payload_saved']
assert capture['identities_SHA256']==rp['SHA256']==cpu['identities_SHA256'] and cpu['size_analysis_SHA256']==sp['SHA256']
assert outcome['encoded/input-identities.json']['SHA256']==rp['SHA256'] and outcome['encoded/input-identities.json']['value']==rows
assert outcome['pre-template-capture-size.json']['SHA256']==sp['SHA256'] and outcome['pre-template-capture-size.json']['value']==size
assert progress==dict(planned_rows=64,completed_rows=64,failed=False) and len(rows)==64
settings=capture['settings'];assert settings['case']=='T01.sft' and settings['template_type']==settings['model_type']=='qwen3_vl'
assert settings['max_length']==4096 and settings['max_pixels']==262144 and settings['seed']==settings['data_seed']==42
assert settings['precision']=='bfloat16' and settings['attention']=='sdpa'
lengths=[];fields=0
for ordinal,row in enumerate(rows,1):
 assert row['row_ordinal']==ordinal and 0<row['sequence_length']<=4096
 lengths.append(row['sequence_length']);assert {'input_ids','labels'}<=set(row['fields'])
 for field in row['fields'].values():
  digest=field['SHA256'];assert field['finite'] is True and len(digest)==64 and set(digest)<=set('0123456789abcdef')
 fields+=len(row['fields'])
assert min(lengths)==cpu['sequence_length_min'] and max(lengths)==cpu['sequence_length_max']==size['max_length']
assert fields==cpu['encoded_fields'] and size['vocab']==151936
assert size['maximum_full_F32_logits_bytes']==max(lengths)*151936*4 and size['all_pre_template_lengths_fit_F32_output_limit']
assert not size['complete_serialized_capture_budget_qualified'] and not size['final_native_collator_observed'] and not size['actual_distributed_first_batch_observed']
for ordinal,(row,bound) in enumerate(zip(rows,size['rows']),1):
 assert bound['row_ordinal']==ordinal and bound['sequence_length']==row['sequence_length']
 assert bound['hypothetical_full_F32_logits_bytes']==row['sequence_length']*151936*4
installed={}
for name,pin in PINS.items():
 p=W/name;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin;installed[name]=dict(bytes=len(raw),SHA256=pin)
processor,_=read('src/PROCESSOR_METADATA.json');model=Path(processor['original_model_root'])
for name,pin in processor['files'].items():
 p=model/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
assets,_=read('src/ORIGINAL_INPUT_FILES.json')
for pin in assets['files']:
 p=Path(pin['source']);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
for name,pin in cpu['actual_sources'].items():
 p=Path(name);assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
owners={}
for label,pin in [('supervisor',outcome['owner']),('child',outcome['child']),('CPU',cpu['owner'])]:
 try:current=int(Path('/proc/%d/stat'%pin['PID']).read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:current=None
 owners[label]=dict(PID=pin['PID'],original_startticks=pin['startticks'],current_startticks=current,original_identity_present=current==pin['startticks'])
 assert current!=pin['startticks']
group=[]
for name in os.listdir('/proc'):
 if name.isdigit():
  try:f=Path('/proc/'+name+'/stat').read_text().split(') ',1)[1].split()
  except (FileNotFoundError,ProcessLookupError):continue
  if int(f[2])==outcome['child']['PGID'] and f[0]!='Z':group.append(int(name))
assert not group and 'torch' not in sys.modules
tests=(W/'tests.txt').read_bytes();assert b'Ran 5 tests' in tests and b'OK' in tests
print(json.dumps(dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=outcome['claim'],request_id=outcome['request_id'],
 original_files={'intent':ip,'outcome':op,'CPU_result':cp,'capture':cap,'identities':rp,'size':sp},epoch=outcome['epoch'],deadline=outcome['deadline'],
 installed_files=len(installed),installed_file_pins=installed,runtime_source_files=len(cpu['actual_sources']),processor_files=len(processor['files']),original_input_files=len(assets['files']),
 actual_template=capture['actual_template_type'],rows=64,encoded_fields=fields,sequence_length_min=min(lengths),sequence_length_max=max(lengths),
 maximum_hypothetical_full_F32_logits_bytes=size['maximum_full_F32_logits_bytes'],full_serialized_capture_budget_qualified=False,
 shared_tests=5,owners=owners,owned_group_empty=True,wall_seconds=outcome['wall_seconds'],CPU_wall_seconds=cpu['wall_seconds'],peak_RSS_bytes=cpu['peak_RSS_bytes'],peak_group_RSS_bytes=outcome['peak_group_RSS_bytes'],
 framework_commit=cpu['framework_commit'],framework_subset_files=cpu['framework_subset_files'],versions=capture['versions'],dataset_SHA256=capture['dataset_SHA256'],
 model_loaded=False,raw_encoded_payload_saved=False,weight_values_read=False,GPU_cards=0,first_update_numerics_qualified=False,GPU_GO=False,Model_GO=False,
 reader_wall_seconds=time.monotonic()-start)))
'''
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
with (D/'root-reader-intent.json').open('x',encoding='utf8') as f:json.dump(dict(payload_SHA256=hashlib.sha256(remote).hexdigest(),claim=claim,request_id=original['request_id'],read_only=True),f,indent=2)
r=subprocess.run(ssh,input=remote,capture_output=True,timeout=60)
with (D/'root-reader.stdout.private.json').open('xb') as f:f.write(r.stdout)
with (D/'root-reader.stderr.private.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
v=receipt.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),rows=v.get('rows'),encoded_fields=v.get('encoded_fields'),sequence_length_min=v.get('sequence_length_min'),sequence_length_max=v.get('sequence_length_max'),owned_group_empty=v.get('owned_group_empty'),GPU_cards=0)))
