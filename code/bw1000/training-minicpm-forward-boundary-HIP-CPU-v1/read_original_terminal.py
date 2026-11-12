"""Independent original CPU182 closure; no test/model replay or new Slurm step."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
original=json.loads((D/'transport-outcome.json').read_bytes());out=original['value'];assert original['transport_rc']==0 and original['remote_exec_acknowledged'] and out['passed']
claim=original['claim'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
expected=dict(claim=claim,request=original['request_id'],outcome=out,epoch=out['epoch'],deadline=out['deadline'],step=out['accounting'][0][0],task=task)
remote=('EXPECTED='+repr(expected)+'\n').encode()+b'''
from pathlib import Path
import datetime,hashlib,json,socket,stat,subprocess,sys
task=EXPECTED['task'];W=Path(task['original_namespace']);label=W.name;pins={}
def read(name):
 p=W/name;s=p.stat();assert not p.is_symlink() and 0<s.st_size<=8*1024**2;raw=p.read_bytes();after=p.stat();assert (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)
 pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return raw
raw=read('outcome.json');out=json.loads(raw);assert out==EXPECTED['outcome'] and out['passed'] and not out['forced_cleanup'] and out['owned_srun_reaped']
intent=json.loads(read('intent.json'));assert intent['claim']==EXPECTED['claim'] and intent['request_id']==EXPECTED['request'] and intent['epoch']==out['epoch']==EXPECTED['epoch'] and intent['deadline']==out['deadline']==EXPECTED['deadline']==out['epoch']+300 and intent['work_deadline']==out['epoch']+220 and intent['GPU_cards']==0
assert len(intent['actual_nodes'])==len(set(intent['actual_nodes']))==40 and intent['actual_parent_resources']['JobState']=='RUNNING'
for name in task['remote_files']:
 value=read('source/'+name);assert hashlib.sha256(value).hexdigest()==task['source_SHA256'][name]
mf=json.loads((W/'source/SOURCE_FILES.json').read_bytes());assert set(mf['files'])<=set(task['remote_files']) and mf['framework_commit']==task['framework_commit']
for name,pin in mf['files'].items():assert (W/'source'/name).stat().st_size==pin['bytes'] and hashlib.sha256((W/'source'/name).read_bytes()).hexdigest()==pin['SHA256']
fm=json.loads((W/'source/FRAMEWORK_FILES.json').read_bytes());assert fm['framework_commit']==mf['framework_commit'] and len(fm['files'])==353
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert out['accounting_columns']==columns
rows=[s.split('|') for s in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if '|'+label+'|' in s]
assert rows==out['accounting'] and len(rows)==1 and rows[0][0]==EXPECTED['step'] and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]=='f11r4n10'
active=subprocess.check_output(['squeue','--steps','-h','-j','1087656','-o','%i'],text=True,timeout=15).splitlines();assert rows[0][0] not in active
assert hashlib.sha256(read('CPU.log')).hexdigest()==out['log_SHA256']
result_raw=read('evidence/CPU-result.json');result=json.loads(result_raw);record=out['CPU-result.json'];assert len(result_raw)==record['bytes'] and hashlib.sha256(result_raw).hexdigest()==record['SHA256'] and result==record['value']
assert result['passed'] and result['tests_run']==26 and result['new_boundary_hook_tests']==8 and result['affected_resume_tests']==18 and result['new_resume_integration_tests']==2
assert result['source_files_verified']==353 and result['remote_source_files_verified']==len(mf['files']) and result['source_manifest_SHA256']==task['source_SHA256']['SOURCE_FILES.json'] and result['framework_manifest_SHA256']==task['source_SHA256']['FRAMEWORK_FILES.json']
assert result['CUDA_initialized'] is False and result['GPU_cards']==0 and result['raw_tensor_transfer_bytes']==0 and not result['original_model_loaded'] and not result['original_checkpoint_loaded'] and not result['native_GPU_restore_executed']
assert result['owner']['step']==rows[0][0] and result['owner']['node']==rows[0][5] and result['epoch']==out['epoch'] and result['deadline']==out['epoch']+220 and 0<result['wall_seconds']<220 and out['wall_seconds']<300
traw=read('evidence/tests.txt');assert hashlib.sha256(traw).hexdigest()==result['tests_SHA256'];tests=traw.decode();ok=[r for r in tests.splitlines() if r.startswith('test_') and r.endswith(' ... ok')]
assert 'Ran 26 tests' in tests and tests.rstrip().endswith('OK') and len(ok)==26 and sum('BoundaryHookTest.' in r for r in ok)==8 and sum('ResumeWitnessTest.' in r for r in ok)==18
assert result['plugin_registered_and_base_constructor_verified'] and not result['recovery_qualified'] and not result['Model_GO'] and result['HIP']
deps=json.loads((W/'source/DEPENDENCIES.json').read_bytes());assert result['actual_runtime_providers']==deps['providers']
assert {k:result[k] for k in deps['versions']}==deps['versions']
for key,pin in result['actual_runtime_providers'].items():
 path=pin['path'];mapped=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
 if path.startswith('/minicpm-transformers/'):actual=mapped/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site'/path[len('/minicpm-transformers/'):]
 elif path.startswith('/swift-runtime/'):actual=mapped/'results/training-vlm-swift-offline-runtime-v1'/path[len('/swift-runtime/'):]
 elif path.startswith('/resume-framework/'):actual=W/'framework'/path[len('/resume-framework/'):]
 else:actual=mapped/'containers/dtk26-sandbox-r1'/path.lstrip('/')
 assert actual.stat().st_size==pin['bytes'] and hashlib.sha256(actual.read_bytes()).hexdigest()==pin['SHA256']
owner=json.loads(read('owned-srun.json'));host=socket.gethostname();host_matches=host==owner['host'];match=None
if host_matches:
 path=Path('/proc',str(owner['PID']),'stat');match=path.exists() and int(path.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not match
assert 'torch' not in sys.modules
print(json.dumps(dict(passed=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),claim=EXPECTED['claim'],request_id=EXPECTED['request'],original_artifacts=pins,source_files_verified=len(task['remote_files']),full_framework_source_files_verified=353,framework_commit=fm['framework_commit'],framework_manifest_SHA256=task['source_SHA256']['FRAMEWORK_FILES.json'],accounting_columns=columns,accounting=rows,original_step_not_active=True,epoch=out['epoch'],deadline=out['deadline'],forced_cleanup=False,owned_srun_reaped=True,owned_srun_HOST_matches=host_matches,current_HOST=host,original_owner_HOST=owner['host'],owned_srun_startticks_still_present=match,owner_identity_independently_verified=host_matches and match is False,actual_parent_resources=intent['actual_parent_resources'],actual_parent_node_names=intent['actual_nodes'],result=result,whole_wall_seconds=out['wall_seconds'],reader_Torch_imported=False,new_Slurm_steps=0,actual_model_loaded=False,actual_native_boundary_traces=0,qualification='Exact new CPU26 terminal/source/log/accounting/dependency bytes verified;ownerPID independence is explicit by hostmatch field. No actual MiniCPM firstdivergence,numericalrecovery,quality orperformance qualification.')))
'''
with (D/'root-reader-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=original['request_id'],payload_SHA256=hashlib.sha256(remote).hexdigest(),read_only=True,tests_replayed=False),f,indent=2)
ns={'__file__':str(D/'reuse.py')};s=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(s.split('REMOTE=',1)[0],'existing_ordinary_SSH','exec'),ns)
r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=90)
for n,raw in [('root-reader.stdout.private.json',r.stdout),('root-reader.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(raw)
v=dict(transport_rc=r.returncode,script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),payload_SHA256=hashlib.sha256(remote).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:v['value']=json.loads(r.stdout)
with (D/'root-reader-result.json').open('x') as f:json.dump(v,f,indent=2)
value=v.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=value.get('passed'),actual_tests=value.get('result',{}).get('tests_run'),framework_files=value.get('full_framework_source_files_verified'),owner_identity_independently_verified=value.get('owner_identity_independently_verified'),accounting=value.get('accounting'),GPU_cards=0)))
