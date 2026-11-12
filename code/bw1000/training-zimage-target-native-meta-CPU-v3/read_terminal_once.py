"""Independently consume only this request's bounded metadata, source and accounting."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
D=Path(__file__).resolve().parent;B=D.parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
tag=sys.argv[1]
assert tag.isalnum()
intent=json.loads((D/'launch-intent.json').read_bytes());claim=intent['claim'];request=intent['request_id']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
expected={n:task['source_SHA256'][n] for n in task['remote_files']}
program=r'''
from pathlib import Path
import datetime,hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-target-native-meta-CPU-v3')
def load(n):
 p=W/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size<512*1024
 raw=p.read_bytes();return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
i,ip=load('intent.json');o,op=load('outcome.json');s,sp=load('owned-srun.json');c,cp=load('evidence/CPU-result.json')
assert i['request_id']==o['request_id']==REQUEST and i['claim']==o['claim']==CLAIM
assert i['host']==s['host']
assert cp['SHA256']==o['CPU-result.json']['SHA256'] and cp['bytes']==o['CPU-result.json']['bytes']
assert o['CPU-result.json']['value']==c and c['owner']['node']=='f11r4n13'
for n,pin in EXPECTED.items():
 p=W/'source'/n;assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==pin
assert o['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
step=c['owner']['step'];assert step.startswith('1087656.') and all(x.isdecimal() for x in step.split('.'))
q=subprocess.run(['sacct','-n','-P','-j',step,'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],capture_output=True,text=True,timeout=15)
assert q.returncode==0
rows=[x.split('|') for x in q.stdout.splitlines() if x.split('|')[0]==step]
assert len(rows)==1 and rows[0]==o['accounting'][0] and rows[0][2].split()[0] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] and rows[0][5]=='f11r4n13'
live=subprocess.run(['squeue','--steps','-h','-j','1087656','-o','%i'],capture_output=True,text=True,timeout=15)
assert live.returncode==0 and step not in live.stdout.splitlines()
same_controller=i['host']==socket.gethostname();owned_alive=None
if same_controller:
 p=Path('/proc',str(s['PID']),'stat');owned_alive=False
 if p.exists():
  stat=p.read_text().split(') ',1)[1].split();owned_alive=int(stat[19])==s['startticks']
 assert owned_alive is False
assert o['owned_srun_reaped']
if o['passed']:assert not o['forced_cleanup'] and o['returncode']==0 and c['passed'] and rows[0][2:4]==['COMPLETED','0:0']
else:assert not c['passed'] or o['forced_cleanup'] or o['returncode']!=0
native=None;failure=None
if c['passed']:
 native,np=load('evidence/native-meta-result.json');assert np['SHA256']==c['native_result']['SHA256'] and np['bytes']==c['native_result']['bytes'] and native==c['native_result']['value']
 assert not c['CUDA_initialized'] and c['forbidden_file_attempts']==0
else:
 p=W/'evidence/failure.private.txt'
 if p.exists():
  assert p.is_file() and not p.is_symlink() and p.stat().st_size<65536
  raw=p.read_bytes();failure=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),tail=raw.decode(errors='replace').splitlines()[-5:])
assert c['GPU_cards']==0 and c['weights_or_data_read_bytes']==0
reuse=json.loads((W/'source/REUSED_SOURCE.json').read_bytes());previous=Path(reuse['root']);assert hashlib.sha256((previous/'outcome.json').read_bytes()).hexdigest()==reuse['outcome_SHA256']
fm=json.loads((W/'source/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==380
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']

dep,dp=load('dependency-extracted-files.json');assert len(dep['packages'])==6 and dp['SHA256']==reuse['dependency_manifest_SHA256']
dependency_root=Path(reuse['dependency_root']);assert hashlib.sha256((dependency_root/'outcome.json').read_bytes()).hexdigest()==reuse['dependency_outcome_SHA256']
for name,pin in dep['files'].items():
 p=dependency_root/'dependencies/site'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256']
print(json.dumps(dict(independent_reader_passed=True,request_id=REQUEST,claim=CLAIM,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_intent_pin=ip,original_outcome_pin=op,CPU_result_pin=cp,owned_srun_pin=sp,source_files_verified=len(EXPECTED),actual_accounting_columns=o['accounting_columns'],actual_accounting=rows,actual_step_live=False,owned_srun_alive=owned_alive,controller_PID_recheck='SAME_HOST_EXACT_STARTTICKS' if same_controller else 'LOGIN_HOST_DIFFERS_NOT_RECHECKED',framework_files_verified=380,dependency_files_verified=len(dep['files']),dependency_manifest_pin=dp,failure_evidence=failure,original_check_passed=o['passed'],original_outcome=o,Model_GO=False,GPU_GO=False)))
'''
payload=('REQUEST='+repr(request)+'\nCLAIM='+repr(claim)+'\nEXPECTED='+repr(expected)+'\n').encode()+program.encode()
tree=ast.parse((B/'probe_ssh_transport_r9543.py').read_bytes());assign={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
items=assign['a'].elts;known=Path(ast.literal_eval(assign['r'].args[0]))/'known_hosts'
argv=['ssh.exe','-T','-i',str(Path.home()/'.ssh/hygon-261111.txt'),'-p',ast.literal_eval(items[3]),'-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','BatchMode=yes','-o','ConnectTimeout=10',ast.literal_eval(items[16]),'/usr/local/python3.12/bin/python3 -B -u -']
r=subprocess.run(argv,input=payload,capture_output=True,timeout=45)
for n,raw in [('stdout.private.json',r.stdout),('stderr.private.txt',r.stderr)]:
 with (D/(tag+'-reader-'+n)).open('xb') as f:f.write(raw)
out=dict(transport_rc=r.returncode,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:out['value']=json.loads(r.stdout)
with (D/(tag+'-reader-result.json')).open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});c=v.get('original_outcome',{}).get('CPU-result.json',{}).get('value',{})
print(json.dumps(dict(transport_rc=r.returncode,independent_reader_passed=v.get('independent_reader_passed'),accounting=v.get('actual_accounting'),missing_required_imports=c.get('missing_required_imports'),diffsynth=c.get('diffsynth'),native_source_import_GO=c.get('native_source_import_GO'),CUDA_initialized=c.get('CUDA_initialized'),GPU_GO=False,Model_GO=False)))
