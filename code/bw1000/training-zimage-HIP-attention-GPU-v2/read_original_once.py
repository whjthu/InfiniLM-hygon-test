"""Read only this exact controller/request; independently consume its actual terminal."""
from pathlib import Path
import ast,datetime,hashlib,json,re,subprocess,sys,time
D=Path(__file__).resolve().parent;B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
tag=sys.argv[1];assert re.fullmatch('[a-z0-9-]{1,30}',tag)
i=json.loads((D/'launch-intent.json').read_bytes());claim=i['claim'];request=i['request_id']
plan=json.loads((D/'SOURCE_FILES.json').read_bytes());expected={n:v['SHA256'] for n,v in plan['files'].items()};expected['SOURCE_FILES.json']=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest()
remote=r'''
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-HIP-attention-GPU-v2')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(4*1024**2),b''):h.update(raw)
 return h.hexdigest()
def load(name):
 p=W/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<1024**2
 raw=p.read_bytes();return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def query(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);assert r.returncode==0 and len(r.stdout)<2*1024**2;return r.stdout
intent,ip=load('intent.json');owner,op=load('keeper-owner.json');assert intent['claim_commit']==owner['claim_commit']==CLAIM and intent['request_id']==owner['request_id']==REQUEST
same_host=owner['host']==socket.gethostname();alive=None
if same_host:
 p=Path('/proc',str(owner['PID']),'stat');alive=False
 if p.exists():
  stat=p.read_text().split(') ',1)[1].split();alive=int(stat[19])==owner['startticks'] and stat[0]!='Z'
if not (W/'controller-result.json').exists():
 print(json.dumps(dict(state='ORIGINAL_CONTROLLER_NOT_TERMINAL',request_id=REQUEST,claim=CLAIM,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_epoch=intent['epoch'],original_deadline=intent['deadline'],original_controller_alive=alive,same_login_host=same_host,intent_pin=ip,owner_pin=op,actual_step_listing=query(['squeue','--steps','-h','-j','1087656','-o','%i|%j|%T|%N']),replay_allowed=False)));raise SystemExit(0)
result,rp=load('controller-result.json');assert result['request_id']==REQUEST and result['claim_commit']==CLAIM and result['epoch']==intent['epoch'] and result['deadline']==intent['deadline']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert result['accounting_columns']==columns
accounting=[];live=query(['squeue','--steps','-h','-j','1087656','-o','%i']).splitlines()
for run in result['runs']:
 assert run['accounting_columns']==columns
 row=run['terminal']['row'];assert row and row[0].startswith('1087656.') and row[1]==run['name'] and row[5]=='f11r4n13'
 rows=[x.split('|') for x in query(['sacct','-n','-P','-j',row[0],'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if x.split('|')[0]==row[0]]
 assert rows==[row] and row[2].split()[0] in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'] and row[0] not in live
 assert sha(W/(run['name']+'.log'))==run['log_SHA256']
 accounting.append(row)
for n,pin in EXPECTED.items():assert sha(W/'src'/n)==pin
fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==382
for n,pin in fm['files'].items():
 p=W/'framework'/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
post=None;assigned=None;artifact_count=0;artifact_bytes=0
if result['post_clean']:
 before,bp=load('before.json');after,ap=load('after.json');assert before['pci']==after['pci'] and before['boot_id']==after['boot_id'] and len(set(before['pci']))==8 and before['kfd_holder_count']==after['kfd_holder_count']==0
 post=dict(before=before,after=after,before_pin=bp,after_pin=ap)
if (W/'assigned.json').exists():assigned,_=load('assigned.json')
d=result.get('diagnostic')
if d:
 raw=(W/'diagnostic/result.json').read_bytes();assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['SHA256'] and json.loads(raw)==d['value'];v=d['value']
 assert len(v['plan'])==12 and v['actual_native_model_loads']==v['actual_native_optimizer_updates']==0 and not v['Model_GO']
 for n,pin in v['artifacts'].items():
  assert Path(n).name==n and n.endswith('.private.pt');p=W/'diagnostic'/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];artifact_count+=1;artifact_bytes+=pin['bytes']
 assert artifact_count<=18 and artifact_bytes<=512*1024**2
 if v['passed']:
  assert len(v['cases'])==3 and all(x['complete'] and len(x['outcomes'])==4 and all(y['passed'] for y in x['outcomes']) for x in v['cases'])
  assert v['counters']==dict(forward_attempted=12,forward_returned=12,backward_attempted=12,backward_synchronized=12)
  assert v['input_audits']['completed']==12 and v['input_audits']['buffers_checked']==48
  for case in v['cases']:
   assert all(m['max_abs']<=.001 and m['relative_l2']<=.02 for y in case['outcomes'] for m in y['CPU_F64_metrics'])
   assert all(y['repeat_exact'] is True for y in case['outcomes'][2:])
  assert result['SDK_numerical_passed'] and result['post_clean'] and assigned['pci'] in post['before']['pci'] and assigned['visible_cards']==1
preflight_metadata_pins={}
for name in ['link-ldd.private.txt','link-observed-names.json','link-preflight.json']:
 p=W/name
 if p.exists():
  assert p.is_file() and not p.is_symlink() and p.stat().st_size<131072;raw=p.read_bytes();preflight_metadata_pins[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
if 'link-preflight.json' in preflight_metadata_pins:
 link=json.loads((W/'link-preflight.json').read_bytes());assert link['passed'] and link['epoch']==intent['epoch'] and not link['CUDA_initialized']
 assert preflight_metadata_pins['link-ldd.private.txt']['SHA256']==link['passive_ldd_SHA256'] and preflight_metadata_pins['link-observed-names.json']['SHA256']==link['observed_names_SHA256']
failure=None
for name in ['diagnostic/failure.private.txt','keeper.log']:
 p=W/name
 if p.exists() and p.stat().st_size<128*1024:
  raw=p.read_bytes()
  if name.endswith('failure.private.txt') or not result['completed']:failure=dict(path=name,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),tail=raw.decode(errors='replace').splitlines()[-6:])
print(json.dumps(dict(state='ORIGINAL_TERMINAL_INDEPENDENTLY_CONSUMED',independent_reader_passed=True,request_id=REQUEST,claim=CLAIM,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_accounting=accounting,intent_pin=ip,controller_result_pin=rp,original_controller_alive=alive,same_login_host=same_host,framework_files_verified=382,source_files_verified=len(EXPECTED),private_controlled_tensor_artifacts_verified=artifact_count,private_controlled_tensor_bytes_verified=artifact_bytes,original_result=result,assigned=assigned,post=post,failure_evidence=failure,preflight_metadata_pins=preflight_metadata_pins,GPU_replayed=False,Model_GO=False)))
'''
payload=('REQUEST='+repr(request)+'\nCLAIM='+repr(claim)+'\nEXPECTED='+repr(expected)+'\n').encode()+remote.encode()
tree=ast.parse((B/'probe_ssh_transport_r9543.py').read_bytes());a={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)};items=a['a'].elts;known=Path(ast.literal_eval(a['r'].args[0]))/'known_hosts'
ssh=['ssh.exe','-T','-i',str(Path.home()/'.ssh/hygon-261111.txt'),'-p',ast.literal_eval(items[3]),'-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','BatchMode=yes','-o','ConnectTimeout=10',ast.literal_eval(items[16]),'/usr/local/python3.12/bin/python3 -B -u -']
try:r=subprocess.run(ssh,input=payload,capture_output=True,timeout=60);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b'';error='TimeoutExpired'
for n,raw in [('stdout.private.json',stdout),('stderr.private.txt',stderr)]:
 with (D/(tag+'-reader-'+n)).open('xb') as f:f.write(raw)
out=dict(transport_rc=rc,error_type=error,stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest())
if rc==0:out['value']=json.loads(stdout)
with (D/(tag+'-reader-result.json')).open('x') as f:json.dump(out,f,indent=2)
v=out.get('value',{});o=v.get('original_result',{});print(json.dumps(dict(transport_rc=rc,state=v.get('state'),independent_reader_passed=v.get('independent_reader_passed'),actual_accounting=v.get('actual_accounting'),original_controller_alive=v.get('original_controller_alive'),completed=o.get('completed'),SDK_numerical_passed=o.get('SDK_numerical_passed'),post_clean=o.get('post_clean'),Model_GO=False)))
