from pathlib import Path
import hashlib,json,shlex,subprocess,time,uuid
D=Path(__file__).resolve().parents[1];B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];request=uuid.uuid4().hex;git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(n):return subprocess.check_output(git+['show',claim+':'+n])
t=json.loads(blob('coordination/tasks/training-sdxl-paired-input-stage-v1.json'));assert t['transfer_GO'] and not t['GPU_GO'];parts={}
for n,pin in t['source_SHA256'].items():
 raw=blob(t['source']+'/'+n);assert hashlib.sha256(raw).hexdigest()==pin;parts[n]=raw
plan=json.loads(parts['PLAN.json']);ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'ordinary_cluster_ssh','exec'),ns)
anchor=time.monotonic()
with (D/'paired-transfer-intent.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,source=plan['source_file'],target=plan['target_file'],bytes=plan['bytes'],SHA256=plan['SHA256'],exec_request_attempted_source=True,exec_request_attempted_target=False,replay_allowed=False),f,indent=2)
source_code='PLAN='+repr(plan)+'\n'+parts['send_exact_file.py'].decode();source_ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -c '+shlex.quote(source_code)];
try:s=subprocess.run(source_ssh,capture_output=True,timeout=45)
except subprocess.TimeoutExpired as e:
 with (D/'paired-transfer-source-timeout.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,source_observation_timeout=True,target_exec_request_attempted=False,received_bytes=len(e.stdout or b''),replay_allowed=False),f,indent=2)
 raise SystemExit('source transport observation timeout;target not invoked')
with (D/'paired-transfer-source.stderr.private.txt').open('xb') as f:f.write(s.stderr)
assert s.returncode==0 and len(s.stdout)==plan['bytes'] and hashlib.sha256(s.stdout).hexdigest()==plan['SHA256'],('source transport failed',s.returncode)
with (D/'paired-transfer-source-receipt.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,source_rc=s.returncode,bytes=len(s.stdout),SHA256=hashlib.sha256(s.stdout).hexdigest(),raw_source_in_transient_localRAM=True,local_raw_file_saved=False),f,indent=2)
receive_code='PLAN='+repr(plan)+'\nCLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\n'+parts['receive_exact_file.py'].decode();argv=ns['ssh'][:-1]+['/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(receive_code)]
with (D/'paired-transfer-target-request.json').open('x') as f:json.dump(dict(claim=claim,request_id=request,exec_request_attempted_target=True,bytes=plan['bytes'],target=plan['target_file']),f,indent=2)
try:r=subprocess.run(argv,input=s.stdout,capture_output=True,timeout=max(1,min(135,180-(time.monotonic()-anchor))));rc=r.returncode;out=r.stdout;err=r.stderr
except subprocess.TimeoutExpired as e:rc=None;out=e.stdout or b'';err=e.stderr or b''
for n,raw in [('paired-transfer-target.stdout.private.json',out),('paired-transfer-target.stderr.private.txt',err)]:
 with (D/n).open('xb') as f:f.write(raw)
value=json.loads(out) if rc==0 else None;record=dict(claim=claim,request_id=request,source_rc=0,target_rc=rc,value=value,remote_outcome='UNKNOWN' if rc in (None,255) else 'ORIGINAL_RETURNED',replay_allowed=False,raw_bytes_exported=plan['bytes'],destination=plan['target_file'])
with (D/'paired-transfer-outcome.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(dict(source_rc=0,target_rc=rc,passed=value.get('passed') if value else None,bytes=plan['bytes'],target_scope='privateHygon project results',new_Slurm=0,GPU_cards=0)))
