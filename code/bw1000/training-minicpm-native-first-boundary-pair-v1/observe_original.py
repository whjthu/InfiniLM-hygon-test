from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).parent;B=D.parent;launch=json.loads((D/'launch-intent.json').read_bytes());transport=json.loads((D/'transport-outcome.json').read_bytes());owner=transport['value']['owned_controller']
remote=r'''from pathlib import Path
import datetime,hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native-first-boundary-pair-v1');label=W.name;intent=json.loads((W/'intent.json').read_bytes());assert intent['claim_commit']==CLAIM and intent['request_id']==REQUEST
host_matches=socket.gethostname()==OWNER['host'];alive=None
if host_matches:
 p=Path('/proc',str(OWNER['PID']),'stat');alive=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==OWNER['startticks'] and p.read_text().split(') ',1)[1].split()[0]!='Z'
rows=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+suffix+'|' in line for suffix in ('','-pre','-post','-audit'))]
progress=[]
for arm in ('control','restored'):
 for rank in range(8):
  p=W/arm/'train/resume-witness'/('rank-%02d'%rank)/'progress.json'
  if p.is_file():
   v=json.loads(p.read_bytes());progress.append(dict(arm=arm,rank=rank,inputs=v['inputs'],outputs=v['outputs'],completed_steps=v['completed_steps'],failed=v['failed'],RNG_loads=v['RNG_loads'],saved_checkpoints=v['saved_checkpoints']))
boundary_progress=[]
for arm in ('control','restored'):
 for rank in range(8):
  folder=W/arm/'train/resume-witness'/('rank-%02d'%rank);p=folder/'forward-boundaries-progress.json';row=dict(arm=arm,rank=rank,planned=52,progress_present=p.is_file(),final_present=(folder/'forward-boundaries.json').is_file())
  if p.is_file():
   v=json.loads(p.read_bytes());row.update(completed=v.get('completed_boundaries'),failed=v.get('failed'),closed=v.get('closed'))
  boundary_progress.append(row)
logs={}
for name in ('keeper.log','control/native.log','restored/native.log'):
 p=W/name
 if p.exists():logs[name]=dict(bytes=p.stat().st_size)
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_id=REQUEST,epoch=intent['epoch'],deadline=intent['deadline'],same_HOST_as_keeper=host_matches,owned_keeper_alive=alive,outcome_present=(W/'outcome.json').is_file(),accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'],accounting=rows,assigned_present=(W/'assigned.json').is_file(),preflight_present=(W/'source-and-asset-preflight.json').is_file(),progress=progress,boundary_progress=boundary_progress,planned_boundary_records=832,log_sizes=logs)))
'''
ns={'__file__':str(D/'reuse.py')};pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
r=subprocess.run(ns['ssh'],input=('CLAIM='+repr(launch['claim_commit'])+'\nREQUEST='+repr(launch['request_id'])+'\nOWNER='+repr(owner)+'\n'+remote).encode(),capture_output=True,timeout=30)
if r.returncode:print(json.dumps(dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())));raise SystemExit(1)
v=json.loads(r.stdout);folder=D/'observations';folder.mkdir(exist_ok=True)
with (folder/(str(time.time_ns())+'.json')).open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))