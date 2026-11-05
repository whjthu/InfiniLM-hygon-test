from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,os,re,socket,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-decode-chain-reference/gpu-chain-v2'
OUTER=R/'incoming/Deep-C1-gap-enclosure-r467/profile-source-v1/CPU_outer.sh'
def pin(p):return sha256(p.read_bytes()).hexdigest()
def root(path):return R/Path(path).relative_to('/refactor')
def now():return datetime.now(timezone.utc).isoformat()
c=json.loads(files['GPU_RESOURCE_CLAIM.json'])
claim_sha=sha256(files['GPU_RESOURCE_CLAIM.json']).hexdigest()
assert claim_sha=='874cc709c171e23ecb9e9d8182bf4869aade579781e2a12fcb2fed9ef131c841'
assert c['GPU_GO'] and c['once'] and not c['retry'] and not c['Model_GO']
assert (c['parent_job'],c['node'],c['cards'],c['cores'],c['hard_seconds'])==('910376','a02r4n12',8,32,120)
parent=c['parent_job'];node=c['node'];label=c['label'];out=root(c['output_dir'])
assert set(files)==set(c['files_SHA256'])|{'GPU_RESOURCE_CLAIM.json'}
assert all(sha256(files[n]).hexdigest()==h for n,h in c['files_SHA256'].items())
assert pin(OUTER)==c['outer_sha256'] and pin(root(c['loader_receipt_path']))==c['loader_receipt_sha256']
assert pin(root(c['native_path']))==c['native_sha256']
assert pin(root(c['runtime_identity_CPU_receipt_path']))==c['runtime_identity_CPU_receipt_SHA256']
assert not D.exists() and not out.exists() and not list((R/'logs').glob(label+'-*'))
def capture(*args):
 return subprocess.check_output(args,text=True,timeout=12)
def nodes(spec):
 return capture('scontrol','show','hostnames',spec).splitlines()
def account():
 return [s.split('|') for s in capture('sacct','-S','2026-11-05','-j',parent,'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,ElapsedRaw,NodeList%100').splitlines()]
job=capture('scontrol','show','job',parent,'-o')
assert 'JobState=RUNNING' in job and 'UserId=aclsylqidf(' in job
assert node in nodes(re.search(r'(?:^| )NodeList=([^ ]+)',job).group(1))
a=account()
assert [x[2:4] for x in a if x[0]==c['CPU_loader_step']]==[['COMPLETED','0:0']]
assert [x[2:4] for x in a if x[0]==c['CPU_runtime_identity_step']]==[['COMPLETED','0:0']]
assert [x[2:4] for x in a if x[0]==c['CPU_linked_producer_step']]==[['FAILED','1:0']]
for j,n,state,code,elapsed,hosts in a:
 assert not n.startswith(label)
 if re.fullmatch(parent+r'[.]\d+',j) and state in ('RUNNING','PENDING','CONFIGURING','COMPLETING'):
  assert node not in nodes(hosts)
D.mkdir(parents=True)
for n,b in files.items():
 assert Path(n).name==n
 with (D/n).open('xb') as f:f.write(b)
 assert (D/n).read_bytes()==b
subprocess.run(['bash','-n',str(D/'inside.sh')],check=True,timeout=8)
out.mkdir()
def save(name,obj):
 with (out/name).open('x') as f:json.dump(obj,f,indent=2)
r=dict(HOST=socket.gethostname(),UTC=now(),publication=publication,claim_sha256=claim_sha,label=label,parent=parent,node=node,GPU_submitted=False)
save('intent.json',r)
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_')) and k not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')}
common=['srun','--jobid='+parent,'--overlap','--exact','--immediate=12','--cpu-bind=none','-N1','-n1','--ntasks-per-node=1','-w',node]
cpu_env={**env,'CUDA_VISIBLE_DEVICES':'-1','HIP_VISIBLE_DEVICES':'-1','ROCR_VISIBLE_DEVICES':'-1'}
def idle(phase):
 args=common+['-c1','--gres=none','--mem=512M','-t','00:01:00','--job-name='+label+'-'+phase,'python3','-B',str(D/'idle.py')]
 q=subprocess.run(args,env=cpu_env,capture_output=True,text=True,timeout=c['aux_seconds_each'])
 value=json.loads(q.stdout)
 save(phase+'.json',dict(rc=q.returncode,**value))
 assert q.returncode==0 and not value['kfd_pids'] and not value['worker_pids']
 return value
phase='before'
try:
 before=idle('before')
 r['before']=before
 phase='gpu'
 cmd=common+['--kill-on-bad-exit=1','-c32','--gres=dcu:8','--mem=32768M','-t','00:02:00','--job-name='+label,'--export=ALL','--output='+str(R/'logs'/(label+'-%J.log')),'bash',str(OUTER),'probe']
 ge={**env,'WAVE_INSIDE_R2323':'Deep-decode-chain-reference/gpu-chain-v2/inside.sh','DEEP_RAW_CHAIN_CLAIM_SHA':claim_sha}
 with (out/'client.log').open('x') as f:
  p=subprocess.Popen(cmd,env=ge,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 stat=Path('/proc/%d/stat'%p.pid)
 r.update(GPU_submitted=True,srun_PID=p.pid,startticks=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None)
 save('launch.json',r)
 r['gpu_rc']=p.wait(timeout=135)
 phase='post'
 post=idle('post')
 r['post']=post
 assert post['pci']==before['pci']
 phase='terminal'
 a=[x for x in account() if x[1].startswith(label)]
 r['accounting']=a
 p,=[x for x in a if x[1]==label]
 r['actual_GPU']=p[0]
 assert r['gpu_rc']==0 and p[2:4]==['COMPLETED','0:0']
 packet=out/('decode-ccl-'+p[0]+'-node0.json')
 b=packet.read_bytes()
 assert len(b)<4*1024*1024
 import sys
 sys.dont_write_bytecode=True
 sys.path.insert(0,str(D))
 from summarize import summarize
 report=summarize([json.loads(b)],8,p[0],c['native_sha256'],deep_chain=True)
 report['independent_cleanup_verified']=True
 report['node_packet_sha256']=sha256(b).hexdigest()
 save('summary.json',report)
 r.update(passed=True,summary_sha256=pin(out/'summary.json'),measurements=[dict(past=x['past'],arm=x['arm'],fences=x['timed_fences'],wall_median_ms=x['clocks']['wall_ms']['median_ms'],stream_median_ms=x['clocks']['stream_elapsed_ms']['median_ms'],host_enqueue_median_ms=x['clocks']['enqueue_phase_wall_ms']['median_ms']) for x in report['rows']])
except Exception as exc:
 r.update(passed=False,phase=phase,error=type(exc).__name__,error_detail=str(exc)[-300:],terminal_unknown=isinstance(exc,subprocess.TimeoutExpired))
 if phase in ('terminal','post'):
  r['accounting']=[x for x in account() if x[1].startswith(label)]
logdir=R/'logs'
for f in sorted(logdir.glob(label+'-*.log'))[:2]:
 b=f.read_bytes()
 r.setdefault('log_pins',{})[f.name]=dict(bytes=len(b),sha256=sha256(b).hexdigest(),tail=b[-1100:].decode(errors='replace').splitlines()[-7:] if not r['passed'] else [])
save('result.json',r)
print(json.dumps(r))