"""Install one published finite job controller;never replay unknown execution."""
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-zimage-added-HIP-attention-GPU-v1';label=W.name;epoch=time.time();deadline=epoch+660
assert not W.exists()
def query(args):return subprocess.check_output(args,text=True,timeout=min(15,epoch+100-time.time())).strip()
rows=query(['sacct','-n','-P','-j','1103606','--format=JobIDRaw,JobName%100,State'])
assert not any('|'+label+suffix+'|' in rows for suffix in ('','-pre','-post','-failure-post'))
p=query(['squeue','-h','-j','1103606','-o','%T|%D|%N']).split('|');assert p[:2]==['RUNNING','40']
nodes=query(['scontrol','show','hostnames',p[2]]).splitlines();assert len(nodes)==len(set(nodes))==40 and 'f13r2n07' in nodes
for line in query(['squeue','--steps','-h','-j','1103606','-o','%i|%N']).splitlines():
 handle,nodes=line.split('|',1)
 if handle not in ('1103606.batch','1103606.extern') and nodes not in ('','(null)'):assert 'f13r2n07' not in query(['scontrol','show','hostnames',nodes]).splitlines(),'T07 node has a live step'
W.mkdir(mode=0o700);(W/'src').mkdir(mode=0o700)
with (W/'intent.json').open('x') as f:json.dump(dict(claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,source_manifest_SHA256=FILES['SOURCE_FILES.json']['SHA256'],installer_host=socket.gethostname(),resource=dict(parent='1103606',node='f13r2n07',GPU_cards=1,CPU=8,memory_MiB=16384),formal_performance=False),f,indent=2)
print(json.dumps(dict(stage='REMOTE_INTENT_CREATED',claim=CLAIM,request_id=REQUEST,epoch=epoch)),flush=True)
for name,pin in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];dest=W/'src'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(raw)
manifest=json.loads((W/'src/SOURCE_FILES.json').read_bytes());assert set(manifest['files'])==set(FILES)-{'SOURCE_FILES.json'}
for name,pin in manifest['files'].items():assert pin['SHA256']==FILES[name]['SHA256'] and pin['bytes']==FILES[name]['bytes']
with (W/'keeper.log').open('xb') as log:
 child=subprocess.Popen(['/usr/local/python3.12/bin/python3','-B','-u',str(W/'src/controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();owner=dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),epoch=epoch,deadline=deadline,request_id=REQUEST,claim_commit=CLAIM)
 with (W/'keeper-owner.json').open('x') as f:json.dump(owner,f,indent=2)
 until=time.monotonic()+4
 while not (W/'controller-ready.json').exists() and child.poll() is None and time.monotonic()<until:time.sleep(.1)
 ready=json.loads((W/'controller-ready.json').read_bytes()) if (W/'controller-ready.json').exists() else None
 if ready:assert ready['PID']==owner['PID'] and ready['startticks']==owner['startticks'] and ready['epoch']==epoch
 print(json.dumps(dict(claim=CLAIM,request_id=REQUEST,owned_controller=owner,controller_ready=ready,controller_returncode=child.poll(),new_SSH_service=False,new_poller_service=False)),flush=True)
