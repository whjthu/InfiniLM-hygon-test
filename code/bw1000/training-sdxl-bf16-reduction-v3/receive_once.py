"""Install and start one finite, owned per-job controller; never renew/replay it."""
import base64,hashlib,json,os,socket,subprocess,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-sdxl-bf16-reduction-v3';label=W.name
# Original epoch begins before installation and survives every phase.
epoch=time.time();deadline=epoch+1200
assert not W.exists()
rows=subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State'],text=True,timeout=15)
assert not any('|'+label+suffix+'|' in rows for suffix in ('','-policy','-pre','-post','-failure-post','-compare'))
parent=subprocess.check_output(['squeue','-h','-j','1087656','-o','%T|%D|%N'],text=True,timeout=15).strip().split('|');assert parent[:2]==['RUNNING','40']
assert 'f11r4n11' in subprocess.check_output(['scontrol','show','hostnames',parent[2]],text=True,timeout=15).splitlines()
W.mkdir(mode=0o700);(W/'src').mkdir(mode=0o700)
for name,item in FILES.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 raw=base64.b64decode(item['b64'],validate=True);assert len(raw)<200000 and hashlib.sha256(raw).hexdigest()==item['SHA256']
 target=W/'src'/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(raw)
manifest=json.loads((W/'src/SOURCE_FILES.json').read_bytes());assert set(manifest['files'])==set(FILES)-{'SOURCE_FILES.json'}
for name,pin in manifest['files'].items():assert pin['SHA256']==FILES[name]['SHA256'] and (W/'src'/name).stat().st_size==pin['bytes']
prior=R/'results/training-sdxl-bf16-reduction-v2';proof=json.loads((prior/'intent.json').read_bytes());assert proof['claim_commit']=='07d9c8a4367c0908102d9d21c9498a46ecd70b3b' and proof['request_id']=='59385fc46064432cbb703e61dcbf6746'
raw=(prior/'host-guard-tests.log').read_bytes();assert len(raw)<20000 and b'Ran 6 tests' in raw and raw.rstrip().endswith(b'OK')
for name in ('source/tools/hygon_sdxl_assigned_kfd.py','source/tests/test_sdxl_assigned_kfd.py'):
 assert hashlib.sha256((prior/'src'/name).read_bytes()).hexdigest()==FILES[name]['SHA256']
with (W/'reused-host-guard-tests.json').open('x') as f:json.dump(dict(original_root=str(prior),tests=6,passed=True,reexecuted=False,log_SHA256=hashlib.sha256(raw).hexdigest()),f,indent=2)
with (W/'intent.json').open('x') as f:json.dump(dict(claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,source_manifest_SHA256=FILES['SOURCE_FILES.json']['SHA256'],files={n:x['SHA256'] for n,x in FILES.items()},installer_host=socket.gethostname(),installer_boot_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),resource=dict(parent='1087656',node='f11r4n11',GPU_cards=1,CPU=8,memory_MiB=64000),formal_performance=False),f,indent=2)
with (W/'keeper.log').open('xb') as log:
 p=subprocess.Popen(['/usr/local/python3.12/bin/python3','-B','-u',str(W/'src/controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()
 owner=dict(PID=p.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),epoch=epoch,deadline=deadline,request_id=REQUEST,claim_commit=CLAIM)
 with (W/'keeper-owner.json').open('x') as f:json.dump(owner,f,indent=2)
 until=time.monotonic()+4
 while not (W/'controller-ready.json').exists() and p.poll() is None and time.monotonic()<until:time.sleep(.1)
 ready=json.loads((W/'controller-ready.json').read_bytes()) if (W/'controller-ready.json').exists() else None
 if ready:assert ready['PID']==owner['PID'] and ready['startticks']==owner['startticks'] and ready['epoch']==epoch
 print(json.dumps(dict(claim_commit=CLAIM,request_id=REQUEST,owned_controller=owner,controller_ready=ready,controller_returncode=p.poll(),GPU_execution_status='PENDING_ACTUAL_CONTROLLER_ADMISSION',new_SSH_service=False,new_poller_service=False,output=str(W))),flush=True)
