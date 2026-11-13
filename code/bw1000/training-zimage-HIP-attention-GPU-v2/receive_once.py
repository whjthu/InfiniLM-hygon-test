"""Install one published finite job controller;never replay unknown execution."""
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-zimage-HIP-attention-GPU-v2';label=W.name;epoch=time.time();deadline=epoch+660
assert not W.exists()
def query(args):return subprocess.check_output(args,text=True,timeout=min(15,epoch+100-time.time())).strip()
rows=query(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State'])
assert not any('|'+label+suffix+'|' in rows for suffix in ('','-pre','-post','-failure-post'))
p=query(['squeue','-h','-j','1087656','-o','%T|%D|%N']).split('|');assert p[:2]==['RUNNING','40']
assert 'f11r4n13' in query(['scontrol','show','hostnames',p[2]]).splitlines()
for line in query(['squeue','--steps','-h','-j','1087656','-o','%i|%N']).splitlines():
 handle,nodes=line.split('|',1)
 if handle not in ('1087656.batch','1087656.extern') and nodes not in ('','(null)'):assert 'f11r4n13' not in query(['scontrol','show','hostnames',nodes]).splitlines(),'T07 node has a live step'
W.mkdir(mode=0o700);(W/'src').mkdir(mode=0o700)
for name,pin in FILES.items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 raw=base64.b64decode(pin['b64'],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];dest=W/'src'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(raw)
manifest=json.loads((W/'src/SOURCE_FILES.json').read_bytes());assert set(manifest['files'])==set(FILES)-{'SOURCE_FILES.json'}
for name,pin in manifest['files'].items():assert pin['SHA256']==FILES[name]['SHA256'] and pin['bytes']==FILES[name]['bytes']
reuse=json.loads((W/'src/REUSED_SOURCE.json').read_bytes());prior=Path(reuse['root']);assert hashlib.sha256((prior/'outcome.json').read_bytes()).hexdigest()==reuse['outcome_SHA256']
o=json.loads((prior/'outcome.json').read_bytes());assert o['passed'] and o['accounting'][0][0]==reuse['CPU188_step'] and o['request_id']==reuse['CPU188_request']
qualified=json.loads((W/'src/LIBRARY_BINDING.json').read_bytes());qroot=Path(qualified['rocm_map_original_namespace']);qraw=(qroot/'outcome.json').read_bytes();assert len(qraw)==qualified['rocm_map_original_outcome_pin']['bytes'] and hashlib.sha256(qraw).hexdigest()==qualified['rocm_map_original_outcome_pin']['SHA256'];qo=json.loads(qraw);assert qo['passed'] and qo['accounting'][0][0]==qualified['rocm_map_original_step'] and qo['accounting'][0][2:4]==['COMPLETED','0:0']
qc=qo['CPU-result.json'];assert qc['bytes']==qualified['rocm_map_original_CPU_result_pin']['bytes'] and qc['SHA256']==qualified['rocm_map_original_CPU_result_pin']['SHA256'];qv=qc['value'];assert qv['passed'] and not qv['CUDA_initialized'] and qv['observed_names']==qualified['resolved_names'] and qv['loader_environment_pin']==qualified['loader_environment_pin']
for name,row in qv['observed_libraries'].items():
 p=qualified['resolved_libraries'][row['realpath']];assert p['bytes']==row['bytes'] and p['SHA256']==row['SHA256']
raw=(prior/'source/FRAMEWORK_FILES.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==reuse['framework_manifest_SHA256'];fm=json.loads(raw);assert fm==json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes()) and len(fm['files'])==382
(W/'framework').mkdir()
for name,pin in fm['files'].items():
 src=prior/'framework'/name;raw=src.read_bytes();assert src.is_file() and not src.is_symlink() and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];dest=W/'framework'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(raw)
 dest.chmod(pin['mode'])
dep=Path(reuse['dependency_root']);raw=(dep/'dependency-extracted-files.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==reuse['dependency_manifest_SHA256'];mapping=json.loads(raw);assert len(mapping['files'])==2852
for name,pin in mapping['files'].items():
 src=dep/'dependencies/site'/name;assert src.is_file() and not src.is_symlink() and src.stat().st_size==pin['bytes'] and hashlib.sha256(src.read_bytes()).hexdigest()==pin['SHA256']
assert time.time()<epoch+100
with (W/'intent.json').open('x') as f:json.dump(dict(claim_commit=CLAIM,request_id=REQUEST,epoch=epoch,deadline=deadline,source_manifest_SHA256=FILES['SOURCE_FILES.json']['SHA256'],installer_host=socket.gethostname(),resource=dict(parent='1087656',node='f11r4n13',GPU_cards=1,CPU=8,memory_MiB=16384),formal_performance=False),f,indent=2)
with (W/'keeper.log').open('xb') as log:
 child=subprocess.Popen(['/usr/local/python3.12/bin/python3','-B','-u',str(W/'src/controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split();owner=dict(PID=child.pid,PGID=int(stat[2]),startticks=int(stat[19]),host=socket.gethostname(),boot_id_SHA256=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),epoch=epoch,deadline=deadline,request_id=REQUEST,claim_commit=CLAIM)
 with (W/'keeper-owner.json').open('x') as f:json.dump(owner,f,indent=2)
 until=time.monotonic()+4
 while not (W/'controller-ready.json').exists() and child.poll() is None and time.monotonic()<until:time.sleep(.1)
 ready=json.loads((W/'controller-ready.json').read_bytes()) if (W/'controller-ready.json').exists() else None
 if ready:assert ready['PID']==owner['PID'] and ready['startticks']==owner['startticks'] and ready['epoch']==epoch
 print(json.dumps(dict(claim=CLAIM,request_id=REQUEST,owned_controller=owner,controller_ready=ready,controller_returncode=child.poll(),new_SSH_service=False,new_poller_service=False)),flush=True)
