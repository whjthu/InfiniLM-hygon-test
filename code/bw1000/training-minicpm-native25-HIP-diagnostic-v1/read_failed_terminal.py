from pathlib import Path
import hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native25-HIP-diagnostic-v1');label=W.name
raw=(W/'outcome.json').read_bytes();out=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==EXPECTED['files']['outcome.json']['SHA256'] and out['keeper_passed'] is False
intent=json.loads((W/'intent.json').read_bytes());assert intent['claim_commit']=='10e6f7a80fe56419d09260f543de7a22c89e7f2b' and intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']==out['epoch']+7800
mf_raw=(W/'src/SOURCE_FILES.json').read_bytes();assert hashlib.sha256(mf_raw).hexdigest()==intent['source_manifest_SHA256']=='dd33c8bfc0322a2dfeb6832a963c2cda35aa773593132977b4048ae8e68a5477'
mf=json.loads(mf_raw);assert len(mf['files'])==13
for name,pin in mf['files'].items():
 p=W/'src'/name;data=p.read_bytes();assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['SHA256']==intent['files'][name]
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+s+'|' in x for s in ('','-pre','-post'))]
assert rows==EXPECTED['accounting'] and [x[0] for x in rows]==['1087656.131','1087656.132','1087656.133'] and [x[2:4] for x in rows]==[['COMPLETED','0:0'],['FAILED','1:0'],['COMPLETED','0:0']]
logs={};snapshots={}
for suffix in ('-pre','','-post'):
 path=W/(label+suffix+'.log');assert path.stat().st_size<50000;data=path.read_bytes();logs[suffix or 'GPU']=dict(bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())
 receipt=json.loads((W/(label+suffix+'-receipt.json')).read_bytes());assert receipt['log_SHA256']==logs[suffix or 'GPU']['SHA256']
 if suffix:
  value,=[json.loads(x) for x in data.splitlines() if x.startswith(b'{')];snapshots[suffix]=value;assert value['kfd_holder_count']==0 and value['master_port_available'] is True and len(value['pci'])==8
 else:
  text=data.decode();assert 'PermissionError' in text and 'gpu_holders' in text and 'for fd in (proc / "fd").iterdir():' in text
  assert 'line 21, in main' in text
assert snapshots['-pre']['boot_id']==snapshots['-post']['boot_id'] and snapshots['-pre']['pci']==snapshots['-post']['pci']
assert all(not (W/name).exists() for name in ('assigned.json','native-intent.json','child-owner.json','native-outcome.json','native.log'))
pre=(W/'source-and-asset-preflight.json').read_bytes();assert hashlib.sha256(pre).hexdigest()==EXPECTED['files']['source-and-asset-preflight.json']['SHA256'] and json.loads(pre)['passed']
print(json.dumps(dict(independent_failed_terminal_verified=True,claim_commit=intent['claim_commit'],outcome_SHA256=hashlib.sha256(raw).hexdigest(),source_files_verified=13,source_manifest_SHA256=intent['source_manifest_SHA256'],accounting_columns=out['runs'][0]['accounting_columns'],accounting=rows,logs=logs,preflight=json.loads(pre),failure=dict(type='PermissionError',stage='container_sameUID_fd_inventory',source_file='native.py',source_line=21,project_HIP_SDK_calls_before_failure=0,container_implicit_SDK_calls='UNKNOWN',model_process_spawn_not_reached=True),post=dict(actual_HOST_PCI_KFD_FD_port_clean=True,boot_unchanged=True,PCI_unchanged=True,native_owner_registry_present=False,original_keeper_passed=False),planned_native_updates=25,completed_native_updates=0,rank_artifacts=0,GPU_step_allocated=8,Model_GO=False)))
