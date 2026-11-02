import pathlib,json,hashlib,subprocess,datetime,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');STEP='910374.1320';LABEL='Deep-HC-on-current-profile-r393-Model-v1';OUT=R/'results/Deep393-1320-original-terminal-archive-R-only.json';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def host(p):
 if str(p).startswith('/refactor/'):return R/pathlib.Path(p).relative_to('/refactor')
 if str(p).startswith('/shared/'):return pathlib.Path('/public/home/aclsylqidf/shared')/pathlib.Path(p).relative_to('/shared')
 return pathlib.Path(p)
if OUT.exists():
 b=OUT.read_bytes();print(json.dumps(dict(archive=json.loads(b),remote_archive_path=str(OUT),remote_archive_SHA256=hashlib.sha256(b).hexdigest(),existing_original=True)));raise SystemExit(0)
wpath=R/'results'/(LABEL+'-watch.json');w=json.loads(wpath.read_bytes());assert w['completed']and w['actual_GPU']==STEP and w.get('terminal'),'Originalstepnotterminal:follow it, noarchiveyet'
accounting=[l for l in subprocess.check_output(['sacct','-S','2026-11-02','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%120,State,ExitCode,Elapsed,Start,End,NodeList'],text=True,timeout=25).splitlines()if LABEL in l]
assert any(l.startswith('910407.785|')and l.split('|')[2].split()[0]in ['COMPLETED','FAILED','TIMEOUT','CANCELLED']for l in accounting),'Follow originalCPUkeeper toterminal; noresubmit'
postpath=R/'results/Deep393-1320-independent-post-original.json'
if postpath.exists():post=json.loads(postpath.read_bytes())
else:
 probe="""import pathlib,os,json,socket,datetime
root=pathlib.Path('/sys/class/kfd/kfd/proc');pids=[int(p.name)for p in root.iterdir()if p.name.isdigit()];matches=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0));matched=[v for v in args if v.endswith(('/bootstrap_r206b.py','/worker_r206b.py','/modern_Deep_independent64_r9278.py','/client_chunk_pair.py'))]
  if matched:matches.append({'PID':int(p.name),'executables':matched})
 except(OSError,PermissionError):pass
port=False
with socket.socket()as s:
 try:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',43882));port=True
 except OSError:pass
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'KFD_available':root.exists(),'kfd_pids':pids,'owned_or_old_Deep_processes':matches,'HTTP43882_bindable':port,'GPU_imported':False};print(json.dumps(q))
"""
 compile(probe,'readonly_independent_post','exec')
 args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a02r4n01','--gres=none','--mem=512M','-t','00:02:00','--job-name=Deep393-independent-post','python3','-c',probe]
 p=subprocess.run(args,capture_output=True,text=True,timeout=160);values=[json.loads(l)for l in p.stdout.splitlines()if l.startswith('{')];assert p.returncode==0 and len(values)==1,(p.returncode,p.stdout,p.stderr);post=values[0]
 with postpath.open('x')as f:json.dump(post,f,indent=2);f.write('\n')
paths={};files={}
def add(path):
 p=host(path);b=p.read_bytes();files[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf-8'));return str(p)
paths['watch']=add(wpath);rpath=R/'results'/('Deep-HC-on-current-profile-r393-'+STEP+'.json');paths['report']=add(rpath);r=json.loads(files[paths['report']]['text']);root=rpath.with_suffix('')
paths['client']=add(root/'prefill-pair/result.json')
paths['config']=add(pathlib.Path(r['model_directory'])/'config.json');assert files[paths['config']]['SHA256']=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
S=R/'incoming/Deep-HC-on-current-profile-r393/source-v1';D=R/'incoming/Deep-HC-on-current-profile-r393/controller-v1'
for name,path in [('owner_manifest',S/'owner/SOURCE_MANIFEST.json'),('source_manifest',S/'SOURCE_MANIFEST.json'),('controller_manifest',D/'SOURCE_MANIFEST.json'),('CPU_whole',R/'candidates/Deep-HC-on-current-profile-r393-CPU-v1/CPU-whole.json'),('CPU_entry',R/'candidates/Deep-HC-on-current-profile-r393-CPU-v1/CPU-entry.json')]:paths[name]=add(path)
paths['execution_claim']=add(S.parent/'MODEL_EXECUTION_CLAIM-ca32a72c.json');paths['runtime_GO']=add(S.parent/'execution-GO-ca32a72c.json')
ref=json.loads((S/'owner/FROZEN_FIXTURE_REF.json').read_bytes());paths['fixture']=add(ref['path']);assert files[paths['fixture']]['SHA256']==ref['SHA256'];paths['legacy']=add(r.get('legacy_load_receipt')or R/'results'/('deep-c4-independent64-r9278-n01-'+STEP+'.json'))
if r.get('legacy_load_receipt_SHA256'):assert files[paths['legacy']]['SHA256']==r['legacy_load_receipt_SHA256']
paths['wires']=[]
for i in range(6):
 folder=root/'prefill-pair'/f'{i:02d}-mode1-1024';paths['wires'].append(dict(index=i,request=add(folder/'request-001.request.json'),response=add(folder/'request-001.wire')))
paths['profile_packets']=[];paths['profile_metadata']=[]
for request in r['requests']:
 for v in request.get('owner_profiles',[]):
  for kind,record in v['scalar_packets'].items():
   p=add(record['path']);assert files[p]['SHA256']==record['SHA256'];paths['profile_packets'].append(dict(kind=kind,container_path=record['path'],host_path=p,SHA256=record['SHA256']))
  paths['profile_metadata'].append(add(v['metadata_path']))
a=dict(schema=1,step=STEP,paths=paths,raw_files=files,accounting=accounting,independent_post=post,independent_post_path=str(postpath),independent_post_SHA256=sha(postpath),collected_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw_payload_policy='R/localonly. No rawHTTP/history/ownerpacket source inGit; publishonlySHA/counts/compactmetrics andreader.')
with OUT.open('x')as f:json.dump(a,f,ensure_ascii=False,indent=2);f.write('\n')
b=OUT.read_bytes();print(json.dumps(dict(archive=a,remote_archive_path=str(OUT),remote_archive_SHA256=hashlib.sha256(b).hexdigest(),existing_original=False)))
