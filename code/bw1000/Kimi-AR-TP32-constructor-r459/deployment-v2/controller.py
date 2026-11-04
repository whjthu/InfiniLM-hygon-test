from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,sys,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Kimi-AR-TP32-constructor-r459/deployment-v2'
RUN='ar-tp32-ctor-20261104-b';LABEL='Kimi-r459-AR-constructor'
NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def host(p):return R/Path(p).relative_to('/refactor')
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and(not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','KIMI_','INFINILM_','GLOBAL_','K459_'))or k in['SLURM_CONF','SLURM_CONF_SERVER'])}
def publish(p,q):
 with Path(p).open('x')as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def config():return json.loads((D/'config.json').read_bytes())
def gpu_command():
 return ['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1',
 '-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:10:00',
 '--job-name='+LABEL+'-'+RUN,'--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),'bash',str(D/'outer.sh'),'kimi']
def scope_contract(m):
 q=config();c=m['resource_claim']
 assert Path(__file__).resolve().parent==D.resolve()
 assert NODES==q['nodes']==c['nodes'] and len(set(NODES))==4
 assert q['run']==RUN and c['parent']==910380 and c['cards']==32
 assert (c['TP'],c['PP'],c['logical_capacity'],c['physical_rows'],c['state_rows'],c['blocks'])==(32,1,1,1,2,132)
 assert (c['GPU_max_s'],c['Native_max_s'],c['CPU_keeper']['max_s'])==(600,540,900)
 assert c['weights']==c['forward']==c['HTTP']==0 and c['constructor_calls']==32
 cmd=gpu_command();assert all(v in cmd for v in ['-N4','-n4','-c64','--gres=dcu:8','--mem=128000M','--jobid=910380'])
 assert cmd[cmd.index('-w')+1]==','.join(NODES) and cmd[cmd.index('-t')+1]=='00:10:00'
 return q
def verify(digest):
 assert sha(D/'SOURCE_MANIFEST.json')==digest
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());q=scope_contract(m)
 for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
 for name in ['inside.sh','outer.sh']:assert b'\r'not in(D/name).read_bytes()
 for p,h in m['parent_files_SHA256'].items():
  path=host(p);assert sha(path)==h,p
  if path.name=='CPU_SOURCE_MANIFEST.json':
   for name,digest in json.loads(path.read_bytes())['files_SHA256'].items():assert sha(path.parent/name)==digest,(p,name)
 receipt=host(q['CPU_receipt']);assert sha(receipt)==q['CPU_receipt_SHA256']
 r=json.loads(receipt.read_bytes());assert r['passed']and r['completed']and r['CPU_only']and r['actual_CPU']=='910407.966'
 assert not r['Native_dlopen']and not r['Runtime_created']and not r['GPU']and r['native_SHA256']==q['Native_SHA256']
 assert sha(host(q['native']))==q['Native_SHA256']
 for p,h in r['all_linked_object_SHA256'].items():assert sha(host(p))==h,p
 for pin in q['dependency_libraries'].values():assert sha(host(pin['path']))==pin['SHA256']
 assert q['environment']['INFINILM_KIMI_MOE_WORKSPACE_R4665']=='0'
 assert q['environment']['INFINILM_KIMI_MOE_EXECUTION_R4665']=='0'
 return m
def command(argv,timeout=25):return subprocess.check_output(argv,text=True,timeout=timeout)
def rows():return[x.split('|')for x in command(['sacct','-S','2026-10-31','-j','910380,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
def hosts(s):return command(['scontrol','show','hostnames',s]).splitlines()
def exact(label):
 found=[r for r in rows()if r[1]==label];assert len(found)<=1;return found[0]if found else None
def membership():
 result={}
 for parent,nodes in [('910380',NODES),('910407',['a14r2n20'])]:
  text=command(['scontrol','show','job',parent,'-o']);assert 'JobState=RUNNING'in text and'UserId=aclsylqidf('in text
  assert set(nodes)<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',text).group(1)));result[parent]=text
 return result
def whole_cli():
 for name in ['inside.sh','outer.sh']:subprocess.run(['bash','-n',str(D/name)],check=True,timeout=10)
 env=clean_environment(os.environ);env.update(K459_METADATA_ONLY='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',SLURM_JOB_ID='recording-HOST',SLURM_STEP_ID='metadata-only',SLURM_PROCID='0')
 # Real shell environment and four actual metadata consumers; only hostnames/roles recorded.
 cmd=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--home','/public/home/aclsylqidf/hygon-inference-20261013/build-deps/root:/root',
 '--bind',str(R)+':/refactor','--bind','/public/home/aclsylqidf/shared:/shared:ro',
 '--bind','/public/home/aclsylqidf/hygon-inference-20261013:/old:ro','--bind','/dev/shm:/local',
 str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc',str(Path('/refactor')/D.relative_to(R)/'inside.sh')]
 z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
 proof=dict(argv=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr,Native_dlopen=False,Slurm=0,GPU=0)
 assert z.returncode==0,proof
 lines=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert len(lines)==1 and lines[0]['passed']and len(lines[0]['roles'])==4
 return proof
def close_owned(label,proc,recorded,exact,cleanup_limit,term_s=20.,kill_s=20.):
 """TERM then KILL only this keeper's known srun PGID/Slurm step, bounded."""
 result=dict(label=label,srun_PID=proc.pid,recorded_step=recorded,signals=[],observations=[],closed=False)
 step=recorded;row=None
 def group_live():
  try:os.killpg(proc.pid,0);return True
  except ProcessLookupError:return False
 def observe():
  nonlocal step,row
  try:
   got=exact(label)
   if got:
    assert re.fullmatch(r'910380[.]\d+',got[0]) and(step is None or step==got[0]),'Owned cleanup step identity mismatch'
    step=got[0];row=got
   result['observations'].append(dict(step=step,row=got,srun_rc=proc.poll(),group_live=group_live()))
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:
   result['observations'].append(dict(step=step,observation_error=repr(error),srun_rc=proc.poll(),group_live=group_live()))
  terminal=bool(row and row[2].split()[0].rstrip('+') in TERMINAL)
  # If accounting never yielded a step, absence is UNKNOWN, even if the local srun exits.
  return terminal and proc.poll() is not None and not group_live()
 def signal_owned(kind):
  if step is not None:
   assert re.fullmatch(r'910380[.]\d+',step),'Never signal an unknown or parent allocation'
   try:
    z=subprocess.run(['scancel','--signal='+kind,step],capture_output=True,text=True,timeout=min(10,max(1,cleanup_limit-time.monotonic())))
    result['signals'].append(dict(scope='known_Slurm_step',step=step,signal=kind,rc=z.returncode,stdout=z.stdout,stderr=z.stderr))
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:result['signals'].append(dict(scope='known_Slurm_step',step=step,signal=kind,error=repr(error)))
  try:os.killpg(proc.pid,getattr(signal,'SIG'+kind));result['signals'].append(dict(scope='owned_start_new_session_PGID',PGID=proc.pid,signal=kind))
  except ProcessLookupError:result['signals'].append(dict(scope='owned_start_new_session_PGID',PGID=proc.pid,signal=kind,already_gone=True))
 observe();signal_owned('TERM')
 stop=min(cleanup_limit,time.monotonic()+term_s)
 while time.monotonic()<stop:
  if observe():result['closed']=True;break
  time.sleep(min(1.,max(0,stop-time.monotonic())))
 if not result['closed']:
  signal_owned('KILL');stop=min(cleanup_limit,time.monotonic()+kill_s)
  while time.monotonic()<stop:
   if observe():result['closed']=True;break
   time.sleep(min(1.,max(0,stop-time.monotonic())))
 if not result['closed']:result['closed']=observe()
 result.update(actual_step=step,last_accounting=row,srun_returncode=proc.poll(),PGID_still_live=group_live(),terminal_unknown_or_unclosed=not result['closed'])
 return result


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha)
 if a.preflight_only:
  print(json.dumps(dict(passed=True,source_only=True,proof=whole_cli(),GPU=0,Slurm=0)));return 0
 assert m['GPU_GO']and not m['Model_GO'] and os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 q=config();folder=host(q['runtime_directory']);folder.mkdir(exist_ok=True)
 watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');w=dict(completed=False,passed=False,publication_commit=a.publication_commit,source_manifest_SHA256=a.manifest_sha,
 CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],actual_GPU=None,actual_client=None,HTTP=0,forward=0,weights=0,forced_cleanup=False)
 publish(watch,w);gpu=None;epoch=None;cleanup_limit=time.monotonic()+880;expected={}
 def save():
  w['observed_unix']=time.time();t=watch.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2)+'\n');t.replace(watch)
 def idle(phase):
  env=clean_environment(os.environ);env['K459_PIDS']=json.dumps(expected)
  cmd=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:01:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(70,max(1,cleanup_limit-time.monotonic())))
  obs=dict(phase=phase,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr);w.setdefault('CPU_idle_observations',[]).append(obs);save()
  vals=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert z.returncode==0 and len(vals)==4 and {v['node']for v in vals}==set(NODES)
  assert len({(v['node'],pci)for v in vals for pci in v['physical_PCI']})==32
  return vals
 def reports():
  found=[]
  for n in range(4):
   path=folder/('node%d.json'%n)
   if not path.exists():continue
   try:v=json.loads(path.read_bytes())
   except json.JSONDecodeError:continue
   assert v['run']==RUN and v['source_manifest_SHA256']==a.manifest_sha and v['epoch']==epoch and v['node']==n
   expected[NODES[n]]=[v['PID']]
   if v['completed']:found.append((path,v))
  return found
 try:
  w['membership']=membership()
  for r in rows():
   assert not r[1].startswith(LABEL+'-'+RUN) or (r[0]==w['CPU_keeper'] and r[1]==LABEL+'-'+RUN+'-keeper'), 'Existing label'
   if re.fullmatch(r'910380[.]\d+',r[0])and r[2].split()[0].rstrip('+')not in TERMINAL:assert not set(hosts(r[4]))&set(NODES),'Live overlap'
  w['fresh_before']=idle('before');epoch=time.time();w.update(epoch=epoch,Native_deadline=epoch+540,GPU_deadline=epoch+600,keeper_deadline=epoch+900);save()
  env=clean_environment(os.environ);env.update(WAVE_INSIDE_R2323=D.relative_to(R/'incoming').as_posix()+'/inside.sh',K459_EPOCH=str(epoch),K459_SOURCE_SHA=a.manifest_sha)
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:gpu=subprocess.Popen(gpu_command(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w['GPU_srun_PID']=gpu.pid;save()
  while time.time()<epoch+600:
   row=exact(LABEL+'-'+RUN)
   if row:
    assert w['actual_GPU']in(None,row[0]);w['actual_GPU']=row[0];w['GPU_accounting']=row
    if row[2].split()[0].rstrip('+')in TERMINAL:
     assert row[2:4]==['COMPLETED','0:0'],row
     if gpu.poll()==0:break
   if gpu.poll()not in(None,0):raise RuntimeError('GPU srun failed')
   reports();save();time.sleep(5)
  else:raise TimeoutError('Original GPU600 deadline')
  values=reports();assert len(values)==4
  for path,v in values:
   assert v['passed']and v['create_calls']==v['destroy_calls']==1 and v['actual_step']==w['actual_GPU']
   assert not any(e['event']=='cleanup_error'for e in v['events'])
   actual={p['PCI']for p in v['hardware']};prior=next(o for o in w['fresh_before']if o['node']==v['hostname']);assert actual==set(prior['physical_PCI'])
  w['node_reports']=[dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size)for p,v in values]
  w['normal_owners']=32;w['post']=idle('post');w.update(completed=True,passed=True,state='TP32_AR_CONSTRUCTOR_ONLY_NORMAL32_NO_MODEL_QUALIFICATION')
 except BaseException as error:
  w.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc())
  try:reports()
  except BaseException as err:w['report_error']=repr(err)
  if gpu is not None:
   w['forced_cleanup']=True
   try:w['owned_cleanup']=close_owned(LABEL+'-'+RUN,gpu,w['actual_GPU'],exact,cleanup_limit)
   except BaseException as err:w['cleanup_error']=repr(err)
  try:w['failure_post']=idle('failure-post')
  except BaseException as err:w['post_error']=repr(err)
 finally:save()
 print(json.dumps({k:w.get(k)for k in ['completed','passed','CPU_keeper','actual_GPU','normal_owners','error']}));return 0 if w['passed']else 1
if __name__=='__main__':raise SystemExit(main())
