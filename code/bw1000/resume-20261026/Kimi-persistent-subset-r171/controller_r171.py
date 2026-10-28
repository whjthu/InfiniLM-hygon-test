"""Frozen, inactive by default. Root release file required before any Slurm call."""
from pathlib import Path
import os,json,subprocess,time,datetime,hashlib,sys,re
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-persistent-subset-r171';PARENT='910374';NODE='a02r4n01';ACTIVE={'RUNNING','PENDING','COMPLETING','CONFIGURING'};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='910407';assert len(sys.argv)==2,'GPU_GO=false; root-owned release JSON required'
release=json.loads(Path(sys.argv[1]).read_bytes());assert release['GPU_authorized']is True and release['source_published']is True,'GPU_GO=false until root explicitly releases after interface acceptance';assert release['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',release['publication_commit'])
for n,h in json.loads((D/'source-manifest-r171.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
bp=R/'candidates/Kimi-persistent-subset-r171/CPU-build.json';assert sha(bp)==release['CPU_build_SHA256'];b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']and b['source_manifest_SHA256']==sha(D/'source-manifest-r171.json')and sha(str(b['native']).replace('/refactor',str(R),1))==b['native_SHA256']
env=dict(os.environ)
for k in list(env):
 if (k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','LD_PRELOAD']:env.pop(k)
key=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];label='Kimi-r171-subset-'+key;watch=R/'results'/(label+'-control.json');intent=R/'results/Kimi-r171-subset-launch-intent.json';assert not watch.exists()and not intent.exists();w={'completed':False,'passed':False,'GPU_submitted':False,'control_step':key,'Model_GO':False};actual=None;process=None
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(watch)
def cmd(a,seconds=25):
 p=subprocess.run(a,env=env,capture_output=True,text=True,timeout=seconds);assert p.returncode==0,(Path(a[0]).name,p.returncode);return p.stdout
def account(name):
 rows=[s.split('|')for s in cmd(['sacct','-S','2026-10-29','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120']).splitlines()if len(s.split('|'))>4 and s.split('|')[1]==name];assert len(rows)<=1;return rows[0]if rows else None
def idle(suffix):
 name=label+'-'+suffix;text=cmd(['srun','--immediate=15','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=256M','-w',NODE,'-t','00:01:00','--job-name='+name,'python3',str(D/'idle_r171.py')],90);rows=[json.loads(s)for s in text.splitlines()if s.startswith('{')];assert len(rows)==1 and rows[0]['passed']and not rows[0]['kfd_pids'];limit=time.monotonic()+25
 while True:
  row=account(name)
  if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
  assert time.monotonic()<limit;time.sleep(2)
 assert row[0]==rows[0]['step']and row[2:4]==['COMPLETED','0:0'];return {'row':rows[0],'terminal':row}
save()
try:
 raw=cmd(['scontrol','show','job',PARENT,'-o']);assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw;nodes=next(s.split('=',1)[1]for s in raw.split()if s.startswith('NodeList='));assert NODE in cmd(['scontrol','show','hostnames',nodes]).splitlines()
 with intent.open('x')as f:json.dump({'control_step':key,'release_file_SHA256':sha(sys.argv[1])},f)
 w['before']=idle('before');runtime=R/'results'/(label+'-manifest.json');m=dict(release,CPU_LD_LIBRARY_PATH=b['LD_LIBRARY_PATH'],candidate_Native_SHA256=b['native_SHA256']);runtime.open('x').write(json.dumps(m,indent=2));w['runtime_manifest_SHA256']=sha(runtime);save()
 child=dict(env,MANIFEST_R171=str(runtime).replace(str(R),'/refactor',1),WAVE_INSIDE_R2323='Kimi-persistent-subset-r171/GPU_inside_r171.sh');fo=(R/'logs'/(label+'-client.log')).open('x')
 args=['srun','--immediate=15','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c4','--gres=dcu:1','--mem=16000M','-w',NODE,'-t','00:04:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J-%t.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];process=subprocess.Popen(args,env=child,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w['GPU_submitted']=True;save();deadline=time.monotonic()+295
 while True:
  row=account(label)
  if row:
   if actual is None:actual=row[0]
   assert actual==row[0]and actual.startswith(PARENT+'.');w.update(actual_handle=actual,actual_row=row)
   if row[2].split()[0].rstrip('+')not in ACTIVE:w['terminal']=row;save();break
  assert time.monotonic()<deadline;save();time.sleep(4)
 process.wait(timeout=25);fo.close();w['post']=idle('post');assert row[2:4]==['COMPLETED','0:0'];p=R/'results'/('Kimi-persistent-subset-r171-'+actual+'.json');q=json.loads(p.read_bytes());x=q['data'];assert q['actual_step']==actual and q['returncode']==0 and q['candidate_Native_SHA256']==b['native_SHA256']and all(v['actual_DSO_SHA256']==b['native_SHA256']for v in q['actual_bindings'])
 assert x['completed']and x['normal_Runtime_close']and x['packed_sets']==92 and x['packed_unique_storages']==368 and x['checked_leaf_aliases']==494592 and x['cache']['all_bytes_zero_readback']==1086011136 and x['workspace']['layer_fields']==92 and x['workspace']['unique_storage']==1 and not x['communicator_created']and x['model_forward_calls']==0
 assert all(x['lease_tests'][k]for k in ['normal_drain_event_and_full_pattern','host_exception_drain_event_and_full_pattern','reentrant_rejected','crossowner_rejected','wrong_owner_thread_rejected_before_GPU','ownerB_isolated']);assert all(v for k,v in x['weak_owner_tensor_release'].items()if k!='raw_address_reuse_forced')
 lp=R/'logs'/(label+'-'+actual+'-0.log');assert lp.read_text(errors='replace').count('KIMI_PERSISTENT_SUBSET_R171_NORMAL')==1;w.update(completed=True,passed=True,result_path=str(p),result_SHA256=sha(p),post_KFD_empty=True,scope='persistent subset only; fullModel UNKNOWN_STOP');save()
except BaseException as e:
 w['error_class']=type(e).__name__;save()
 if w['GPU_submitted']:
  try:
   row=account(label)
   if row and(actual is None or row[0]==actual):
    actual=row[0]
    if row[2].split()[0].rstrip('+')in ACTIVE:
     assert actual.startswith(PARENT+'.');cmd(['scancel','--signal=TERM',actual]);limit=time.monotonic()+65
     while time.monotonic()<limit:
      row=account(label)
      if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
      time.sleep(3)
    w['terminal']=row
   elif process and process.poll()is None:process.terminate();process.wait(timeout=25)
   w['error_post']=idle('errorpost')
  except BaseException as pe:w['cleanup_error_class']=type(pe).__name__
  save()
 raise
finally:print(json.dumps({k:w.get(k)for k in ['completed','passed','GPU_submitted','actual_handle','post_KFD_empty','error_class']}),flush=True)
