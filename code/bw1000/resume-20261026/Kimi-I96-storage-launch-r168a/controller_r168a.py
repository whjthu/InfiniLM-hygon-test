"""Root CPU keeper: sequential regular/exact one-GPU processes; no CCL/Model."""
from pathlib import Path
import os,json,subprocess,time,datetime,hashlib,socket,re,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-I96-storage-launch-r168a';S=R/'incoming/Kimi-I96-shared-storage-probe-r168';PARENT='910374';NODE='a02r4n01';ACTIVE={'RUNNING','PENDING','COMPLETING','CONFIGURING'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='910407','root CPU keeper required'
publication=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',publication)
key=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];label='Kimi-I96-storage-r168a-'+key
watch=R/'results'/(label+'-control.json');intent=R/'results/Kimi-I96-storage-r168a-launch-intent.json';assert not watch.exists()and not intent.exists(),'one launch only, no restart'
env=dict(os.environ)
for k in list(env):
 if (k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','LD_PRELOAD']:env.pop(k)
w={'completed':False,'passed':False,'GPU_submitted':False,'control_actual_step':key,'parent':PARENT,'node':NODE,'source_commit':publication,'arms':{},'scope':'same packed production helper allocation only; no fakecomm/CCL/TP32 execution/Model','Model_GO':False};actual=None;process=None;arm=None;global_deadline=time.monotonic()+840
def save():
 w['observedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(watch)
def cmd(a,seconds=25):
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=seconds);assert z.returncode==0,('command_failed',Path(a[0]).name,z.returncode);return z.stdout
def account(name):
 rows=[s.split('|')for s in cmd(['sacct','-S','2026-10-28','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120']).splitlines()if len(s.split('|'))>4 and s.split('|')[1]==name];assert len(rows)<=1;return rows[0]if rows else None
def idle(suffix):
 name=label+'-'+suffix
 text=cmd(['srun','--immediate=15','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=256M','-w',NODE,'-t','00:01:00','--job-name='+name,'python3',str(D/'idle_r168a.py')],90)
 rows=[json.loads(s)for s in text.splitlines()if s.startswith('{')];assert len(rows)==1 and rows[0]['node']==NODE and rows[0]['passed']and not rows[0]['kfd_pids'];limit=time.monotonic()+25
 while True:
  row=account(name)
  if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
  assert time.monotonic()<limit;time.sleep(2)
 assert row[0]==rows[0]['step']and row[2:4]==['COMPLETED','0:0'];return {'row':rows[0],'terminal':row}
save()
try:
 pins=json.loads((D/'launch-source-manifest-r168a.json').read_bytes())
 for n,h in pins['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(S/'source-manifest-r168.json')==pins['component_source_manifest_SHA256']=='3a7d669d709d13d9282164aba2dbb1081538741c6775e0e06cffee026b1467a3'
 for n,h in json.loads((S/'source-manifest-r168.json').read_bytes())['files_SHA256'].items():assert sha(S/n)==h,n
 bp=R/'candidates/Kimi-I96-shared-storage-probe-r168/CPU-build.json';assert sha(bp)==pins['CPU_build_SHA256'];b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']and b['actual_step']=='910407.523'and b['source_manifest_SHA256']==pins['component_source_manifest_SHA256']
 for a in ['regular','exact']:
  assert b['arms'][a]['native_SHA256']==pins['arms_Native_SHA256'][a]and sha(str(b['arms'][a]['native']).replace('/refactor',str(R),1))==pins['arms_Native_SHA256'][a]
  assert b['arms'][a]['source_SHA256']==sha(S/('kimi_moe_storage_'+a+'_r168.cpp'))
 raw=cmd(['scontrol','show','job',PARENT,'-o']);assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw;spec=next(s.split('=',1)[1]for s in raw.split()if s.startswith('NodeList='));assert NODE in cmd(['scontrol','show','hostnames',spec]).splitlines()
 with intent.open('x')as f:json.dump({'control_step':key,'publication_commit':publication,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
 w['fresh_before']=idle('before');save()
 for arm in ['regular','exact']:
  assert time.monotonic()<global_deadline;actual=None;process=None;name=label+'-'+arm
  runtime=R/'results'/(label+'-'+arm+'-manifest.json');m={'GPU_authorized':True,'source_published':True,'publication_commit':publication,'CPU_build_SHA256':pins['CPU_build_SHA256'],'CPU_LD_LIBRARY_PATH':b['LD_LIBRARY_PATH'],'arm':arm,'expected_parent':PARENT,'expected_node':NODE,'candidate_Native_SHA256':pins['arms_Native_SHA256'][arm]}
  with runtime.open('x')as f:json.dump(m,f,indent=2)
  w['arms'][arm]={'submitted':False,'runtime_manifest_SHA256':sha(runtime)};save()
  childenv=dict(env,MANIFEST_R168A=str(runtime).replace(str(R),'/refactor',1),WAVE_INSIDE_R2323='Kimi-I96-storage-launch-r168a/GPU_inside_r168a.sh')
  log=R/'logs'/(name+'-client.log');fo=log.open('x')
  args=['srun','--immediate=15','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c4','--gres=dcu:1','--mem=16000M','-w',NODE,'-t','00:04:00','--job-name='+name,'--output='+str(R/'logs'/(name+'-%J-%t.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  process=subprocess.Popen(args,env=childenv,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w['GPU_submitted']=True;w['arms'][arm]['submitted']=True;save();deadline=min(global_deadline,time.monotonic()+295)
  while True:
   row=account(name)
   if row:
    if actual is None:actual=row[0]
    assert actual==row[0]and actual.startswith(PARENT+'.');w['arms'][arm].update(actual_handle=actual,actual_row=row)
    if row[2].split()[0].rstrip('+')not in ACTIVE:w['arms'][arm]['terminal']=row;save();break
   if process.poll()is not None and not row:raise RuntimeError('owned srun ended before accounting receipt')
   assert time.monotonic()<deadline,'bounded arm timeout';save();time.sleep(4)
  process.wait(timeout=25);fo.close();assert row[2:4]==['COMPLETED','0:0']
  p=R/'results'/('Kimi-I96-storage-r168-'+arm+'-'+actual+'.json');q=json.loads(p.read_bytes());x=q['data'];assert q['arm']==arm and q['actual_step']==actual and q['returncode']==0 and q['candidate_Native_SHA256']==q['actual_dladdr_SHA256']==pins['arms_Native_SHA256'][arm]and q['CPU_receipt_SHA256']==pins['CPU_build_SHA256']and q['publication_commit']==publication
  assert q['candidate_source_SHA256']==b['arms'][arm]['source_SHA256']and q['source_manifest_SHA256']==pins['component_source_manifest_SHA256']
  assert x['completed']and x['normal_Runtime_close']and not x['communicator_created']and not x['weight_payload_loaded']and x['forward_calls']==0 and x['actual_alias_geometry']['unique_storages']==4 and x['actual_alias_geometry']['leaf_aliases_checked']==5376
  for k in ['before_Runtime','after_Runtime','after_shared_banks_construct','after_shared_banks_destroy','after_Runtime_destroy']:assert x[k]['free_bytes']>4*1024**3 and x[k]['pool_reserved_bytes']=='UNKNOWN_NO_PUBLIC_POOL_STATS_API'
  lp=R/'logs'/(name+'-'+actual+'-0.log');text=lp.read_text(errors='replace');assert text.count('KIMI_I96_STORAGE_R168_NORMAL')==1
  phases={k:x[k]for k in ['before_Runtime','after_Runtime','after_shared_banks_construct','after_shared_banks_destroy','after_Runtime_destroy']};allocation=phases['after_shared_banks_construct']['device_used_bytes']-phases['after_Runtime']['device_used_bytes'];background=phases['after_Runtime']['device_used_bytes']-phases['before_Runtime']['device_used_bytes']
  w['arms'][arm].update(completed=True,passed=True,result_path=str(p),result_SHA256=sha(p),log_SHA256=sha(lp),phases=phases,actual_alias_geometry=x['actual_alias_geometry'],observed_bank_construct_delta_bytes=allocation,Runtime_background_delta_bytes=background,background_within16MiB=abs(background)<=16*1024**2,normal_Runtime_close=True,dladdr_SHA_verified=True);save()
  if arm=='regular':w['fresh_between']=idle('between');save()
 w['fresh_post']=idle('post');delta=w['arms']['regular']['observed_bank_construct_delta_bytes']-w['arms']['exact']['observed_bank_construct_delta_bytes'];design=135725056
 w['comparison']={'observed_regular_minus_exact_bytes':delta,'design_single_layer_bytes':design,'deviation_bytes':delta-design,'within16MiB':abs(delta-design)<=16*1024**2,'over100MiB':delta>100*1024**2,'allocation_delta_closed':abs(delta-design)<=16*1024**2 and delta>100*1024**2,'actual_malloc_granularity':'UNKNOWN; use measured delta','all92_bytes_is_extrapolation_only':12486705152,'not_58point9_minus11point629':True,'fullModel_peak':'UNKNOWN_STOP'};w.update(completed=True,passed=True,post_KFD_empty=True);save()
except BaseException as e:
 w.update(error_class=type(e).__name__,error_stage='after_GPU'if w['GPU_submitted']else'before_GPU');save()
 if arm and w['arms'].get(arm,{}).get('submitted'):
  try:
   row=account(label+'-'+arm)
   if row and(actual is None or row[0]==actual):
    actual=row[0]
    if row[2].split()[0].rstrip('+')in ACTIVE:
     assert actual.startswith(PARENT+'.');cmd(['scancel','--signal=TERM',actual]);limit=time.monotonic()+65
     while time.monotonic()<limit:
      row=account(label+'-'+arm)
      if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
      time.sleep(3)
    w['arms'][arm]['terminal']=row
   elif process and process.poll()is None:process.terminate();process.wait(timeout=25)
   w['error_post_KFD']=idle('errorpost')
  except BaseException as pe:w['cleanup_error_class']=type(pe).__name__
  save()
 raise
finally:print(json.dumps({k:w.get(k)for k in ['completed','passed','GPU_submitted','post_KFD_empty','comparison','error_class']}),flush=True)
