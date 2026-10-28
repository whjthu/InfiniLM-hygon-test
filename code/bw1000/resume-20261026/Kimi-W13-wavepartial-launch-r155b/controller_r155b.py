"""One owned TP16 component launch; no compile, model import or restart."""
from pathlib import Path
import os,json,subprocess,time,datetime,hashlib,socket,re,sys,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-W13-wavepartial-launch-r155b';S=R/'incoming/Kimi-W13-wavepartial-component-r155';PARENT='910374';NODES=['a02r4n01','a02r4n04'];ACTIVE={'RUNNING','PENDING','COMPLETING','CONFIGURING'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='910407','root CPU keeper required'
source_commit=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',source_commit) and source_commit.startswith('6001d4d')
key=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];label='Kimi-W13-wavepartial-r155-'+key
watch=R/'results'/('Kimi-W13-wavepartial-r155-'+key+'-control.json');intent=R/'results/Kimi-W13-wavepartial-r155-launch-intent.json'
assert not watch.exists() and not intent.exists(),'no restart/second launch'
env=dict(os.environ)
for k in list(env):
 if (k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_')) and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k)
w={'completed':False,'passed':False,'GPU_submitted':False,'control_actual_step':key,'label':label,'nodes':NODES,'source_commit':source_commit,'scope':'controlled L1/L47 TP16 components; not full Model/current449 routes'}
actual=None;process=None
# Private logs remain on cluster; only booleans and hashes enter report.
def save():
 w['observedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();temp=watch.with_suffix('.tmp');temp.write_text(json.dumps(w,indent=2));temp.replace(watch)
def cmd(a,seconds=30):
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=seconds);assert z.returncode==0,('command_failed',Path(a[0]).name,z.returncode);return z.stdout

def account(name):
 rows=[s.split('|')for s in cmd(['sacct','-S','2026-10-25','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120']).splitlines()if len(s.split('|'))>4 and s.split('|')[1]==name]
 assert len(rows)<=1;return rows[0]if rows else None

def idle(suffix):
 name=label+'-'+suffix
 text=cmd(['srun','--immediate=30','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+name,'python3',str(D/'idle_r155b.py')],150)
 rows=[json.loads(s)for s in text.splitlines()if s.startswith('{')];assert len(rows)==2 and {q['node']for q in rows}==set(NODES) and all(q['passed']and not q['kfd_pids']for q in rows)
 steps={q['step']for q in rows};assert len(steps)==1;step=steps.pop();limit=time.monotonic()+40
 while True:
  row=account(name)
  if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
  assert time.monotonic()<limit;time.sleep(2)
 assert row[0]==step and row[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':row}

save()
try:
 pins=json.loads((D/'launch-source-manifest-r155b.json').read_bytes())
 for n,h in pins['files_SHA256'].items():assert sha(D/n)==h
 assert sha(S/'source-manifest-r155.json')=='a4e967fe29460fb7066d220634f13e1def0590d8d4930feb2c18d070879d6183'
 src=json.loads((S/'source-manifest-r155.json').read_bytes())
 for n,h in src['files_SHA256'].items():assert sha(S/n)==h
 bp=R/'candidates/Kimi-MoE-phase-component-r155/CPU-build.json';assert sha(bp)=='c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd';b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']
 assert b['native_SHA256']=='d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b'
 for field in ['native','library']:assert sha(str(b[field]).replace('/refactor',str(R),1))==b[field+'_SHA256']
 raw=cmd(['scontrol','show','job',PARENT,'-o']);assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw
 spec=next(s.split('=',1)[1]for s in raw.split()if s.startswith('NodeList='));assert set(NODES)<=set(cmd(['scontrol','show','hostnames',spec]).splitlines())
 # O_EXCL launch intent permanently prevents automatic resubmission, including failure.
 with intent.open('x')as f:json.dump({'control_step':key,'label':label,'source_commit':source_commit,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
 w['fresh_before']=idle('before');save()
 addr=next(q['ib0_ipv4'][0]for q in w['fresh_before']['rows']if q['node']==NODES[0]);assert addr.startswith('12.')
 runtime=R/'results'/('Kimi-W13-wavepartial-r155-'+key+'-manifest.json')
 m={'GPU_authorized':True,'source_published':True,'publication_branch':'bw1000-experiments','publication_commit':source_commit,'parent_job':PARENT,'expected_nodes':NODES,'CPU_build_path':'/refactor/candidates/Kimi-MoE-phase-component-r155/CPU-build.json','CPU_build_SHA256':sha(bp),'master_addr':addr,'master_port':43920}
 with runtime.open('x')as f:json.dump(m,f,indent=2)
 w['runtime_manifest_SHA256']=sha(runtime);w['ready']=True;save()
 childenv=dict(env,MANIFEST_R155=str(runtime).replace(str(R),'/refactor',1),WAVE_INSIDE_R2323='Kimi-W13-wavepartial-launch-r155b/GPU_inside_r155b.sh')
 log=R/'logs'/(label+'-client.log');fo=log.open('x')
 args=['srun','--immediate=30','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:12:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J-%t.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 process=subprocess.Popen(args,env=childenv,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w['GPU_submitted']=True;save();deadline=time.monotonic()+810
 while True:
  row=account(label)
  if row:
   if actual is None:actual=row[0]
   assert actual==row[0];w['actual_handle']=actual;w['actual_row']=row
   if row[2].split()[0].rstrip('+')not in ACTIVE:w['terminal']=row;save();break
  if process.poll()is not None and not row:raise RuntimeError('srun ended before actual owned GPU step')
  assert time.monotonic()<deadline,'bounded controller timeout';save();time.sleep(8)
 process.wait(timeout=35);fo.close();w['post_KFD']=idle('post');w['results']=[]
 for node in range(2):
  p=R/'results'/('Kimi-MoE-phase-r155-'+actual+'-node%d.json'%node);q=json.loads(p.read_bytes());assert q['completed']and q['normal_CUDA_cleanup']and len(q['experiments'])==2
  for e in q['experiments']:
   assert e['native']['all8normal_before_communicators'] and len(e['native']['reports'])==8
   for owner in e['native']['reports']:assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['arms'])==26 and len(owner['profile'])==18
  lp=R/'logs'/(label+'-'+actual+'-'+str(node)+'.log');text=lp.read_text(errors='replace');assert text.count('KIMI_MOE_PHASE_R150_NORMAL')==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text
  w['results'].append({'node':node,'path':str(p),'SHA256':sha(p),'completed':True,'all16_layer_owner_closes':True,'network_IBext_noSocket':True,'log_SHA256':sha(lp)})
 assert w['terminal'][2:4]==['COMPLETED','0:0'];w.update(completed=True,passed=True,post_KFD_empty=True,normal_layer_owner_closes=32);save()
except BaseException as e:
 w['error_class']=type(e).__name__;w['error_stage']='after_launch'if w['GPU_submitted']else 'before_launch';save()
 if w['GPU_submitted']:
  try:
   row=account(label)
   if row and (actual is None or row[0]==actual):
    actual=row[0];w['actual_handle']=actual
    if row[2].split()[0].rstrip('+')in ACTIVE:
     assert actual.startswith(PARENT+'.');cmd(['scancel','--signal=TERM',actual]);limit=time.monotonic()+100
     while time.monotonic()<limit:
      row=account(label)
      if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
      time.sleep(3)
    w['terminal']=row;save()
   elif process and process.poll()is None:process.terminate();process.wait(timeout=30)
   w['error_post_KFD']=idle('errorpost')
  except BaseException as pe:w['cleanup_error_class']=type(pe).__name__
  save()
 raise
finally:
 print(json.dumps({k:w.get(k)for k in ['completed','passed','GPU_submitted','actual_handle','error_class','post_KFD_empty']}),flush=True)
