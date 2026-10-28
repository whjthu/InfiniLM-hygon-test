"""One owned TP32 I96 L1 component launch; no compile or full Model."""
from pathlib import Path
import os,json,subprocess,time,datetime,hashlib,socket,re,sys,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-TP32-I96-L1-launch-r163a';S=R/'incoming/Kimi-TP32-I96-template-component-r163';PARENT='910380';NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'];ACTIVE={'RUNNING','PENDING','COMPLETING','CONFIGURING'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='910407','root CPU keeper required'
source_commit=sys.argv[1];assert re.fullmatch('[0-9a-f]{40}',source_commit)
key=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];label='Kimi-TP32-I96-r163a-'+key
watch=R/'results'/('Kimi-TP32-I96-r163a-'+key+'-control.json');intent=R/'results/Kimi-TP32-I96-r163a-launch-intent.json'
assert not watch.exists() and not intent.exists(),'no restart/second launch'
env=dict(os.environ)
for k in list(env):
 if (k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_')) and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:env.pop(k)
w={'completed':False,'passed':False,'GPU_submitted':False,'control_actual_step':key,'label':label,'nodes':NODES,'source_commit':source_commit,'scope':'TP32PP1 L1 genericI96 vs TT2Tile4I96/W2pair; T8/T16/T64/Gaussian64; not full Model/current449 traffic'}
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
 text=cmd(['srun','--immediate=30','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+name,'python3',str(D/'idle_r163a.py')],150)
 rows=[json.loads(s)for s in text.splitlines()if s.startswith('{')];assert len(rows)==4 and {q['node']for q in rows}==set(NODES) and all(q['passed']and not q['kfd_pids']for q in rows)
 steps={q['step']for q in rows};assert len(steps)==1;step=steps.pop();limit=time.monotonic()+40
 while True:
  row=account(name)
  if row and row[2].split()[0].rstrip('+')not in ACTIVE:break
  assert time.monotonic()<limit;time.sleep(2)
 assert row[0]==step and row[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':row}

save()
try:
 pins=json.loads((D/'launch-source-manifest-r163a.json').read_bytes())
 for n,h in pins['files_SHA256'].items():assert sha(D/n)==h
 assert sha(S/'source-manifest-r163.json')=='e4a2029fb5cc026779567e8c3966e6dd09579ce201d6f9874892ee7293cedb15'
 src=json.loads((S/'source-manifest-r163.json').read_bytes())
 for n,h in src['files_SHA256'].items():assert sha(S/n)==h
 bp=R/'candidates/Kimi-TP32-I96-template-r163/CPU-build.json';assert sha(bp)=='30eafbbf4934bd446c1a77a01a12be822e97a7964c2d8b2c58ef830d9ee3f055';b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']
 assert b['actual_step']=='910407.517'and b['source_manifest_SHA256']==sha(S/'source-manifest-r163.json')
 assert b['native_SHA256']=='2e7c52d34c9ec6ccc6306517e77b76f56265e3309071979d43d425bc28254197'
 assert b['library_SHA256']=='94d7519e4472e8c4643792a7efedfd644b97978e5e6b2cbf347da29c8debc6f1'
 for field in ['native','library']:assert sha(str(b[field]).replace('/refactor',str(R),1))==b[field+'_SHA256']
 raw=cmd(['scontrol','show','job',PARENT,'-o']);assert 'UserId=aclsylqidf('in raw and 'JobState=RUNNING'in raw
 spec=next(s.split('=',1)[1]for s in raw.split()if s.startswith('NodeList='));assert set(NODES)<=set(cmd(['scontrol','show','hostnames',spec]).splitlines())
 # O_EXCL launch intent permanently prevents automatic resubmission, including failure.
 with intent.open('x')as f:json.dump({'control_step':key,'label':label,'source_commit':source_commit,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
 w['fresh_before']=idle('before');save()
 addr=next(q['ib0_ipv4'][0]for q in w['fresh_before']['rows']if q['node']==NODES[0]);assert addr.startswith('12.')
 runtime=R/'results'/('Kimi-TP32-I96-r163a-'+key+'-manifest.json')
 m={'GPU_authorized':True,'source_published':True,'publication_branch':'bw1000-experiments','publication_commit':source_commit,'parent_job':PARENT,'expected_nodes':NODES,'CPU_build_path':'/refactor/candidates/Kimi-TP32-I96-template-r163/CPU-build.json','CPU_build_SHA256':sha(bp),'source_manifest_SHA256':sha(S/'source-manifest-r163.json'),'master_addr':addr,'master_port':44363}
 with runtime.open('x')as f:json.dump(m,f,indent=2)
 w['runtime_manifest_SHA256']=sha(runtime);w['ready']=True;save()
 childenv=dict(env,MANIFEST_R163=str(runtime).replace(str(R),'/refactor',1),WAVE_INSIDE_R2323='Kimi-TP32-I96-L1-launch-r163a/GPU_inside_r163a.sh')
 log=R/'logs'/(label+'-client.log');fo=log.open('x')
 args=['srun','--immediate=30','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:12:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J-%t.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
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
 for node in range(4):
  p=R/'results'/('Kimi-TP32-I96-template-r163-'+actual+'-node%d.json'%node);q=json.loads(p.read_bytes());assert q['completed']and q['normal_CUDA_cleanup']and q['actual_step']==actual and q['node']==node and q['Native_SHA256']==b['native_SHA256']
  assert q['source_manifest_SHA256']==sha(S/'source-manifest-r163.json')and q['native']['completed']and q['native']['all8normal_before_communicators']
  owners=q['native']['reports'];assert len(owners)==8 and [o['global_tp_rank']for o in owners]==list(range(node*8,node*8+8))
  for owner in owners:
   assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['arms'])==48
   for a in owner['arms']:assert a['nonfinite']==0 and a['input_readonly_tail_canary']and a['private_provider_counter_delta_zero']and a['actual_I96_branch_hits']==int(a['candidate_I96'])
  assert set(q['input']['cases'])=={'8','16','64','Gaussian64'}
  lp=R/'logs'/(label+'-'+actual+'-'+str(node)+'.log');text=lp.read_text(errors='replace');assert text.count('KIMI_TP32_I96_R163_NORMAL')==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text
  w['results'].append({'node':node,'path':str(p),'SHA256':sha(p),'completed':True,'all8_L1_layer_owner_closes':True,'network_IBext_noSocket':True,'log_SHA256':sha(lp)})
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
