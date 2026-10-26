"""Unique TP16 component supervisor; prepared source fixed before GPU submission."""
import pathlib,json,subprocess,os,socket,time,datetime,hashlib,traceback,re,statistics
R=pathlib.Path.home()/'hygon-inference-refactor-20261014';S=R/'incoming/Kimi-vocab-TP16-component-r9176';PARENT='910380';NODES=['a14r2n08','a14r2n09'];LABEL='Kimi-vocab-TP16-r9176';H=R/'results/Kimi-vocab-TP16-control-r9176.json';assert not H.exists();env=dict(os.environ)
for k in list(env):
 if(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env.pop(k)
q={'CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'GPU_submitted':False,'completed':False,'restart':False,'manifest_SHA256':'4935ad92fed1ab12ff49eba7fd9918ee51b77e6f672b9f42a3dcb76ec6989ff4','scope':'headonly component, noModel'};proc=None
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save():
 q['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=H.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2),encoding='utf-8');p.replace(H)
def account(name):
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-25','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%95,State%35,ExitCode,NodeList%100,Elapsed'],env=env,text=True,timeout=25).splitlines()if len(s.split('|'))>=6 and s.split('|')[1]==name];assert len(rows)<=1,(name,rows);return rows[0]if rows else None
def terminal(name,handle):
 for _ in range(70):
  row=account(name)
  if row:
   assert row[0]==handle
   if row[2].split()[0].rstrip('+')not in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:return row
  save();time.sleep(1)
 raise RuntimeError('authoritativeterminaltimeout '+handle)
def fresh(name):
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+name,'python3',str(R/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];v={'rows':rows,'returncode':z.returncode,'stderr':z.stderr};q[name]=v;save();assert z.returncode==0 and len(rows)==2 and{x['node']for x in rows}==set(NODES)and all(x['passed']and not x['kfd_pids']for x in rows);handles={x['step']for x in rows};assert len(handles)==1;v['terminal']=terminal(name,handles.pop());assert v['terminal'][2:4]==['COMPLETED','0:0'];save();return v
save()
try:
 m=json.loads((S/'manifest-r9176.json').read_bytes());assert sha(S/'manifest-r9176.json')==q['manifest_SHA256']
 for n,h in m['source_SHA256'].items():assert sha(S/n)==h,n
 q['CPU_preflight_authoritative']=terminal('Kimi-vocab-TP16-preflight-r9176','910380.414');assert q['CPU_preflight_authoritative'][2:4]==['COMPLETED','0:0'];save();job=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],env=env,text=True,timeout=25);assert'UserId=aclsylqidf('in job and'JobState=RUNNING'in job;spec=next(x.split('=',1)[1]for x in job.split()if x.startswith('NodeList='));assert set(NODES)<=set(subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=25).splitlines());assert account(LABEL)is None
 q['phase']='fresh';save();before=fresh('Kimi-vocab-TP16-before-r9176');q['head_IP']=next(x['ib0_ipv4'][0]for x in before['rows']if x['node']==NODES[0]);q['baseport']=19646;q['watch_ready_before_GPU']=True;q['phase']='ready_before_unique_GPU';save()
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(NODES),'--gres=dcu:8','--mem=128000M','-t','00:20:00','--job-name='+LABEL,'--export=ALL,GLOBAL_MODEL_MODE_R2354=shca,KIMI_VOCAB_HEAD_ADDR_R9176='+q['head_IP']+',KIMI_VOCAB_BASEPORT_R9176='+str(q['baseport'])+',WAVE_INSIDE_R2323=Kimi-vocab-TP16-component-r9176/GPU_inside.sh','--output='+str(R/'logs'/(LABEL+'-%J-%t.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];fo=(R/'logs/Kimi-vocab-TP16-r9176-client.log').open('x');proc=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);q.update(GPU_submitted=True,GPU_args=a,GPU_client_PID=proc.pid,phase='observing_same_GPU');save();began=time.monotonic()
 while proc.poll()is None:
  row=account(LABEL)
  if row:assert q.get('GPU_handle',row[0])==row[0];q.update(GPU_handle=row[0],GPU_actual=row)
  assert time.monotonic()-began<1280;save();time.sleep(10)
 fo.close();row=account(LABEL);assert row;q.update(GPU_handle=row[0],GPU_srun_returncode=proc.returncode,GPU_terminal=terminal(LABEL,row[0]),phase='audit_post');save()
 try:
  reports=[]
  for node in [0,1]:
   for mode in ['dual','full','shard']:
    p=R/'results'/('Kimi-vocab-TP16-r9176-'+row[0]+'-node%d-'%node+mode+'.json');raw=p.read_bytes();d=json.loads(raw);assert d['completed']and d['all8_localowners_closed']and d['communication_group_destroyed']and d['DSO_SHA256']==m['native_SHA256']and d['manifest_SHA256']==q['manifest_SHA256'];assert len(d['reports'])==8;reports.append(d)
  for mode in ['dual','full','shard']:
   rs=[r for d in reports if d['mode']==mode for r in d['reports']];assert sorted(r['tp_rank']for r in rs)==list(range(16))and all(r['Runtime_owner_closed']and r['heads_destroyed_before_Runtime']for r in rs)
   if mode=='shard':assert all(not r['weight_loading']['full_GPU_storage_exists']and r['weight_loading']['actual_shard_H2D_bytes']==146800640 and r['weight_loading']['first_middle_last_row_readback_exact']for r in rs)
  for node in [0,1]:
   log=(R/'logs'/(LABEL+'-'+row[0]+'-'+str(node)+'.log')).read_text(errors='replace');assert log.count('KIMI_VOCAB_NODE_R9176_NORMAL')==3 and log.count('KIMI_VOCAB_OWNER_R9176_NORMAL')==24 and'Using network IBext_v8'in log and'Using network Socket'not in log
  q['audit_passed']=True;q['result_report_SHA256']={d['host']+'-'+d['mode']:sha(R/'results'/('Kimi-vocab-TP16-r9176-'+row[0]+'-node%d-'%d['node']+d['mode']+'.json'))for d in reports};q['audit_scope']='finite/layout/RMS/rows/localH2D/timing/closure component only; Modelquality/savings not qualified'
 except BaseException as e:q['audit_passed']=False;q['audit_error']=repr(e)
 finally:q['post_KFD']=fresh('Kimi-vocab-TP16-post-r9176');save()
 q['qualified_component']=q['GPU_terminal'][2:4]==['COMPLETED','0:0']and q.get('audit_passed',False);q['completed']=True;q['phase']='completed';save()
except BaseException as e:
 q['error']=repr(e);q['traceback']=traceback.format_exc();q['phase']='failed';save()
 if proc is not None and proc.poll()is not None and not q.get('post_KFD'):
  try:q['post_KFD']=fresh('Kimi-vocab-TP16-exception-post-r9176');save()
  except BaseException as x:q['post_error']=repr(x);save()
 raise
