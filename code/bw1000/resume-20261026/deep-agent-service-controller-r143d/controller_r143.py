"""One owned eight-card resident P0 service. No retry or performance tuning."""
import pathlib,json,hashlib,subprocess,os,time,datetime,re,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-agent-P0-control-r143d';SRC=R/'incoming/Deep-agent-P0-r143d';LABEL='Deep-agent-P0-r143d';NODE='a02r4n01';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();wp=R/(LABEL+'-watch.json');assert not wp.exists();env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};w={'GPU_submitted':False,'actual_handle':None,'completed':False,'Native_execution_passed':False,'P0_passed':False,'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'root_GO_required':True};actual=None;proc=None
TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def save():
 w['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=wp.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(wp)
def account(label):
 rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-27','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=30).splitlines()];return [v for v in rows if v[1]==label]
def idle(suffix):
 p=R/'incoming/verify_gpu_idle_r5738.py';assert sha(p)=='45b7d14902965ca8569000a8c20882d589c7b2468f365e96d0cb552cf6f57b53'
 a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(p)];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['node']==NODE and rows[0]['kfd_pids']==[] and rows[0]['CUDA_not_initialized_by_probe'];return rows[0]
save()
try:
 m=json.loads((D/'controller-manifest-r143.json').read_bytes());g=json.loads((D/'root-GO-after-CPU.json').read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['controller_manifest_SHA256']==sha(D/'controller-manifest-r143.json')and g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 sm=SRC/'source-manifest-r143.json';assert sha(sm)==m['CPU_source_manifest_SHA256']=='88a18786f72820f3a0beef8a8026d8c6ff79e2fbba7894fdb1d2458333d3ae30';s=json.loads(sm.read_bytes())
 for n,h in s['source_SHA256'].items():assert sha(SRC/n)==h,n
 cp=R/'candidates/Deep-agent-p0-r143d/CPU-preflight.json';assert sha(cp)==g['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['static_only']and not c['Native_loaded']and c['source_manifest_SHA256']==sha(sm)and c['Native_SHA256']==m['Native_SHA256']
 assert re.fullmatch(r'910374\.\d+',c['actual_step']);assert not account(LABEL),'Existing exact label: inspect same task, never restart'
 parent=subprocess.check_output(['scontrol','show','job','910374','-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
 w.update(source_manifest_SHA256=sha(sm),CPU_receipt_SHA256=sha(cp),publication_commit=g['publication_commit'],Native_SHA256=m['Native_SHA256'],fresh_before=idle('before'),watch_ready_before_GPU=True);save()
 outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
 a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','02:00:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'env','WAVE_INSIDE_R2323=Deep-agent-P0-r143d/GPU_inside_r143.sh','bash',str(outer),'probe']
 with(R/'logs'/(LABEL+'-client.log')).open('x')as f:proc=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,client_PID=proc.pid,state='follow_one_unique_native_service');save();deadline=time.monotonic()+7400
 while time.monotonic()<deadline:
  rows=account(LABEL);assert len(rows)<=1
  if rows:
   row=rows[0];assert actual is None or actual==row[0];actual=row[0];w.update(actual_handle=actual,Slurm=row)
   if row[2].split()[0].rstrip('+')in TERM:w['terminal']=row;save();break
  save();time.sleep(10)
 else:raise TimeoutError('Observe same owned service; no automatic retry')
 w['post']=idle('post');w['post_empty']=True
 rp=R/'results'/('deep-agent-p0-r143d-'+actual+'.json');w['result_path']=str(rp)
 if rp.exists():
  q=json.loads(rp.read_bytes());w['result_SHA256']=sha(rp);w['summary']={k:q.get(k)for k in ['completed','passed','normal_server_shutdown','normal_Native_shutdown','client_returncode','client_evidence_directory','error']};w['P0_passed']=bool(q.get('passed'));w['Native_execution_passed']=bool(w['terminal'][2:4]==['COMPLETED','0:0']and q.get('normal_server_shutdown')and q.get('normal_Native_shutdown'))
 pp=R/'results'/('deep-agent-p0-r143d-'+actual+'-preweight.json')
 if pp.exists():
  q=json.loads(pp.read_bytes());w['preweight']={'path':str(pp),'SHA256':sha(pp),'passed':q.get('passed'),'error':q.get('error'),'weights_payload_opened':q.get('weights_payload_opened')}
 index_path=R/'results'/('deep-agent-p0-r143d-'+actual+'-evidence.json');assert not index_path.exists();artifacts={}
 for artifact in [rp,pp,R/'results'/('deep-agent-p0-r143d-'+actual+'-identity.json')]:
  if artifact.is_file():artifacts[str(artifact)]={'SHA256':sha(artifact),'bytes':artifact.stat().st_size}
 client_dir=R/'results'/('deep-agent-p0-r143d-'+actual+'-client')
 if client_dir.is_dir():
  for artifact in sorted(client_dir.rglob('*')):
   if artifact.is_file():artifacts[str(artifact)]={'SHA256':sha(artifact),'bytes':artifact.stat().st_size}
 index={'namespace':LABEL,'actual_step':actual,'CPU_receipt_SHA256':sha(cp),'source_manifest_SHA256':sha(sm),'Native_SHA256':m['Native_SHA256'],'original_artifact_prefix':'deep-agent-p0-r143d-'+actual,'artifacts':artifacts,'note':'Immutable backend 143c retains r143 Python filenames only. All original evidence is unique to this actual step; this b index never rewrites it.'};index_path.write_text(json.dumps(index,indent=2)+'\n');w['b_evidence_index']={'path':str(index_path),'SHA256':sha(index_path)}
 log=R/'logs'/(LABEL+'-'+actual+'.log');text=log.read_text(errors='replace');w['normal_marker_count']=text.count('DEEP_AGENT_P0_R143D_NATIVE_NORMAL');w['Native_execution_passed']=bool(w['Native_execution_passed']and w['normal_marker_count']==1);w['completed']=True;w['scope']='Real Native/text/HTTP/SSE/tool qualification; no throughput or latency optimization claim';save()
except BaseException as e:
 w.update(error=repr(e),traceback=traceback.format_exc());save()
 if proc is not None and actual is None:
  for _ in range(6):
   try:
    rows=account(LABEL)
    if len(rows)==1:actual=rows[0][0];w['actual_handle']=actual;save();break
   except BaseException:pass
   time.sleep(5)
 if proc is not None and actual:
  rows=account(LABEL)
  if len(rows)==1 and rows[0][0]==actual and rows[0][2].split()[0].rstrip('+')not in TERM:
   w['before_owned_cancel']=rows[0];save();subprocess.run(['scancel',actual],env=env,check=True,timeout=30)
   deadline=time.monotonic()+180
   while time.monotonic()<deadline:
    rows=account(LABEL)
    if rows and rows[0][0]==actual and rows[0][2].split()[0].rstrip('+')in TERM:w['terminal_after_owned_cancel']=rows[0];break
    time.sleep(5)
  try:w['error_post']=idle('errorpost');w['post_empty']=True
  except BaseException as ce:w['post_error']=repr(ce)
  save()
 raise
