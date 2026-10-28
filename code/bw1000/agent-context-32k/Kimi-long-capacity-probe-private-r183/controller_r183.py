"""One publication-bound, owned32-card no-payload diagnostic. No retries.
Host CPU keeper; the GPU program runs inside the existing outer container.
"""
import argparse,os,json,pathlib,subprocess,time,datetime,socket,secrets
from contract_r183 import ROOT,INC,LABEL,NODES,PARENT,gate,clean_env,sha
ACTIVE={'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'}
norm=lambda s:s.split()[0].rstrip('+')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);ap.add_argument('--go-file',required=True);a=ap.parse_args();m,go,pins=gate(a.manifest_sha,a.go_file)
 assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20'
 env=clean_env();wp=ROOT/'results'/(LABEL+'-watch.json');intent=ROOT/'results'/(LABEL+'-intent.json');assert not wp.exists()and not intent.exists();deadline=time.monotonic()+1320;known=None;terminal=None;proc=None;post_done=False;nonce=None
 w={'completed':False,'qualified_capacity_diagnostic':False,'GPU_submitted':False,'retries':0,'manifest_SHA256':a.manifest_sha,'CPU_receipt_SHA256':pins['CPU_receipt_SHA256'],'Native_SHA256':pins['Native_SHA256'],'publication_commit':go['publication_commit'],'Model_GO':False,'weight_payload_opened':False,'model_forward_calls':0,'keeper_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
 def save():
  w['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=wp.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2)+'\n');p.replace(wp)
 def account(name):
  z=subprocess.check_output(['sacct','-S','2026-10-28','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=25);rows=[line.split('|')for line in z.splitlines()if len(line.split('|'))>=5 and line.split('|')[1]==name];assert len(rows)<=1;return rows[0]if rows else None
 def wait_terminal(name,handle,limit=60):
  stop=min(deadline,time.monotonic()+limit)
  while time.monotonic()<stop:
   row=account(name)
   if row:assert row[0]==handle
   if row and norm(row[2])not in ACTIVE:return row
   time.sleep(2)
  raise TimeoutError('same owned handle remains unknown; no retry')
 def idle(tag):
  name=LABEL+'-'+tag;args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+name,'python3','-B',str(INC/'idle_r183.py'),str(m['master_port']),tag]
  z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];w[tag+'_probe']={'rows':rows,'returncode':z.returncode};save();assert z.returncode==0 and len(rows)==4 and {q['node']for q in rows}==set(NODES)and all(q['passed']and not q['kfd_pids']for q in rows)
  rows.sort(key=lambda q:NODES.index(q['node']));assert len({q['step']for q in rows})==1;term=wait_terminal(name,rows[0]['step']);assert norm(term[2])=='COMPLETED'and term[3]=='0:0';return {'rows':rows,'terminal':term,'completed_monotonic':time.monotonic()}
 with intent.open('x')as f:json.dump({'source_manifest_SHA256':a.manifest_sha,'publication_commit':go['publication_commit'],'one_GPU_step_only':True,'retries':0},f)
 save()
 try:
  parent=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
  spec=next(x.split('=',1)[1]for x in parent.split()if x.startswith('NodeList='));owned=subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=25).splitlines();assert set(NODES)<=set(owned);w['parent_owned_nodes_verified']=True
  assert account(LABEL)is None;fresh=idle('before');w['fresh_before']=fresh;head=fresh['rows'][0]['ib0_ipv4'];nonce=secrets.token_hex(24);run='r183-'+secrets.token_hex(6);assert time.monotonic()-fresh['completed_monotonic']<=30
  exported={'GLOBAL_MODEL_MODE_R2354':'shca','GLOBAL_OFFSET_R2354':'0','GLOBAL_RUN_R2354':run,'GLOBAL_PORT_R2354':str(m['master_port']),'KIMI_ENGINE_HEAD':head,'KIMI_ENGINE_NONCE':nonce,'KIMI_TP_NODES':'2','KIMI_PP_SIZE':'2','KIMI_DRAFT_CARD_COUNT_R9142':'0','KIMI_R183_MANIFEST_SHA':a.manifest_sha,'WAVE_INSIDE_R2323':'Kimi-long-capacity-probe-r183/GPU_inside_r183.sh'}
  args=['srun','--jobid='+PARENT,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:15:00','--job-name='+LABEL,'--output='+str(ROOT/'logs'/(LABEL+'-%J-%t.log')),'--export=ALL,'+','.join(k+'='+v for k,v in exported.items()),'bash',str(ROOT/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  with(ROOT/'logs'/(LABEL+'-GPU-client.log')).open('x')as f:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,owned_srun_PID=proc.pid,command_scope='one new owned32-card step; argv/nonce never persisted');save();exited_at=None
  while time.monotonic()<deadline-170:
   row=account(LABEL)
   if row:
    assert known in [None,row[0]];known=row[0];w.update(actual_handle=known,Slurm=row);save()
    if norm(row[2])not in ACTIVE:terminal=row;w['terminal']=row;break
   elif proc.poll()is not None:
    if exited_at is None:exited_at=time.monotonic()
    elif time.monotonic()-exited_at>60:raise RuntimeError('srun exited; handle unknown; follow same label, no retry')
   time.sleep(10)
  if terminal is None:raise TimeoutError('bounded owned GPU observation')
  proc.wait(timeout=30);w['srun_returncode']=proc.returncode;post=idle('post');post_done=True;w['post_KFD_empty']=True;w['post_KFD']=post;w['nodes']=[];all_ok=norm(terminal[2])=='COMPLETED'and terminal[3]=='0:0'and proc.returncode==0
  for node in range(4):
   path=ROOT/'results'/(LABEL+'-'+known+'-node%d.json'%node);q=json.loads(path.read_bytes());log=ROOT/'logs'/(LABEL+'-'+known+'-'+str(node)+'.log');text=log.read_text(errors='replace');normal=q.get('all8_owned_workers_shutdown_completed')and q.get('all32_normal_shutdown')and text.count('KIMI_LONG_CAPACITY_R183_NORMAL')==1
   valid=q.get('passed')and q['CPU_receipt_SHA256']==pins['CPU_receipt_SHA256']and q['Native_SHA256']==pins['Native_SHA256']and q['source_manifest_SHA256']==a.manifest_sha and q['reset_call_count']==2 and q['actual_cache_constructed_after_first_reset']and not q['weight_payload_opened']and q['finish_calls']==q['model_forward_calls']==0
   network='Using network IBext_v8'in text and 'Using network Socket'not in text
   w['nodes'].append({'node':node,'path':str(path),'SHA256':sha(path),'log_SHA256':sha(log),'normal8':bool(normal),'diagnostic_passed':bool(valid),'IBext_noSocket':network,'phases':q.get('phases',{}),'reset_calls':q.get('reset_call_count'),'first_reset_constructed_cache':q.get('actual_cache_constructed_after_first_reset')});all_ok=bool(all_ok and normal and valid and network)
  w.update(completed=True,qualified_capacity_diagnostic=all_ok,all32_normal_shutdown=all(q['normal8']for q in w['nodes']),exact_transient_peak='UNKNOWN',Native_pool_reserved='UNKNOWN',full_Model_fit='UNKNOWN_STOP',transport_scope='actual Native communicator initialization IBext/noSocket; no model data/GDR/bandwidth claim');save();assert all_ok
 except BaseException as error:
  w.update(error_type=type(error).__name__,state='failed_before_GPU'if proc is None else'failed_or_unknown_owned_step',actual_handle=known);save()
  if known and terminal is None:
   row=account(LABEL)
   if row and row[0]==known and norm(row[2])in ACTIVE:
    subprocess.run(['scancel','--signal=TERM',known],env=env,check=True,timeout=25);terminal=wait_terminal(LABEL,known,90);w['terminal']=terminal;save()
  if terminal and not post_done:
   try:w['error_post_KFD']=idle('errorpost');w['post_KFD_empty']=True
   except BaseException as cleanup_error:w['post_error_type']=type(cleanup_error).__name__
   save()
  raise RuntimeError('r183 owned diagnostic failed: '+type(error).__name__+'; inspect exact watch/owned handle')from None
if __name__=='__main__':main()
