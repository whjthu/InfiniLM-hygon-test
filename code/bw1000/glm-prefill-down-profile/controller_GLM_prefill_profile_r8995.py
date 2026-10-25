"""One finite numeric invocation followed by unconditional CPU KFD post."""
import pathlib,json,hashlib,subprocess,datetime,time,traceback,re,statistics
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';v=json.loads((r/'GLM-prefill-profile-r8995-intent.json').read_text());out=r/'results/GLM-prefill-profile-r8995-terminal.json';assert not out.exists();q={'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'exceptions':[],'original_hard_seconds':360,'original_inner_seconds':300,'Model_adoption':False}
def save():tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(out)
save()
try:
 with(r/'logs/GLM-prefill-profile-r8995-controller.log').open('x')as log:
  p=subprocess.Popen(v['args'],stdout=log,stderr=subprocess.STDOUT);q.update(state='numeric_running',numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait();save()
 for attempt in range(60):
  snap=subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20);rows=[a.split('|')for a in snap.splitlines()if len(a.split('|'))>5 and a.split('|')[1]=='GLM-prefill-profile-r8995'];assert len(rows)<=1; q['Slurm_rows']=rows
  if rows:
   if 'bound_handle'in q:assert q['bound_handle']==rows[0][0]
   else:q['bound_handle']=rows[0][0]
   if rows[0][2].split()[0].split('+')[0]not in['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  if not rows and q['numeric_returncode']!=0:break
  save();time.sleep(2)
 assert q.get('actual_terminal'),('terminalUNKNOWN',rows);job=q['bound_handle'];receipts=[]
 for rank in range(8):
  p=r/'results'/f'GLM-prefill-profile-r8995-{job}-rank{rank}.json';d=json.loads(p.read_text());assert d['completed']and d['rank']==rank and d['producer']==job and d['normal_stream_drain']and d['observer_closed']and d['Torch_postclose']and d['inputs_readonly'];assert d['T']==512 and d['routes']==4096 and d['layer']==3;assert len(d['trials'])==11 and[a['mode']for a in d['trials']]==[0,1,0,1,1,0,1,0,0,1,0];assert all(a['finite']and a['canary']and a['ownrepeat_exact_diagnostic']and a['branch_delta']==[1,0,1,0,0]and a['certificate_delta'][2]>0 and a['buckets_recorded']==(len(d['bucket_geometry'])if a['mode']else 0)for a in d['trials']);receipts.append({'rank':rank,'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':d})
 q['TPmax_trials']=[{'index':j,'mode':receipts[0]['data']['trials'][j]['mode'],'fullwall_ms':max(x['data']['trials'][j]['complete_provider_wall_ms']for x in receipts)}for j in range(11)];q['TPmax_median_ms']={str(mode):statistics.median(t['fullwall_ms']for t in q['TPmax_trials'][2:10]if t['mode']==mode)for mode in [0,1]};q['observer_wall_delta_percent']=100*(q['TPmax_median_ms']['1']/q['TPmax_median_ms']['0']-1)
 log=r/'logs'/f'GLM-prefill-profile-r8995-{job}.log';text=log.read_text(errors='replace');errors=[a for a in text.splitlines()if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|NCCL WARN|HIP error|CUDA error|rocblas_status|out of memory',a,re.I)];q.update(receipts=receipts,all8_normal=True,audit_errors=errors,log_SHA256=hashlib.sha256(log.read_bytes()).hexdigest(),audit_pass=not errors and q['numeric_returncode']==0 and q['Slurm_rows'][0][2:4]==['COMPLETED','0:0']);save()
except BaseException as exc:q['exceptions'].append({'phase':'numeric_or_audit','error':repr(exc),'traceback':traceback.format_exc()});save()
finally:
 try:
  q['post_attempts']=1;save();a=['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w',v['selected_node'],'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-prefill-post-r8995','python3',str(i/'verify_gpu_idle_r5819.py')];p=subprocess.run(a,text=True,capture_output=True,timeout=150);q['post']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  if p.returncode==0:d=json.loads(p.stdout);q['post_pass']=d['node']==v['selected_node']and d['kfd_pids']==[]
 except BaseException as exc:q['exceptions'].append({'phase':'post','error':repr(exc)})
 finally:q.update(state='terminal',completed=True,diagnostic_qualified=bool(q.get('audit_pass')and q.get('post_pass')),utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(q))
