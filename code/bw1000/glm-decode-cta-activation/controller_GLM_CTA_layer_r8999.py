"""Existing finite component controller pattern: one numeric srun, strict audit,
then one unconditional CPU-only KFD post. No retry/reset/cancel/rebaseline."""
import pathlib,json,hashlib,subprocess,datetime,time,traceback,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=json.loads((i/'GLM-CTA-layer-manifest-r8999.json').read_text());v=json.loads((r/'GLM-CTA-layer-r8999-intent.json').read_text());out=r/'results/GLM-CTA-layer-r8999-terminal.json';assert not out.exists()
q={'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'exceptions':[],'manifest_SHA256':hashlib.sha256((i/'GLM-CTA-layer-manifest-r8999.json').read_bytes()).hexdigest()}
def save():temp=out.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(out)
save()
try:
 with(r/'logs/GLM-CTA-layer-r8999-controller.log').open('x')as log:
  p=subprocess.Popen(v['args'],stdout=log,stderr=subprocess.STDOUT);q.update(state='numeric_running',numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait();save()
 for attempt in range(60):
  snap=subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30);rows=[a.split('|')for a in snap.splitlines()if len(a.split('|'))>2 and a.split('|')[1]=='GLM-CTA-layer-r8999'];assert len(rows)<=1,rows
  q['Slurm_rows']=rows
  if rows:
   if 'bound_handle'in q:assert q['bound_handle']==rows[0][0]
   else:q['bound_handle']=rows[0][0]
   if rows[0][2].split()[0].split('+')[0]not in['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  if not rows and q['numeric_returncode']!=0:break
  save();time.sleep(2)
 assert q.get('actual_terminal'),('actual_terminal_UNKNOWN',rows)
 job=q['bound_handle'];receipts=[]
 for rank in range(8):
  p=r/'results'/f'GLM-CTA-layer-r8999-{job}-rank{rank}.json';d=json.loads(p.read_text());assert d['completed']and d['rank']==rank and d['producer']==job and d['all_outputs_finite']and d['normal_reset_close']and d['Torch_postclose']and d['Runtime_weak_expired']and d['inputs_readonly']and d['final_activation_stats'][0]>0 and d['final_activation_stats'][1]>0 and d['final_activation_stats'][2]==0;assert len(d['trials'])==11 and[a['mode']for a in d['trials']]==[0,1,0,1,1,0,1,0,0,1,0];assert all(a['finite']and a['canary']and a['ownrepeat_bits_exact']and a['activation_delta']==([0,1,0]if a['mode']else[1,0,0])for a in d['trials']);receipts.append({'rank':rank,'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':d})
 log=r/'logs'/f'GLM-CTA-layer-r8999-{job}.log';text=log.read_text(errors='replace');errors=[a for a in text.splitlines()if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|NCCL WARN|HIP error|CUDA error|rocblas_status|out of memory',a,re.I)];q['TPmax_trials']=[{'index':j,'mode':receipts[0]['data']['trials'][j]['mode'],**{metric:max(rec['data']['trials'][j][metric]for rec in receipts)for metric in ['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms']}}for j in range(11)];q.update(receipts=receipts,all8_normal=True,audit_errors=errors,log_SHA256=hashlib.sha256(log.read_bytes()).hexdigest(),audit_pass=not errors and q['numeric_returncode']==0 and q['Slurm_rows'][0][2:4]==['COMPLETED','0:0']);save()
except BaseException as exc:q['exceptions'].append({'phase':'numeric_or_audit','error':repr(exc),'traceback':traceback.format_exc()});save()
finally:
 try:
  q['post_attempts']=1;save();a=['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w',v['selected_node'],'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-CTA-layer-post-r8999','python3',str(i/'verify_gpu_idle_r5819.py')];p=subprocess.run(a,text=True,capture_output=True,timeout=150);q['post']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  if p.returncode==0:d=json.loads(p.stdout);q['post_pass']=d['node']==v['selected_node']and d['kfd_pids']==[]
 except BaseException as exc:q['exceptions'].append({'phase':'post','error':repr(exc)})
 finally:q.update(state='terminal',qualified_component=bool(q.get('audit_pass')and q.get('post_pass')),Model_adoption=False,Model_TPS=False,completed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(q))
