import pathlib,json,hashlib,subprocess,datetime,time,traceback
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';tag='Kimi-gate-real-r9032'
v=json.loads((r/(tag+'-intent.json')).read_text());out=r/'results'/(tag+'-terminal.json');assert not out.exists()
q={'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'exceptions':[],'Model_qualified':False}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
save()
try:
 with(r/'logs'/(tag+'-controller.log')).open('x')as f:
  p=subprocess.Popen(v['args'],stdout=f,stderr=subprocess.STDOUT);q.update(state='numeric_running',numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait();save()
 for _ in range(60):
  snap=subprocess.check_output(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20)
  rows=[s.split('|')for s in snap.splitlines()if len(s.split('|'))>5 and s.split('|')[1]==tag];assert len(rows)<=1;q['Slurm_rows']=rows
  if rows:
   if 'bound_handle'in q:assert q['bound_handle']==rows[0][0]
   else:q['bound_handle']=rows[0][0]
   if rows[0][2].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  if not rows and q['numeric_returncode']!=0:break
  save();time.sleep(2)
 assert q.get('actual_terminal');path=r/'results'/f'{tag}-{q["bound_handle"]}.json';d=json.loads(path.read_text());q['data']=d
 assert d['status']=='GATE_PROJECTION_COMPONENT_DIAGNOSTIC_COMPLETE'and d['normal_close']and d['counts_all']==[16,10,96]
 assert len(d['contexts'])==4 and all(e['closed_normally']and all(e[k]for k in ['ownrepeat','finite','canary','host_input_readonly','GPU_weight_readonly'])for e in d['contexts'].values())
 m=json.loads((i/'Kimi-gate-manifest-r9032.json').read_text());actual=d['actual_runtime_dependencies_SHA256']
 for suffix,h in m['runtime_dependencies_SHA256'].items():assert len([p for p in actual if p.endswith(suffix)and actual[p]==h])==1,(suffix,actual)
 q.update(component_audit_pass=q['numeric_returncode']==0 and rows[0][2:4]==['COMPLETED','0:0'],receipt=str(path),receipt_SHA256=hashlib.sha256(path.read_bytes()).hexdigest());save()
except BaseException as e:q['exceptions'].append({'phase':'numeric_or_audit','error':repr(e),'traceback':traceback.format_exc()});save()
finally:
 try:
  q['post_attempts']=1;save();args=['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w',v['selected_node'],'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-gate-post-r9032','python3',str(i/'verify_gpu_idle_r5819.py')];p=subprocess.run(args,text=True,capture_output=True,timeout=150);q['post']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  if p.returncode==0:d=json.loads(p.stdout);q['post_pass']=d['node']==v['selected_node']and d['kfd_pids']==[]
 except BaseException as e:q['exceptions'].append({'phase':'post','error':repr(e)})
 q.update(state='terminal',completed=True,component_diagnostic_complete=bool(q.get('component_audit_pass')and q.get('post_pass')),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(q))
