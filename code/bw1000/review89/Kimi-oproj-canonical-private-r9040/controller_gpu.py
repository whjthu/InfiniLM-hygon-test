import pathlib,json,hashlib,subprocess,datetime,time,traceback
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';tag='Kimi-layer0-real-r9040';v=json.loads((r/(tag+'-intent.json')).read_text());out=r/'results'/(tag+'-terminal.json');assert not out.exists();m=json.loads((i/'Kimi-layer0-manifest-r9040.json').read_text());q={'state':'watchready','watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'exceptions':[],'Model_qualified':False}
def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
save()
try:
 with(r/'logs'/(tag+'-controller.log')).open('x')as f:
  p=subprocess.Popen(v['args'],stdout=f,stderr=subprocess.STDOUT);q.update(state='numeric_running',numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait(timeout=650);save()
 for _ in range(60):
  snap=subprocess.check_output(['sacct','-S','2026-10-25','-j','910375','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True,timeout=20);rows=[s.split('|')for s in snap.splitlines()if len(s.split('|'))>5 and s.split('|')[1]==tag];assert len(rows)<=1;q['Slurm_rows']=rows
  if rows:
   if 'bound_handle'in q:assert q['bound_handle']==rows[0][0]
   else:q['bound_handle']=rows[0][0]
   if rows[0][2].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  if not rows and q['numeric_returncode']!=0:break
  save();time.sleep(2)
 assert q.get('actual_terminal');q['receipts']=[]
 for node in range(2):
  path=r/'results'/f'{tag}-{q["bound_handle"]}-node{node}.json';d=json.loads(path.read_text());q['receipts'].append({'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':d['status'],'normal_close':d.get('normal_close'),'native_calls_thisnode':d.get('native_calls_thisnode')});assert d['status']=='OPROJ_CANONICAL_COMPONENT_DIAGNOSTIC_COMPLETE'and d['normal_close']and d['native_calls_thisnode']==2760;assert all(z['normal_close']and 'error'not in z for z in d['workers']);assert all(d[k]for k in ['finite','canary','host_input_readonly','trace_finite'])
  for suffix,h in m['runtime_dependencies_SHA256'].items():assert d['actual_runtime_dependencies_SHA256'].get(suffix)==h,(suffix,d['actual_runtime_dependencies_SHA256'])
 q['component_audit_pass']=q['numeric_returncode']==0 and rows[0][2:4]==['COMPLETED','0:0'];save()
except BaseException as e:q['exceptions'].append({'phase':'numeric_or_audit','error':repr(e),'traceback':traceback.format_exc()});save()
finally:
 try:
  q['post_attempts']=1;save();args=['srun','--jobid=910375','--overlap','--exact','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(m['nodes']),'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-layer0-post-r9040','python3',str(i/'Kimi_Q0_idle_r8066.py')];p=subprocess.run(args,text=True,capture_output=True,timeout=160);q['post']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};records=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];q['post_pass']=p.returncode==0 and len(records)==2 and {x['node']for x in records}==set(m['nodes'])and all(x['passed']and not x['kfd_pids']for x in records)
 except BaseException as e:q['exceptions'].append({'phase':'post','error':repr(e)})
 q.update(state='terminal',completed=True,component_diagnostic_complete=bool(q.get('component_audit_pass')and q.get('post_pass')),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(q))
