import pathlib,json,hashlib,subprocess,time,traceback
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');src=r/'incoming/Kimi-batch-trace-fixture-r9122b';out=r/'results/Kimi-batch-trace-fixture-r9122b-terminal.json';q={'watch_ready_before_numeric':True,'numeric_attempts':0,'post_attempts':0,'performance_round':False,'exceptions':[]};assert not out.exists()
def save():p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
save()
try:
 v=json.loads((src/'intent.json').read_bytes());assert hashlib.sha256((src/'GPU_driver.py').read_bytes()).hexdigest()==v['driver_SHA256'];assert hashlib.sha256((src/'GPU_inside.sh').read_bytes()).hexdigest()==v['inside_SHA256'];q['numeric_attempts']=1;save()
 with(r/'logs/Kimi-batch-trace-fixture-r9122b-controller.log').open('x')as log:p=subprocess.Popen(v['args'],stdout=log,stderr=subprocess.STDOUT);q['numeric_pid']=p.pid;save();q['numeric_returncode']=p.wait(timeout=180)
 for _ in range(60):
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910371','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed'],text=True).splitlines()if x.split('|')[1]=='Kimi-batch-trace-fixture-r9122b'];assert len(rows)<=1
  if rows:
   q['actual_handle']=rows[0][0];q['Slurm']=rows[0];save()
   if rows[0][2].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:q['actual_terminal']=True;break
  time.sleep(2)
 assert q.get('actual_terminal');p=r/'results'/('Kimi-batch-trace-fixture-r9122b-'+q['actual_handle']+'.json');d=json.loads(p.read_bytes());q['receipt']=d;q['audit_pass']=bool(q['numeric_returncode']==0 and rows[0][2:4]==['COMPLETED','0:0']and d['completed']and d['normal_close'])
except BaseException as e:q['exceptions'].append({'error':repr(e),'traceback':traceback.format_exc()})
finally:
 if q.get('actual_terminal'):
  try:
   q['post_attempts']=1;z=subprocess.run(['srun','--jobid=910371','--overlap','--exact','-N1','-n1','-c1','-w','a02r3n07','--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-batch-trace-post-r9122b','python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];q['post']={'returncode':z.returncode,'rows':rows,'stderr':z.stderr};q['post_pass']=z.returncode==0 and len(rows)==1 and rows[0]['passed']and not rows[0]['kfd_pids']
  except BaseException as e:q['exceptions'].append({'phase':'post','error':repr(e)})
 q['completed']=bool(q.get('audit_pass')and q.get('post_pass'));save()
print(json.dumps(q))
