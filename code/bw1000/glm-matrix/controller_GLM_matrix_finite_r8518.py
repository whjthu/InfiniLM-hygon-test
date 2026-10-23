import pathlib,json,hashlib,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((i/'GLM-matrix-GPU-manifest-r8518.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
intent=json.loads((r/'glm-matrix-intent-r8518.json').read_text());out=r/'glm-matrix-terminal-r8518.json';assert not out.exists()
q={'completed':False,'ready_before_numeric':True,'numeric_attempts':0,'postKFD_attempts':0,'exceptions':[]}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
save()
try:
 with(r/'logs/glm-matrix-finite-r8518-controller.log').open('x')as f:
  p=subprocess.Popen(intent['args'],stdout=f,stderr=subprocess.STDOUT);q.update(numeric_attempts=1,numeric_pid=p.pid);save();q['numeric_returncode']=p.wait();save()
except BaseException as exc:q['exceptions'].append({'phase':'numeric','error':repr(exc),'traceback':traceback.format_exc()})
finally:
 try:
  q['postKFD_attempts']=1;save()
  args=['srun','--jobid=910379','--overlap','--exact','-N1','-n1','-c1','-w','a14r2n06','--gres=dcu:8','--mem=1000M','-t','00:02:00','--job-name=glm-matrix-postKFD-r8518','python3',str(i/'verify_gpu_idle_r5738.py')]
  p=subprocess.run(args,capture_output=True,text=True,timeout=150);q['postKFD']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};q['postKFD_passed']=False
  if p.returncode==0:
   kd=json.loads(p.stdout);q['postKFD_passed']=kd['node']=='a14r2n06'and kd['kfd_pids']==[]
 except BaseException as exc:q['exceptions'].append({'phase':'postKFD','error':repr(exc)});q['postKFD_passed']=False
 finally:save()
try:
 snap=subprocess.check_output(['sacct','-S','2026-10-24','-j','910379','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed'],text=True,timeout=40);q['Slurm_snapshot']=snap
 lines=[l for l in snap.splitlines()if l.split('|')[1]=='glm-matrix-finite-r8518'];q['numeric_Slurm']=lines
 # Parent wait is authoritative termination; adoption additionally needs accounting.
 if q.get('numeric_returncode')==0 and len(lines)==1 and lines[0].split('|')[2]=='COMPLETED'and lines[0].split('|')[3]=='0:0'and q.get('postKFD_passed'):
  step=lines[0].split('|')[0];p=subprocess.run(['python3',str(i/'audit_GLM_matrix_finite_r8518.py'),str(r),step],capture_output=True,text=True,timeout=60);q['audit']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};q['component_passed']=p.returncode==0
except BaseException as exc:q['exceptions'].append({'phase':'accounting_audit','error':repr(exc)})
q.update(completed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();print(json.dumps(q))
