"""One missing post check after authoritative original terminal; never relaunch."""
import datetime,json,pathlib,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
target=r/'results/GLM-spawn-trace-r8690-root-post.json'
intent=r/'results/GLM-spawn-trace-r8690-root-post-intent.json'
assert not target.exists()and not intent.exists(),'Followexistingpost; neverduplicate'
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-24','-j','910376.504','-n','-P','-o','JobID,JobName%100,State,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()]
assert len(rows)==1 and rows[0][:2]==['910376.504','GLM-spawn-trace-r8690']and rows[0][2]in['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'],rows
with intent.open('x')as f:json.dump(dict(original_terminal=rows[0],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_GPU=False,no_replay=True),f)
p=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n14','--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-spawn-trace-r8690-root-post','python3',str(r/'incoming/verify_gpu_idle_r5819.py')],capture_output=True,text=True,timeout=150)
observations=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')]
q=dict(original_terminal=rows[0],post_returncode=p.returncode,post=observations,stderr=p.stderr[-2000:],new_GPU=False,no_replay=True,original_failure='Parenthipprof returned0 despite childtrace failure; accounting wasmomentarilynonterminal, originalcontroller lacked finallypost.',raw_model_weights='NONE')
q['KFD_empty']=p.returncode==0 and len(observations)==1 and observations[0]['node']=='a02r4n14'and observations[0]['kfd_pids']==[]
with target.open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
