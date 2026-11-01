from pathlib import Path
import base64,hashlib,json
D=Path(__file__).resolve().parent;L=D.parents[3];audit=(D/'audit_cost.py').read_bytes()
probe='''from pathlib import Path
import os,json,socket,sys
run=sys.argv[1];expected=json.loads(sys.argv[2]);node=socket.gethostname().split('.')[0]
k=Path('/sys/class/kfd/kfd/proc');assert k.is_dir();kfd=sorted(int(x.name)for x in k.iterdir()if x.name.isdigit())
pattern=b'/GLM-two-cohort-Model-r355/owner/'
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit()or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes()
 except(OSError,PermissionError):continue
 if pattern in cmd and(b'model_entry_r290.py'in cmd or b'client_r355.py'in cmd):owned.append(int(p.name))
alive=[pid for pid in expected if Path('/proc',str(pid)).exists()]
with socket.socket()as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:s.bind(('127.0.0.1',43843));free=True
 except OSError:free=False
print(json.dumps({'CPU_only':True,'GPU_initialized':False,'node':node,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'expected_owner_PIDs':expected,'expected_PIDs_still_exist':alive,'matching_owner_or_client_PIDs':owned,'KFD_pids':kfd,'KFD_empty':not kfd,'port43843_bindable':free,'passed':not alive and not owned and not kfd and free}))
'''
script="""from pathlib import Path
import json,hashlib,base64,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='GLM-C2M1-r355-a';watch=R/'results'/('GLM-r355-watch-'+run+'.json');raw=watch.read_bytes();w=json.loads(raw)
assert w['completed']and w['passed']
directory=R/'incoming/GLM-r355-terminal-audit';assert not directory.exists();directory.mkdir()
(directory/'audit_cost.py').write_bytes(base64.b64decode(AUDIT));assert hashlib.sha256((directory/'audit_cost.py').read_bytes()).hexdigest()==AUDIT_SHA
(directory/'post_probe.py').write_text(PROBE)
paths=[R/'results'/('GLM-r355-'+run+'-stage%d-rank%d.json'%(s,r))for s in range(4)for r in range(8)]
for row,p in zip(w['evidence']['owner_reports'],paths):assert row['path']==str(p)and hashlib.sha256(p.read_bytes()).hexdigest()==row['SHA256']
output=R/'results'/('GLM-r355-final-cost-'+run+'.json');assert not output.exists()
cmd=['python3','-B',str(directory/'audit_cost.py'),'--owner-jsons',*[str(p)for p in paths],'--output',str(output)]
p=subprocess.run(cmd,text=True,capture_output=True,timeout=90);assert p.returncode==0,p.stderr
owners=[json.loads(p.read_bytes())for p in paths]
cost=json.loads(output.read_bytes());post=[]
for stage,(parent,node)in enumerate([(910371,'a02r3n03'),(910371,'a02r3n07'),(910374,'a02r4n04'),(910374,'a02r4n05')]):
 expected=[]
 for rank in range(8):
  phase=R/'results'/('GLM-Model-phase-r9183-'+run+'-stage%d-rank%d.jsonl'%(stage,rank));expected.append(json.loads(phase.read_bytes().splitlines()[-1])['PID'])
 args=['srun','--jobid='+str(parent),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-r355-independent-post-s%d'%stage,'python3','-B',str(directory/'post_probe.py'),run,json.dumps(expected)]
 p=subprocess.run(args,text=True,capture_output=True,timeout=150);assert p.returncode==0,p.stderr
 rows=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];assert len(rows)==1 and rows[0]['node']==node;post.append(rows[0])
account=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407.689,910371.649,910371.650,910374.1265,910374.1266','-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
summary={'watch':w,'watch_SHA256':hashlib.sha256(raw).hexdigest(),'accounting':account,'cost':cost,'cost_path':str(output),'cost_SHA256':hashlib.sha256(output.read_bytes()).hexdigest(),'independent_post':post,'all_independent_post_passed':all(q['passed']for q in post),'minimum_observed_driver_free_bytes_all32':min(o['minimum_driver_free_bytes']for o in owners),'memory_is_peak':False,'owner_count':len(owners),'all32_normal_destroy':all(o['observer_native_destroy_returned0']and o['worker_shutdown_completed']and o['post_close_Torch_liveness']for o in owners),'quality_distinct_workloads':2,'quality_repeated_pairs':6,'quality_HTTP_total':12,'no_online_SLO_or_formal_benchmark_qualification':True}
sealed=R/'results'/('GLM-r355-final-audit-'+run+'.json')
with sealed.open('x')as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
""".replace('AUDIT_SHA',repr(hashlib.sha256(audit).hexdigest())).replace('AUDIT',repr(base64.b64encode(audit).decode())).replace('PROBE',repr(probe))
(L/'remote355-terminal-audit.py').write_text(script,encoding='utf8')
