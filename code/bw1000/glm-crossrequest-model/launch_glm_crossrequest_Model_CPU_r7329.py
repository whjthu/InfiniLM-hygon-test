import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/glm-crossrequest-CPU-launch-r7329.json';assert not out.exists()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j','910374','-n','-P','-o','JobID,State,NodeList%100'],text=True).splitlines()]
for v in rows:
 if '.'in v[0]and v[0].split('.')[-1].isdigit()and v[1]in['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED']:assert'a02r4n05'not in subprocess.check_output(['scontrol','show','hostnames',v[2]],text=True).splitlines(),v
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a02r4n05','--gres=none','--mem=32000M','-t','00:10:00','--job-name=GLM-crossrequest-Model-CPU-r7328','--output='+str(r/'logs/glm-crossrequest-Model-CPU-r7328-%J.log'),'bash',str(r/'incoming/glm_crossrequest_Model_CPU_r7328_outer.sh')]
with out.open('x')as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'CPU_only':True,'no_resubmit':True},f)
with(r/'logs/glm-crossrequest-Model-CPU-r7328-controller.log').open('x')as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'CPU_only':True,'no_resubmit':True}))
