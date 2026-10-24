import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/Deep-HC-RMS-CPU-launch-r7375.json';assert not out.exists()
rows=[l.split('|')for l in subprocess.check_output(['sacct','-S','2026-10-23','-j','910374','-n','-P','-o','JobID,JobName%70,State,NodeList%120'],text=True).splitlines()]
assert not any(len(v)>1 and v[1]=='Deep-HC-RMS-CPU-r7375'for v in rows)
for row in rows:
 if '.'in row[0]and row[0].split('.')[-1].isdigit()and row[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED']:assert'a02r4n05'not in subprocess.check_output(['scontrol','show','hostnames',row[3]],text=True).splitlines(),row
args=['srun','--jobid=910374','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a02r4n05','--gres=none','--mem=4096M','-t','00:05:00','--job-name=Deep-HC-RMS-CPU-r7375','--output='+str(r/'logs/Deep-HC-RMS-CPU-r7375-%J.log'),'bash',str(r/'incoming/Deep_HC_RMS_CPU_r7375_outer.sh')]
with out.open('x')as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'CPU_only':True,'no_resubmit':True},f)
with(r/'logs/Deep-HC-RMS-CPU-r7375-controller.log').open('x')as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'CPU_only':True,'no_resubmit':True}))
