import pathlib,json,subprocess,datetime,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';parent='910407';node='a14r2n20';record=r/'results/Deep-quant1-CPU-r8883-launch.json';assert not record.exists();assert not(r/'candidates/deep-quant1-r8883').exists()
s=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=20);assert 'JobState=RUNNING'in s and node in subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?<!\S)NodeList=(\S+)',s).group(1)],text=True,timeout=20).splitlines()
a=subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,State%30,NodeList%80'],text=True,timeout=30)
for line in a.splitlines():
 z=line.split('|')
 if z[0].startswith(parent+'.')and z[0].split('.')[-1].isdigit() and z[1].split()[0]in['RUNNING','PENDING','COMPLETING']:
  nodes=subprocess.check_output(['scontrol','show','hostnames',z[2]],text=True,timeout=20).splitlines();assert node not in nodes,('active_target_step',z)
args=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c4','-w',node,'--gres=none','--mem=4096M','-t','00:05:00','--job-name=Deep-quant1-CPU-r8883','--output='+str(r/'logs/Deep-quant1-CPU-r8883-%J.log'),'env','WAVE_INSIDE_R2323=build_Deep_quant1_r8883_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with record.open('x')as f:json.dump(dict(state='exclusive_intent',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='CPU4/4GiB/noGPU/dlopen/weights',args=args),f,indent=2)
with(r/'logs/Deep-quant1-CPU-r8883-controller.log').open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,record=str(record),jobname='Deep-quant1-CPU-r8883')))
