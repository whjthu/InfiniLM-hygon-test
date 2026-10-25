import pathlib,json,subprocess,datetime,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';parent='910406';node='a14r2n19';record=r/'results/Kimi-A8-ABI-CPU-r8993-launch.json';assert not record.exists()
s=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=20);assert 'JobState=RUNNING'in s
assert node in subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?<!\S)NodeList=(\S+)',s).group(1)],text=True).splitlines()
rows=subprocess.check_output(['sacct','-S','2026-10-25','-j',parent,'-n','-P','-o','JobID,State%30,NodeList%80'],text=True).splitlines();busy=[]
for line in rows:
 z=line.split('|')
 if z[0].startswith(parent+'.')and z[0].split('.')[-1].isdigit()and z[1].split()[0]in['RUNNING','PENDING','COMPLETING']:
  if node in subprocess.check_output(['scontrol','show','hostnames',z[2]],text=True).splitlines():busy.append(z[0])
assert not busy,busy
args=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c4','-w',node,'--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-A8-ABI-CPU-r8993','--output='+str(r/'logs/Kimi-A8-ABI-CPU-r8993-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_A8_ABI_CPU_r8993_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with record.open('x')as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),args=args,scope='one GPU TU+one host TU,noGPU/noDLopen'),f,indent=2)
with(r/'logs/Kimi-A8-ABI-CPU-r8993-controller.log').open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'controller_pid':p.pid,'record':str(record)}))
