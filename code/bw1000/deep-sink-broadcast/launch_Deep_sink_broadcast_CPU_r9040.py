import pathlib,json,subprocess,datetime,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';record=r/'results/Deep-sink-broadcast-CPU-r9040b-launch.json';assert not record.exists();selected=None
for parent,node in [('910407','a14r2n20')]:
 s=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=20)
 if 'JobState=RUNNING'not in s:continue
 if node not in subprocess.check_output(['scontrol','show','hostnames',re.search(r'(?<!\S)NodeList=(\S+)',s).group(1)],text=True,timeout=20).splitlines():continue
 a=subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,State%30,NodeList%80'],text=True,timeout=30);busy=[]
 for line in a.splitlines():
  z=line.split('|')
  if z[0].startswith(parent+'.')and z[0].split('.')[-1].isdigit()and z[1].split()[0]in['RUNNING','PENDING','COMPLETING']:
   if node in subprocess.check_output(['scontrol','show','hostnames',z[2]],text=True,timeout=20).splitlines():busy.append(z[0])
 if not busy:selected=(parent,node);break
assert selected,'No authorized idle CPU-only node';parent,node=selected
args=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c4','-w',node,'--gres=none','--mem=8192M','-t','00:03:00','--job-name=Deep-sink-broadcast-CPU-r9040b','--output='+str(r/'logs/Deep-sink-broadcast-CPU-r9040b-%J.log'),'env','WAVE_INSIDE_R2323=build_Deep_sink_broadcast_r9040_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with record.open('x')as f:json.dump(dict(state='exclusive_CPU_intent',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=parent,node=node,scope='4CPU/8GiB/3min,1hostTU+link,noGPU/dlopen',args=args),f,indent=2)
with(r/'logs/Deep-sink-broadcast-CPU-r9040b-controller.log').open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,record=str(record),parent=parent,node=node,jobname='Deep-sink-broadcast-CPU-r9040b')))
