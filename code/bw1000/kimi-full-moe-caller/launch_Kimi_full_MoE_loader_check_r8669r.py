import pathlib,subprocess,json,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-full-MoE-loader-CPU-r8669r';parent='910406';node='a14r2n19';p=r/(label+'-launch.json');assert not p.exists()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,NodeList%150'],text=True,timeout=30).splitlines()];assert any(v[0]==parent and v[2].split()[0]=='RUNNING'for v in rows);assert not any(v[1]==label for v in rows)
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
for v in rows:
 if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and v[2].split()[0].rstrip('+')not in terminal:assert node not in subprocess.check_output(['scontrol','show','hostnames',v[3]],text=True,timeout=20).splitlines(),('CPUbusy',v[0])
args=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c4','-w',node,'--gres=none','--mem=4096M','-t','00:05:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=check_Kimi_full_MoE_loader_r8669r_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs'/(label+'-controller.log')).open('x')as f:q=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':q.pid,'CPU_only':True,'dlopen':False,'GPU_init':False};p.write_text(json.dumps(d));print(json.dumps(d))
