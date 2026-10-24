import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';parent='910406';node='a14r2n19';label='Kimi-cyclic-EP2-CPU-r8669zp';receipt=r/'Kimi-cyclic-EP2-CPU-launch-r8669zp.json';assert not receipt.exists()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%150'],text=True,timeout=30).splitlines()]
assert any(v[0]==parent and v[2].split()[0]=='RUNNING'for v in rows)
assert not any(v[1]==label for v in rows), 'do not duplicate unknown or terminal component'
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
for row in rows:
 if row[0].startswith(parent+'.')and row[0].split('.')[-1].isdigit()and row[2].split()[0].rstrip('+')not in terminal:
  assert node not in subprocess.check_output(['scontrol','show','hostnames',row[4]],text=True,timeout=20).splitlines(),('CPUleasebusy',row[0])
proof=json.loads((r/'incoming/Kimi-cyclic-EP2-source-r8669zp.json').read_text())
for name,sha in proof['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/name).read_bytes()).hexdigest()==sha,name
args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w',node,'--gres=none','--mem=4096M','-t','00:05:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=build_Kimi_cyclic_EP2_r8669zp_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU_dlopen=False,pid=p.pid,root_lease=dict(parent=parent,node=node,cpus=4,memory_MiB=4096,limit_minutes=5),fresh_accounting_checked=True,source_SHA256=proof['source_SHA256']);receipt.write_text(json.dumps(d,indent=2));print(json.dumps(d))
