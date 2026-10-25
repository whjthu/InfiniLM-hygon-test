from pathlib import Path
import json,hashlib,ast,base64
r=Path.cwd();o=r/'Deep-eos-compaction-private-r9130e';p=o/'launch_Deep_eos_compaction_r9130e.py';text=p.read_text();old="'--kill-on-bad-exit=1'";assert text.count(old)==1;text=text.replace(old,old+",'--cpu-bind=none'");p.write_bytes(text.encode())
p=o/'Deep-eos-compaction-manifest-r9130e.json';m=json.loads(p.read_bytes());m['incoming_source_SHA256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in o.iterdir()if q.suffix in ['.py','.sh']};p.write_bytes((json.dumps(m,indent=2)+'\n').encode())
files={p.name:{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()}for p in o.iterdir()if p.is_file()}
code="import pathlib,json,hashlib,base64,subprocess\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014';files="+repr(files)+"\nfor name,e in files.items():\n p=r/'incoming'/name;b=base64.b64decode(e['b64']);assert hashlib.sha256(b).hexdigest()==e['SHA256'];assert b'\\r'not in b\n if p.exists():assert p.read_bytes()==b,(name,'nooverwrite')\n else:p.write_bytes(b)\nfor name in files:\n if name.endswith('.sh'):subprocess.run(['bash','-n',str(r/'incoming'/name)],check=True)\nx=subprocess.run(['python3',str(r/'incoming/launch_Deep_eos_compaction_CPU_r9130e.py')],capture_output=True,text=True,timeout=70);assert x.returncode==0,(x.stdout,x.stderr);print(x.stdout)\n";(r/'stage_Deep_eos_CPU_r9130e.py').write_bytes(code.encode())
keeper=(r/'Deep-eos-compaction-private-r9130d/CPU_supervisor_Deep_eos_r9130d.py').read_text().replace('r9130d','r9130e')
needle="source=r/'incoming/launch_Deep_eos_compaction_r9130e.py';q=subprocess.run(['python3',str(source)],capture_output=True,text=True,timeout=90)"
insert=r'''child_env={k:v for k,v in os.environ.items()if not(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ('SLURM_CONF','SLURM_CONF_SERVER'))and k not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
probe_name='Deep-eos-affinity-r9130e';args=['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c64','-w','a02r4n05','--cpu-bind=none','--gres=none','--mem=512M','-t','00:02:00','--job-name='+probe_name,'env','WAVE_INSIDE_R2323=CPU_affinity_Deep_eos_r9130e_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];probe=subprocess.run(args,capture_output=True,text=True,timeout=145,env=child_env);state.update(probe_returncode=probe.returncode,probe_stdout=probe.stdout,probe_stderr=probe.stderr);save(state);assert probe.returncode==0
checks=[json.loads(s)for s in probe.stdout.splitlines()if s.startswith('{')];checks=[v for v in checks if v.get('check')=='8owner_CPUmask_beforeload'];assert len(checks)==1 and checks[0]['passed'];handle=checks[0]['step'];state['actual_CPU_affinity_probe']=checks[0];save(state)
for _ in range(60):
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-j',handle,'-n','-P','-o','JobID,State,ExitCode'],text=True,timeout=20,env=child_env).splitlines()];exact=[v for v in rows if v[0]==handle]
 if exact and exact[0][1]=='COMPLETED'and exact[0][2]=='0:0':state['probe_authoritative_terminal']=exact[0];save(state);break
 time.sleep(1)
else:raise RuntimeError('same affinity probe not authoritative completed; no GPU launch')
source=r/'incoming/launch_Deep_eos_compaction_r9130e.py';q=subprocess.run(['python3',str(source)],capture_output=True,text=True,timeout=90,env=child_env)'''
assert keeper.count(needle)==1;keeper=keeper.replace(needle,insert);p=o/'CPU_supervisor_Deep_eos_r9130e.py';p.write_bytes(keeper.encode());ast.parse(p.read_bytes())
probe='''#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import os,socket,json
v=sorted(os.sched_getaffinity(0));q={'check':'8owner_CPUmask_beforeload','hostname':socket.gethostname(),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'allowed_CPUs':v,'required':32,'passed':len(v)>=32,'GPU_initialized':False};print(json.dumps(q));assert q['passed']
PY
''';(o/'CPU_affinity_Deep_eos_r9130e_inside.sh').write_bytes(probe.encode());print(json.dumps({'clear_allocation_rank_and_GPU_env':True,'probe_actual_container64CPU_nogres':True,'wait_probe_terminal_before_GPU':True,'explicit_GPU_cpu_bind_none':True}))
