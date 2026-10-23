"""Unique actualDTK CPU-only build. No GPU/Model/dlopen/retry."""
import pathlib,json,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';incoming=r/'incoming'
manifest_path=incoming/'GLM-ROWS16-CPU-manifest-r5707.json';manifest=json.loads(manifest_path.read_text())
for name,h in manifest['sources_SHA256'].items():assert hashlib.sha256((incoming/name).read_bytes()).hexdigest()==h,name
assert not(r/'candidates/glm-rows16-model-r5707').exists()
out=r/'glm-rows16-CPU-launch-r5707.json';assert not out.exists()
snapshot=subprocess.check_output(['sacct','-S','2026-10-23','-j','910406','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed,NodeList%160'],text=True)
for line in snapshot.splitlines():
 f=line.split('|')
 if'.'in f[0]and f[0].split('.')[-1].isdigit():assert f[2]not in['RUNNING','PENDING','COMPLETING','CONFIGURING'],line
label='glm-rows16-CPU-r5707'
args=['srun','--jobid=910406','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a14r2n19','--gres=dcu:0','--mem=32000M','-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=','ROCR_VISIBLE_DEVICES=','WAVE_INSIDE_R2323=GLM_rows16_CPU_r5707_inside.sh','bash',str(incoming/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'glm-rows16-CPU-intent-r5707.json').open('x')as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'manifest_SHA256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'fresh_sacct':snapshot,'state':'exclusive_pre_Popen'},f)
with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'submitted':True,'pid':p.pid,'args':args,'manifest_SHA256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'scope':'ActualDTK CPU compile/link/fresh-M/nm/readelf only, noCPUdlopen/noGPU/noModel/newmath; privatecandidate unique; noautoretry'}
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))

