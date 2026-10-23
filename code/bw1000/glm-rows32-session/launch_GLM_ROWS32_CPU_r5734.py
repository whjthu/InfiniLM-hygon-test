"""Unique bounded CPU-only actual-DTK build; never dlopen or GPU."""
import pathlib,json,subprocess,datetime,hashlib,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';incoming=r/'incoming';mp=incoming/'GLM-ROWS32-CPU-manifest-r5734.json';m=json.loads(mp.read_text())
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((incoming/n).read_bytes()).hexdigest()==h,n
assert not(r/'candidates/glm-rows32-model-r5734').exists()
for n in['glm-rows32-CPU-launch-r5734.json','glm-rows32-CPU-intent-r5734.json']:assert not(r/n).exists(),n
job=subprocess.check_output(['scontrol','show','job','910406','-o'],text=True);assert 'JobState=RUNNING'in job and re.search(r'\bNodeList=a14r2n19\b',job)
snapshot=subprocess.check_output(['sacct','-S','2026-10-23','-j','910406','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed,AllocCPUS,NodeList%80'],text=True)
for line in snapshot.splitlines():
 f=line.split('|')
 if re.fullmatch(r'910406\.\d+',f[0]):assert f[2]not in['RUNNING','PENDING','COMPLETING','CONFIGURING'],line
live=subprocess.check_output(['scontrol','show','step','910406','-o'],text=True)
for line in live.splitlines():
 hit=re.search(r'\bStepId=910406\.(\d+)\b',line)
 assert not hit,line
label='glm-rows32-CPU-r5734';args=['srun','--jobid=910406','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a14r2n19','--gres=none','--mem=32000M','-t','00:10:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=','ROCR_VISIBLE_DEVICES=','WAVE_INSIDE_R2323=GLM_ROWS32_CPU_r5734_inside.sh','bash',str(incoming/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'glm-rows32-CPU-intent-r5734.json').open('x')as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'fresh_parent':job,'fresh_sacct':snapshot,'fresh_live_steps':live,'state':'exclusive_pre_Popen'},f)
with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'submitted':True,'pid':p.pid,'manifest_SHA256':hashlib.sha256(mp.read_bytes()).hexdigest(),'scope':'One private actualDTK CPU compile/link, noGPU/Model/dlopen/retry'};(r/'glm-rows32-CPU-launch-r5734.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
