"""One bounded CPU-only container, no model or runtime import."""
from pathlib import Path
import os,json,hashlib,subprocess,datetime,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');O=Path('/public/home/aclsylqidf/hygon-inference-20261013');D=R/'incoming/Kimi-long-feature-HOST-recovery-r305b';tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];scratch=R/'results'/('Kimi-long-feature-recovery-scratch-'+tag);assert not scratch.exists();scratch.mkdir();local=scratch/'local';local.mkdir()
env={k:v for k,v in os.environ.items()if k not in ['LD_PRELOAD','APPTAINERENV_LD_PRELOAD','SINGULARITYENV_LD_PRELOAD']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env[k]='';env['APPTAINERENV_'+k]=''
env['APPTAINERENV_KIMI_R305B_MANIFEST_SHA']=os.environ['KIMI_R305B_MANIFEST_SHA'];env['APPTAINERENV_PYTHONDONTWRITEBYTECODE']='1'
assert not any(env.get(k)for k in ['APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH'])
args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--home',str(O/'build-deps/root')+':/root','--bind',str(O)+':/old:ro','--bind',str(R)+':/refactor','--bind',str(R)+':'+str(R)+':ro','--bind','/opt/hyhal:/opt/hyhal:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro','--bind',str(scratch)+':/tmp','--bind',str(local)+':/local',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','/refactor/incoming/Kimi-long-feature-HOST-recovery-r305b/CPU_inside.sh']
start=time.monotonic();z=subprocess.run(args,env=env,cwd=D,capture_output=True,text=True,timeout=235);(scratch/'stdout.log').write_text(z.stdout);(scratch/'stderr.log').write_text(z.stderr)
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':tag,'CPU_only':True,'GPU':0,'Model':0,'rc':z.returncode,'elapsed_s':time.monotonic()-start,'stdout_SHA256':hashlib.sha256(z.stdout.encode()).hexdigest(),'stderr_SHA256':hashlib.sha256(z.stderr.encode()).hexdigest(),'scratch':str(scratch)}
p=R/'results'/('Kimi-long-feature-recovery-wrapper-'+tag+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q));assert z.returncode==0
