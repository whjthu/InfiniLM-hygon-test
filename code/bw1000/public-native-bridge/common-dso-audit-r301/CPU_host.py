from pathlib import Path
import subprocess,os,json,hashlib,datetime,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');OLD=Path('/public/home/aclsylqidf/hygon-inference-20261013');D=R/'incoming/common-dso-audit-r301';C=R/'candidates/common-dso-audit-r301';assert not C.exists();C.mkdir();scratch=C/'scratch';scratch.mkdir();local=scratch/'local';local.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items()if k not in ['LD_PRELOAD','APPTAINERENV_LD_PRELOAD','SINGULARITYENV_LD_PRELOAD']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env[k]='';env['APPTAINERENV_'+k]=''
env.update(PYTHONDONTWRITEBYTECODE='1',APPTAINERENV_PYTHONDONTWRITEBYTECODE='1')
assert not any(env.get(k)for k in ['APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH'])
args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--home',str(OLD/'build-deps/root')+':/root','--bind',str(OLD)+':/old:ro','--bind',str(R)+':/refactor','--bind',str(R)+':'+str(R)+':ro','--bind','/opt/hyhal:/opt/hyhal:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro','--bind',str(scratch)+':/tmp','--bind',str(local)+':/local',str(R/'containers/dtk26-sandbox-r1'),'python3','-B','/refactor/incoming/common-dso-audit-r301/check_common_dso_container.py']
start=time.monotonic();z=subprocess.run(args,env=env,cwd=R/'incoming/GLM-latest-controller-r296',capture_output=True,text=True,timeout=225)
(C/'stdout.log').write_text(z.stdout);(C/'stderr.log').write_text(z.stderr)
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'weights':0,'Native_dlopen':0,'rc':z.returncode,'completed':z.returncode==0,'passed':z.returncode==0,'elapsed_s':time.monotonic()-start,'stdout_SHA256':sha(C/'stdout.log'),'stderr_SHA256':sha(C/'stderr.log'),'inside_receipt_SHA256':sha(C/'CPU-inside.json')if(C/'CPU-inside.json').exists()else None,'error_tail':z.stderr[-3000:]if z.returncode else None}
(C/'CPU-summary.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q));assert z.returncode==0
