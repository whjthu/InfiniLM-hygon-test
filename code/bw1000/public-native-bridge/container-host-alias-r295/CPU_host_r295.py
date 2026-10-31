"""One CPU-only step; identical image/binds except read-only same-project host alias."""
from pathlib import Path
import subprocess,os,json,hashlib,datetime,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');OLD=Path('/public/home/aclsylqidf/hygon-inference-20261013');D=R/'incoming/container-host-alias-r295';C=R/'candidates/container-host-alias-r295'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='910407' and not C.exists()
C.mkdir();start=time.monotonic()
assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
old=(R/'incoming/resource_wave_local_r3097_outer.sh').read_text()
new=(D/'resource_wave_host_alias_r295_outer.sh').read_text()
needle='--bind "$TASK_ROOT:/refactor"'
assert old.count(needle)==1 and new==old.replace(needle,needle+' --bind "$TASK_ROOT:$TASK_ROOT:ro"')
env={k:v for k,v in os.environ.items()if k not in ['LD_PRELOAD','APPTAINERENV_LD_PRELOAD','SINGULARITYENV_LD_PRELOAD']}
for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:
 env[k]='';env['APPTAINERENV_'+k]=''
env.update(PYTHONDONTWRITEBYTECODE='1',APPTAINERENV_PYTHONDONTWRITEBYTECODE='1',APPTAINERENV_PATH295_MANIFEST_SHA=sha(D/'SOURCE_MANIFEST.json'))
bind_keys=['APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH'];implicit={k:bool(env.get(k))for k in bind_keys};assert not any(implicit.values()),'Unexpected inherited container bind configuration'
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True,'GPU':0,'Model':0,'HTTP':0,'Native_dlopen':0,'weights':0,'one_CPU_step':True,'host_launcher_cwd':str(Path.cwd()),'host_HOME':os.environ.get('HOME'),'inherited_extra_bind_present':implicit,'container_process_cwd':str(R/'incoming/GLM-latest-controller-r292'),'CPU_limit_s':240,'original_wrapper_SHA256':sha(R/'incoming/resource_wave_local_r3097_outer.sh'),'new_wrapper_SHA256':sha(D/'resource_wave_host_alias_r295_outer.sh'),'source_manifest_SHA256':sha(D/'SOURCE_MANIFEST.json'),'only_wrapper_change':'Add read-only bind of same project at its original absolute host path','GPU_environment_difference':'No LD_PRELOAD or GPU_inside sourcing; visible device variables masked; original static Python functions only','phases':[]}
for phase in ['control','candidate']:
 scratch=C/phase;scratch.mkdir();local=scratch/'local';local.mkdir()
 args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--home',str(OLD/'build-deps/root')+':/root','--bind',str(OLD)+':/old:ro','--bind',str(R)+':/refactor','--bind','/opt/hyhal:/opt/hyhal:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro','--bind',str(scratch)+':/tmp','--bind',str(local)+':/local']
 if phase=='candidate':args+=['--bind',str(R)+':'+str(R)+':ro']
 args+=[str(R/'containers/dtk26-sandbox-r1'),'python3','-B','/refactor/incoming/container-host-alias-r295/check_r295.py',phase]
 left=230-int(time.monotonic()-start);assert left>0
 try:z=subprocess.run(args,env=env,cwd=R/'incoming/GLM-latest-controller-r292',capture_output=True,text=True,timeout=left)
 except subprocess.TimeoutExpired as e:
  q['phases'].append({'phase':phase,'state':'CPU_CONTAINER_TIMEOUT','timeout_s':left});q.update(completed=False,passed=False,elapsed_s=time.monotonic()-start);(C/'CPU-summary.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));raise
 (scratch/'stdout.log').write_text(z.stdout);(scratch/'stderr.log').write_text(z.stderr)
 row={'phase':phase,'rc':z.returncode,'stdout_SHA256':hashlib.sha256(z.stdout.encode()).hexdigest(),'stderr_SHA256':hashlib.sha256(z.stderr.encode()).hexdigest()};q['phases'].append(row)
 if z.returncode:row['error_tail']=z.stderr[-3500:];q.update(completed=False,passed=False);(C/'CPU-summary.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));raise RuntimeError('CPU namespace phase failed: '+phase)
 row['receipt_SHA256']=sha(C/('CPU-'+phase+'.json'))
q.update(completed=True,passed=True,elapsed_s=time.monotonic()-start,full_Model_or_ABI_qualified=False)
(C/'CPU-summary.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
