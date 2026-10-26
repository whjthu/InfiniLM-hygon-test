import sys,os,pathlib,json,socket,hashlib
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('ROCR_VISIBLE_DEVICES')==''and os.environ.get('HIP_VISIBLE_DEVICES')==''
src=pathlib.Path('/refactor/incoming/Kimi-modern-PP4-cap32-r9145');sys.path.insert(0,str(src))
import target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143
assert 'torch'not in sys.modules and pathlib.Path(target_protocol.__file__).resolve()==src/'target_protocol.py'
m=json.loads((src/'model-manifest-PP4C32-r9145.json').read_text());entry_contract_r9143.validate(m);model,b=entry_contract_r9143.verify_inputs(m);node=int(os.environ['SLURM_PROCID'])+int(os.environ['GLOBAL_OFFSET_R2354']);assert socket.gethostname()==m['nodes'][node];assert int(os.environ['SLURM_CPUS_PER_TASK'])==64 and len(os.sched_getaffinity(0))>=64
import subprocess
ldd=subprocess.run(['ldd',b['native']],text=True,capture_output=True,timeout=40);assert ldd.returncode==0 and 'not found'not in ldd.stdout,(ldd.stdout,ldd.stderr)
q={'ldd_notfound':False,'CPU_only':True,'GPU':False,'CUDA_initialized':False,'Runtime_created':False,'dlopen':False,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':socket.gethostname(),'rank':node,'CPUs_per_task':64,'affinity_count':len(os.sched_getaffinity(0)),'imported_modules':{x.__name__:x.__file__ for x in [target_protocol,entry_contract_r9143,parameter_loader_r9143,native_binding_r9143]},'manifest_SHA256':hashlib.sha256((src/'model-manifest-PP4C32-r9145.json').read_bytes()).hexdigest(),'all_source_modelview_DSO_object_closure':True};print(json.dumps(q),flush=True)
