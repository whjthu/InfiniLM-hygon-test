from pathlib import Path
import datetime,hashlib,io,json,os,signal,stat,sys,time,unittest
W=Path('/evidence');source=Path('/resume-framework');stage_source=Path('/resume-stage');sys.path.insert(0,str(source));start=time.monotonic();epoch=float(os.environ['RESUME_CPU_EPOCH']);deadline=float(os.environ['RESUME_CPU_DEADLINE']);initial=deadline-time.time()
assert 0<initial<=220
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('entry-state CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,initial)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner['node']=='f11r4n10' and os.environ['SLURM_CPUS_PER_TASK']=='4'
with (W/'owner.json').open('x') as f:json.dump(owner,f,indent=2)
result=None;stage='source'
try:
 mf=json.loads((stage_source/'SOURCE_FILES.json').read_bytes());remote_source_verified=[]
 for name,row in mf['files'].items():assert (stage_source/name).stat().st_size==row['bytes'] and h(stage_source/name)==row['SHA256'];remote_source_verified.append(name)
 fm=json.loads((stage_source/'FRAMEWORK_FILES.json').read_bytes());verified=[]
 for name,row in fm['files'].items():assert (source/name).stat().st_size==row['bytes'] and h(source/name)==row['SHA256'] and stat.S_IMODE((source/name).stat().st_mode)==row['mode'];verified.append(name)
 assert len(verified)==353 and fm['framework_commit']=='f9aaaa8f89624fae3c0ed960780b62e1fee375da'
 stage='dependency-bytes'
 dependencies=json.loads((stage_source/'DEPENDENCIES.json').read_bytes())
 for pin in dependencies['providers'].values():
  p=Path(pin['path']);assert p.stat().st_size==pin['bytes'] and h(p)==pin['SHA256']
 stage='unit-tests';import torch
 assert not torch.cuda.is_initialized()
 def forbid_GPU(*args,**kwargs):raise RuntimeError('GPU init forbidden in CPU26 scope')
 torch.cuda._lazy_init=forbid_GPU

 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_forward_boundary_trace','vlm.tests.test_resume_witness']);tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (W/'tests.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==26,log.getvalue()[-1400:]
 stage='plugin-registration-and-runtime-providers'
 import transformers,peft,accelerate
 import vlm.plugins.resume_training as plugin
 holder=object();assert plugin.NativeResume(None,holder).trainer is holder and plugin.callbacks_map[plugin.KEY] is plugin.NativeResume
 from accelerate import data_loader,accelerator
 from accelerate.utils import random as rng_module
 providers={}
 for key,module in [('accelerate',accelerate),('accelerator',accelerator),('data_loader',data_loader),('accelerate_random',rng_module),('transformers',transformers),('peft',peft),('torch',torch)]:
  p=Path(module.__file__);providers[key]=dict(path=str(p),bytes=p.stat().st_size,SHA256=h(p))
 assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
 assert torch.version.hip and not torch.version.cuda
 assert dict(torch=torch.__version__,HIP=torch.version.hip,transformers=transformers.__version__,peft=peft.__version__,accelerate=accelerate.__version__)==dependencies['versions']
 result=dict(passed=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,deadline=deadline,source_files_verified=len(verified),source_manifest_SHA256=h(stage_source/'SOURCE_FILES.json'),framework_manifest_SHA256=h(stage_source/'FRAMEWORK_FILES.json'),framework_commit=fm['framework_commit'],remote_source_files_verified=len(remote_source_verified),tests_run=26,new_boundary_hook_tests=8,affected_resume_tests=18,new_resume_integration_tests=2,tests_SHA256=h(W/'tests.txt'),test_log=log.getvalue(),plugin_registered_and_base_constructor_verified=True,actual_runtime_providers=providers,torch=torch.__version__,HIP=torch.version.hip,transformers=transformers.__version__,peft=peft.__version__,accelerate=accelerate.__version__,Python=sys.version,CUDA_initialized=False,GPU_cards=0,original_model_loaded=False,original_checkpoint_loaded=False,raw_tensor_transfer_bytes=0,native_GPU_restore_executed=False,recovery_qualified=False,Model_GO=False,wall_seconds=time.monotonic()-start,qualification='Complete353 source tree and changed observer/plugin CPU fixtures on actual HIP container only. No actual MiniCPM recovery or GPU/quality/performance qualification.')
except BaseException as error:
 result=dict(passed=False,owner=owner,epoch=epoch,deadline=deadline,stage=stage,error_type=type(error).__name__,message=str(error)[-1800:],wall_seconds=time.monotonic()-start,GPU_cards=0,Model_GO=False)
finally:
 with (W/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
