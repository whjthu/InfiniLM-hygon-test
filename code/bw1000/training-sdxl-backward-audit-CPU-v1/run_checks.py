"""Nine actual CPU Torch/PEFT observer tests; no model/data/device workload."""
from pathlib import Path
import ast,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,unittest
W=Path('/evidence');S=Path('/input-stage');F=Path('/input-framework');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==220
def expired(*_):raise TimeoutError('original CPU epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n11' and os.environ['SLURM_CPUS_PER_TASK']=='4';phase='source';versions={};tests=None
try:
 fm=json.loads((S/'FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==330 and len(fm['files'])==5
 for name,pin in fm['files'].items():
  p=F/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];ast.parse(p.read_bytes())
 assert sha(F/'tools/hygon_sdxl_first_step.py')=='cdf2f908887db4163c29c32793491ed781fcafa2585a177e85f0ec4af75abded'
 deps=json.loads((S/'DEPENDENCIES.json').read_bytes());versions={k:metadata.version(k) for k in deps['versions']};assert versions==deps['versions']
 import torch
 assert torch.version.hip==deps['HIP'] and not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_kw):raise RuntimeError('device context forbidden for backward observer CPU tests')
 torch.cuda._lazy_init=forbid;sys.path.insert(0,str(F/'generation/src'));phase='actual-Torch-PEFT-CPU-tests';log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(F/'generation/tests'),pattern='test_lora_backward_audit.py');tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==9
 import peft
 providers={}
 for name,path in [('torch',Path(torch.__file__)),('torch_extension',Path(torch._C.__file__)),('peft',Path(peft.__file__))]:providers[name]=dict(path=str(path),bytes=path.stat().st_size,SHA256=sha(path))
 assert not torch.cuda.is_initialized();result=dict(passed=True,phase='TERMINAL_SUBSET5_NATIVE_LORA_BACKWARD_CPU9_PASS',tests=tests.testsRun,versions=versions,HIP=torch.version.hip,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],total_framework_files=330,actual_source_subset_files=5,full330_framework_installed=False,providers=providers,tests_log_SHA256=sha(W/'tests.private.txt'),CUDA_initialized=False,GPU_cards=0,model_weight_bytes_read=0,raw_input_bytes_read=0,full_UNet_instantiated=False,native_model_backward_observed=False,numerical_alignment=False,Model_GO=False,qualification='Nine actual target Torch/PEFT tiny CPU tests for additive hooks and CPUFP64 local-reference behavior. Not actual UNet operands, model numerical alignment, performance or full330 target closure.')
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,tests=tests.testsRun if tests else None,versions=versions,GPU_cards=0,model_weight_bytes_read=0,raw_input_bytes_read=0,Model_GO=False)
result.update(owner=owner,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,epoch=epoch,deadline=deadline);(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
