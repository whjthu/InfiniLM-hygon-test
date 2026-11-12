from pathlib import Path
import ast,hashlib,importlib.metadata as metadata,io,json,os,resource,signal,sys,time,traceback,unittest
W=Path('/evidence');S=Path('/input-stage');F=Path('/input-framework');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==220

def expired(*_):raise TimeoutError('original paired-helper CPU epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n11' and os.environ['SLURM_CPUS_PER_TASK']=='4';phase='source';tests=None
try:
 fm=json.loads((S/'FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==337 and len(fm['files'])==6
 for n,pin in fm['files'].items():
  p=F/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];ast.parse(p.read_bytes())
 deps=json.loads((S/'DEPENDENCIES.json').read_bytes());versions={n:metadata.version(n) for n in deps['versions']};assert versions==deps['versions'];import torch,numpy
 assert torch.version.hip==deps['HIP'] and not torch.cuda.is_initialized();torch.set_num_threads(2)
 def forbid(*_a,**_kw):raise RuntimeError('GPU context forbidden in paired-helperCPU')
 torch.cuda._lazy_init=forbid;sys.path.insert(0,str(F/'generation/src'));phase='actual-CPU-algebra-and-CLI-tests';log=io.StringIO();suite=unittest.TestSuite()
 for root,pattern in [(F/'generation/tests','test_trellis_operand_attribution.py'),(F/'generation/tests','test_lora_rank4_operand_attribution.py'),(F/'tests','test_sdxl_operand_attribution_entry.py')]:suite.addTests(unittest.defaultTestLoader.discover(str(root),pattern=pattern,top_level_dir=str(root)))
 tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(W/'tests.private.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==12 and not torch.cuda.is_initialized();providers={}
 for n,p in [('torch',Path(torch.__file__)),('torch_extension',Path(torch._C.__file__)),('numpy',Path(numpy.__file__))]:providers[n]=dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p))
 result=dict(passed=True,phase='TERMINAL_UNIFIED337_SUBSET6_PAIRED_ALGEBRA_CPU12_PASS',tests=12,legacy_rank2_FP32_tests=3,rank4_BF16_tests=5,entry_guard_tests=4,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=337,actual_framework_subset=6,full_framework_installed=False,versions=versions,HIP=torch.version.hip,providers=providers,CUDA_initialized=False,GPU_cards=0,real_operand_archives_read=False,model_weight_bytes_read=0,raw_input_bytes_read=0,paired_attribution_executed=False,Model_GO=False)
except BaseException as error:
 (W/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,phase=phase,error_type=type(error).__name__,tests=tests.testsRun if tests else None,GPU_cards=0,real_operand_archives_read=False,Model_GO=False)
result.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(W/'CPU-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
