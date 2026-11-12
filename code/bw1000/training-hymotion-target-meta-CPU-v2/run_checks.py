"""One actual target native meta import through the unified framework; no training."""
from pathlib import Path
import ast,hashlib,importlib.metadata as metadata,importlib.util,json,os,resource,signal,sys,time,traceback
W=Path('/evidence');S=Path('/input-stage');F=Path('/input-framework');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic();assert epoch<=time.time()<deadline and deadline-epoch==220

def expired(*_):raise TimeoutError('original target meta CPU epoch exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n15' and os.environ['SLURM_CPUS_PER_TASK']=='4';phase='published-source';versions={};value=None
try:
 pins=json.loads((S/'SOURCE_FILES.json').read_bytes())['files']
 for name,pin in pins.items():
  p=S/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 fm=json.loads((S/'FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==333 and len(fm['files'])==10
 for name,pin in fm['files'].items():
  p=F/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];ast.parse(p.read_bytes())
 deps=json.loads((S/'DEPENDENCIES.json').read_bytes());versions={n:metadata.version(n) for n in deps['known_expected_versions']};assert versions==deps['known_expected_versions'];phase='actual-unified-entry-native-meta'
 entry_spec=importlib.util.spec_from_file_location('hymotion_meta_check',F/'tools/hygon_hymotion_meta_check.py');entry=importlib.util.module_from_spec(entry_spec);entry_spec.loader.exec_module(entry);from argparse import Namespace
 value=entry.run(Namespace(epoch=epoch,deadline=deadline,spec=S/'META_SPEC.json',spec_sha256=sha(S/'META_SPEC.json'),source_root=S/'upstream',output=W/'native-meta-result.json'))
 import torch,accelerate
 assert value['optional_GPU_extensions_excluded']==['flash_attn','flash_attn_2_cuda'] and not value['production_attention_qualified'];assert torch.version.hip==deps['HIP'] and not torch.cuda.is_initialized() and value['owner']['PID']==owner['PID'];providers={}
 for n,p in [('torch',Path(torch.__file__)),('torch_extension',Path(torch._C.__file__)),('accelerate',Path(accelerate.__file__))]:providers[n]=dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p))
 imported={}
 for n,module in sorted(sys.modules.items()):
  if n=='gencuda' or n.startswith('gencuda.'):
   p=Path(module.__file__).resolve();rel=p.relative_to(F.resolve()).as_posix();assert rel in fm['files'] and sha(p)==fm['files'][rel]['SHA256'];imported[n]=dict(path=rel,SHA256=sha(p))
 result=dict(passed=True,phase='TERMINAL_UNIFIED333_SUBSET10_NATIVE_HYMOTION_META_CPU_PASS',framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=333,actual_framework_source_subset=10,full_framework_installed=False,native_meta_result=dict(bytes=(W/'native-meta-result.json').stat().st_size,SHA256=sha(W/'native-meta-result.json'),value=value),runtime_unified_modules=imported,providers=providers,versions=value['versions'],HIP=torch.version.hip,CUDA_initialized=False,GPU_cards=0,weights_or_cache_tensor_bytes_read=0,frame_decision_pending=True,Model_GO=False)
except BaseException as error:
 (W/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,phase=phase,error_type=type(error).__name__,versions=versions,GPU_cards=0,weights_or_cache_tensor_bytes_read=0,frame_decision_pending=True,Model_GO=False)
result.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(W/'CPU-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
