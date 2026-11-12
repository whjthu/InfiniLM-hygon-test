"""Actual unified target native meta caller; no tensor bodies or GPU initialization."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,resource,signal,sys,time,traceback
from argparse import Namespace
W=Path('/evidence');S=Path('/input-stage');F=Path('/input-framework');epoch=float(os.environ['INPUT_CPU_EPOCH']);deadline=float(os.environ['INPUT_CPU_DEADLINE']);start=time.monotonic()
assert deadline-epoch==220 and epoch<=time.time()<deadline
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original meta CPU epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']);assert owner['node']=='f11r4n13' and os.environ['SLURM_CPUS_PER_TASK']=='4'
attempts=[];phase='source';torch=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
 if event=='socket.connect':raise RuntimeError('network forbidden in native CPU meta scope')
 if event=='open' and args and isinstance(args[0],(bytes,str)):
  name=os.fsdecode(args[0])
  if name.endswith(('.safetensors','.pt','.pth','.ckpt','.bin')):
   attempts.append(hashlib.sha256(os.fsencode(args[0])).hexdigest());raise RuntimeError('weights and tensor files forbidden in native CPU meta scope')
sys.addaudithook(forbid)
try:
 for n,pin in json.loads((S/'SOURCE_FILES.json').read_bytes())['files'].items():
  p=S/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==380
 for n,pin in fm['files'].items():
  p=F/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
  if n.endswith('.py'):ast.parse(p.read_bytes())
 phase='actual-offline-dependency-imports'
 import importlib,importlib.metadata as md
 import torch
 assert not torch.cuda.is_initialized() and torch.version.hip=='6.3.26113'
 def deny_GPU(*a,**k):raise RuntimeError('GPU initialization forbidden in nativeCPU scope')
 torch.cuda._lazy_init=deny_GPU
 deps=json.loads((S/'PUBLIC_PACKAGES.json').read_bytes());dependency_versions={};requirements=[]
 from packaging.requirements import Requirement
 for pin in deps['packages'].values():
  assert md.version(pin['name'])==pin['version'];dependency_versions[pin['name']]=pin['version']
  distribution=md.distribution(pin['name']);assert str(distribution.locate_file('')).startswith('/zimage-runtime')
  for text in distribution.requires or []:
   requirement=Requirement(text)
   if requirement.marker is None or requirement.marker.evaluate({'extra':''}):
    actual=md.version(requirement.name);assert requirement.specifier.contains(actual,prereleases=True);requirements.append(dict(name=requirement.name,version=actual,requirement=str(requirement)))
 eager=json.loads((S/'native-eager-imports.json').read_bytes());assert len(eager)==24
 imported_dependencies={}
 for name in eager:
  if name=='diffsynth':continue
  module=importlib.import_module(name);imported_dependencies[name]=getattr(module,'__file__',None)
 assert not torch.cuda.is_initialized()
 phase='unified-native-meta-entry'
 module_spec=importlib.util.spec_from_file_location('zimage_meta_check',F/'tools/hygon_zimage_meta_check.py');entry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(entry)
 native=entry.run(Namespace(epoch=epoch,deadline=deadline,spec=S/'META_SPEC.json',spec_sha256=sha(S/'META_SPEC.json'),source_root=Path('/native-source'),output=W/'native-meta-result.json'))
 import torch
 assert native['passed'] and not native['CUDA_initialized'] and not attempts
 result=dict(passed=True,phase='TARGET_UNIFIED380_NATIVE_ZIMAGE_META_PASS',framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_files=380,dependency_versions=dependency_versions,dependency_requirements=requirements,dependency_imports=imported_dependencies,native_result=dict(bytes=(W/'native-meta-result.json').stat().st_size,SHA256=sha(W/'native-meta-result.json'),value=native),CUDA_initialized=False,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
except BaseException as e:
 (W/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,phase=phase,error_type=type(e).__name__,CUDA_initialized=torch.cuda.is_initialized() if torch else None,weights_or_data_read_bytes=0,GPU_cards=0,GPU_GO=False,Model_GO=False)
result.update(owner=owner,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,forbidden_file_attempts=len(attempts))
(W/'CPU-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
