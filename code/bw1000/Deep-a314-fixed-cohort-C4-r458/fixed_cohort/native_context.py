"""One engine on one owner thread; its qualified Native owns eight internal workers."""
import ctypes,hashlib,json,os,threading
from pathlib import Path
from .checkpoint_loader import load_checkpoint,NATIVE_SHA
from .common_mapping_audit import validate_deep
from .model_affinity_check_r2628 import verify as verify_affinity
from .forward import make_forward
from .profile import check_modes

def observe_threads(lib):
 owners=verify_affinity(lib)['actual_owner_report']['owners'];rows=[]
 for owner in owners:
  mask={int(x)for x in owner['verified_mask'].split(',')};matches=[]
  for task in Path('/proc/self/task').iterdir():
   try:
    if set(os.sched_getaffinity(int(task.name)))==mask:matches.append(int(task.name))
   except ProcessLookupError:pass
  if len(matches)==1:rows.append(dict(device=owner['device'],TID=matches[0]))
 return dict(status='OBSERVED'if len(rows)==8 else'UNKNOWN',owners=rows)

def providers(binding,receipt):
 lib=binding.lib;validate_deep(receipt['actual_DTK_bindings'])
 class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int;actual={}
 for name in ['create','load','finish','forward','reset','destroy']:
  info=Info();assert dl(ctypes.cast(getattr(lib,'modern_native_engine_'+name),ctypes.c_void_p),ctypes.byref(info));p=Path(info.fname.decode()).resolve()
  assert p==Path(receipt['library']).resolve()and hashlib.sha256(p.read_bytes()).hexdigest()==NATIVE_SHA;actual[name]=str(p)
 maps={Path(x.split()[-1]).resolve()for x in Path('/proc/self/maps').read_text().splitlines()if len(x.split())>=6 and x.split()[-1].startswith('/')}
 for name,proof in receipt['actual_DTK_bindings'].items():
  expected=Path(name).resolve();stem=Path(name).name.split('.so')[0];found={p for p in maps if stem in p.name};assert found=={expected}and hashlib.sha256(expected.read_bytes()).hexdigest()==proof['SHA256']
 return actual

def blas_snapshot():
 lib=ctypes.CDLL(None);lib.modern_blas_policy_r243_json.restype=ctypes.c_char_p
 value=json.loads(lib.modern_blas_policy_r243_json().decode())
 assert value['failures']==0 and all(v['created']==v['verified']for v in value['owners'])
 return value

class FixedCohortContext:
 def __init__(self,binding,receipt,*,config,cfg,mapping,torch,safe_open,report,deadline_check):
  self.binding,self.receipt=binding,receipt;self.config,self.cfg,self.mapping=config,cfg,mapping
  self.torch,self.safe_open=torch,safe_open;self.report,self.deadline_check=report,deadline_check
  self.ctx=None;self.thread=None;self.closed=False;self.forward=None
 def check_thread(self):
  assert self.thread==threading.get_ident()and not self.closed
 def __enter__(self):
  if self.thread is not None:raise RuntimeError('Deep context cannot be replayed')
  self.thread=threading.get_ident();lib=self.binding.lib;check=self.binding.check
  assert self.config['Native_SHA256']==NATIVE_SHA and self.config['HC_mode']==0
  assert self.config['profile']=='Deep-a314-fixed-cohort-C4-r458' and self.config['rows']==4 and self.config['capacity_per_row']==32768
  assert self.config['r366']==self.config['fullpre']==0
  check_modes(lib)
  assert lib.deep_hc_canonical_get_r7130()==0 and lib.modern_deep_c4_tile_get_r5920()==0
  assert lib.modern_deep_layout_get_r7052()==0 and lib.deep_moe_tail_get_r7211()==0
  check(lib.modern_deep_allselected_set_r813(1))
  self.deadline_check('Native create',minimum=120)
  self.ctx=lib.modern_native_engine_create(json.dumps(self.cfg).encode(),4,32768)
  if not self.ctx:raise RuntimeError(lib.modern_native_engine_error().decode())
  try:
   self.report.update(profile='Deep-a314-fixed-cohort-C4-r458',cache_rows=4,cache_capacity_per_row=32768,owner_affinity_before_weight_load=verify_affinity(lib),GPU_memory_after_create=[list(self.torch.cuda.mem_get_info(i))for i in range(8)])
   assert all(f>8*1024**3 for f,t in self.report['GPU_memory_after_create'])
   self.report['actual_providers_before_weights']=providers(self.binding,self.receipt)
   load_checkpoint(self.binding,self.ctx,model=self.config['model_path'],torch=self.torch,safe_open=self.safe_open,mapping=self.mapping,report=self.report,deadline_check=self.deadline_check)
   self.report['owner_affinity_after_weight_load']=verify_affinity(lib);self.report['GPU_memory_after_load']=[list(self.torch.cuda.mem_get_info(i))for i in range(8)]
   assert all(f>=16*1024**3 for f,t in self.report['GPU_memory_after_load'])
   self.report['owner_thread_observation']=observe_threads(lib)
   self.forward=make_forward(lib,self.ctx,self.cfg,self.report,check);return self
  except BaseException as error:
   try:self.close()
   except BaseException as cleanup:
    if hasattr(error,'add_note'):error.add_note('Deep startup cleanup: '+repr(cleanup))
   raise
 def reset(self):self.check_thread();self.binding.check(self.binding.lib.modern_native_engine_reset(self.ctx))
 def floor(self):
  self.check_thread();values=[list(self.torch.cuda.mem_get_info(i))for i in range(8)];assert all(f>=16*1024**3 for f,t in values);return values
 def close(self):
  self.check_thread();failure=None
  try:
   try:
    policy=blas_snapshot()
    if self.report.get('HC_dispatch_checked_forwards',0):assert len({v['tid']for v in policy['owners']if'libinfiniops.so'in v['caller']})==8
    self.report['blas_policy_before_destroy']=policy
   except BaseException as error:failure=error
   self.binding.check(self.binding.lib.modern_native_engine_destroy(self.ctx));self.report['engine_worker_shutdown_completed']=True
   observed=self.report.get('owner_thread_observation',{});rows=observed.get('owners',[])
   self.report['observed_all8_threads_exited']=len(rows)==8 and len({r['TID']for r in rows})==8 and not any(Path('/proc/self/task/'+str(r['TID'])).exists()for r in rows)
   self.report['normal_Native_shutdown']=failure is None
  finally:self.ctx=None;self.closed=True
  if failure is not None:raise failure
 def __exit__(self,typ,error,tb):
  try:self.close()
  except BaseException as cleanup:
   if error is None:raise
   if hasattr(error,'add_note'):error.add_note('Deep Native destroy: '+repr(cleanup))
  return False
