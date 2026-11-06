"""One independent phase process; actual Native Runtime/21 cache owners only."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,socket,time,threading,traceback
from binding import API
from workload import run_phase,ORDER
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
D=Path(__file__).resolve().parent
class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
def publish(p,q):t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
class DriverHistory:
 def __init__(self,cuda):self.cuda=cuda;self.samples=[];self.errors=[];self.stop=threading.Event();self.thread=None
 def start(self):
  def work():
   if self.cuda.cudaSetDevice(0):self.errors.append('sampler setdevice');return
   while not self.stop.is_set():
    f,t=C.c_size_t(),C.c_size_t();rc=self.cuda.cudaMemGetInfo(C.byref(f),C.byref(t));self.samples.append([time.monotonic(),time.time(),int(rc),f.value,t.value])
    if rc:self.errors.append('sampler memoryquery');return
    self.stop.wait(.005)
  self.thread=threading.Thread(target=work,daemon=True,name='bounded-phase-memory');self.thread.start()
 def close(self):
  self.stop.set();self.thread.join(timeout=5)
  if self.thread.is_alive():self.errors.append('sampler not joined')
def main(a):
 epoch=float(os.environ['DEEP_SLOT_GPU_START']);global_deadline=epoch+480;limit=min(global_deadline,a.phase_deadline)
 q={'completed':False,'passed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'PID':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'UID':os.getuid(),'node':socket.gethostname().split('.')[0],'phase':a.phase,'mode':a.mode,'original_epoch':epoch,'phase_deadline':limit,'source_manifest_SHA256':os.environ['DEEP_SLOT_GPU_SOURCE_SHA'],'planned_measured':126,'planned_warm':21,'schedule':None,'normal_Runtime_close':False,'measured_failed':0,'explicit_Model_or_weights':False};api=None;history=None;last=0.;start=time.monotonic()
 def clock():
  if time.time()>=limit:raise TimeoutError('original component epoch/phase budget, no retry')
 def save(schedule=None,force=False):
  nonlocal last
  if schedule is not None:q['schedule']=schedule
  if force or time.monotonic()-last>=.5:publish(a.output,q);last=time.monotonic()
 save(force=True)
 try:
  assert q['node']=='a02r4n01'and a.mode==ORDER[a.phase]and 'torch'not in __import__('sys').modules;clock();mf=D/'GPU_SOURCE_MANIFEST.json';assert H(mf)==q['source_manifest_SHA256'];m=json.loads(mf.read_bytes());assert m['GPU_GO']and not m['Model_GO'];assert all(H(D/n)==h for n,h in m['files_SHA256'].items());assert H(m['CPU_receipt'])==m['CPU_receipt_SHA256'];cpu=json.loads(Path(m['CPU_receipt']).read_bytes());assert cpu['completed']and cpu['passed']and cpu['actual_CPU']=='910407.1116';v=cpu['libraries']['probe'];assert v['path']==m['probe_path']and v['SHA256']==m['probe_SHA256']and H(v['path'])==v['SHA256'];assert H(m['Native_path'])==m['Native_SHA256'];assert len(v['all_transitive_resolved_library_SHA256'])==69
  for p,h in v['all_transitive_resolved_library_SHA256'].items():clock();assert H(p)==h,p
  lib=C.CDLL(m['probe_path'],mode=C.RTLD_LOCAL);dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int;q['symbol_owners']={}
  for name,expected in m['symbol_owners'].items():
   f=getattr(lib,name);i=DlInfo();assert dl.dladdr(C.cast(f,C.c_void_p),C.byref(i))and i.name;p=Path(i.name.decode()).resolve();assert p==Path(expected['path']).resolve()and H(p)==expected['SHA256'];q['symbol_owners'][name]={'path':str(p),'SHA256':H(p)}
  cuda=C.CDLL(m['cuda_runtime_path'],mode=C.RTLD_GLOBAL)
  for n,args in [('cudaSetDevice',[C.c_int]),('cudaGetDeviceCount',[C.POINTER(C.c_int)]),('cudaMemGetInfo',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]),('cudaDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int])]:f=getattr(cuda,n);f.argtypes=args;f.restype=C.c_int
  count=C.c_int();assert cuda.cudaGetDeviceCount(C.byref(count))==0 and count.value==1 and cuda.cudaSetDevice(0)==0;pci=C.create_string_buffer(32);assert cuda.cudaDeviceGetPCIBusId(pci,32,0)==0;q['physical_PCI']=pci.value.decode();q['visible_count']=count.value;f,t=C.c_size_t(),C.c_size_t();assert cuda.cudaMemGetInfo(C.byref(f),C.byref(t))==0;assert t.value==68702699520;q['first_driver_point']=[time.monotonic(),time.time(),0,f.value,t.value];q['initialization_scope']='Libraryload/first CUDA initialization have only a boundary; peaks before continuous sampling UNKNOWN.';history=DriverHistory(cuda);history.start();clock();api=API(lib,a.mode);q['after_create']=api.observe();save(force=True)
  schedule=run_phase(api,a.mode,a.directory,a.phase,clock=clock,save=save);q['schedule']=schedule;api.close();q['close']=api.closed;q['normal_Runtime_close']=bool(api.closed and api.closed['rc']==0 and api.closed['values'][2]==1);clock();q['passed']=bool(schedule['passed']and q['normal_Runtime_close']);save(force=True)
 except BaseException as e:
  q.update(error=repr(e),traceback=traceback.format_exc());cur=(q.get('schedule')or{}).get('current_operation',{});q['measured_failed']=int(cur.get('kind')=='measured_gather')
 finally:
  if api and api.ctx:
   try:api.close();q['close']=api.closed;q['normal_Runtime_close']=bool(api.closed and api.closed['rc']==0 and api.closed['values'][2]==1)
   except BaseException as e:q['cleanup_error']=repr(e)
  if history:
   history.close();hp=a.output.parent/('phase%d-driver-history.json'%a.phase);hp.write_text(json.dumps(history.samples)+'\n');q['driver_history']={'path':str(hp),'bytes':hp.stat().st_size,'SHA256':H(hp),'samples':len(history.samples),'errors':history.errors};points=[q.get('first_driver_point',[])]+history.samples;points=[x for x in points if x];q['driver_min_observed']=min((x[3]for x in points),default=0);q['driver_gap_max_s']=max((b[0]-a[0]for a,b in zip(history.samples,history.samples[1:])),default=None);q['sampled_floor_passed']=bool(points and not history.errors and all(x[2]==0 and x[3]>=16*1024**3 and x[4]==68702699520 for x in points));q['passed']=bool(q['passed']and q['sampled_floor_passed'])
  q.update(completed=True,elapsed_worker_s=time.monotonic()-start,Torch_imported='torch'in __import__('sys').modules);save(force=True)
 return 0 if q['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',type=int,required=True);p.add_argument('--mode',type=int,required=True);p.add_argument('--phase-deadline',type=float,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();assert not a.output.exists();raise SystemExit(main(a))
