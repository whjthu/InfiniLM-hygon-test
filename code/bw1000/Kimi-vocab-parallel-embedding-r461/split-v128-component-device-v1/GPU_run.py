"""One physical GPU, one actual Runtime, local H3 attention component, no Model."""
from pathlib import Path
import ctypes as C,hashlib,json,os,socket,sys,time,threading,traceback
import numpy as np
D=Path(__file__).resolve().parent;SOURCE=D.parent/'split-v128-component-source-v1';sys.path.insert(0,str(SOURCE))
from binding import API
from schedule import run_schedule,CASES,ORDER
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(v):return(v.astype(np.uint32)<<16).view(np.float32)
def compare(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);delta=a-b;nrms=float(np.sqrt(np.mean(delta*delta)))/max(1e-20,float(np.sqrt(np.mean(b*b))));ma=float(np.max(np.abs(delta)))
 return dict(NRMS=nrms,maxabs=ma,agreement=nrms<=.03 and ma<=.05)
def publish(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
class CaptureAPI(API):
 def __init__(self,lib,record,clock):super().__init__(lib);self.record=record;self.clock=clock;self.closed=None
 def query(self,past,q,k,v):
  self.clock();count=q.shape[0];guard=np.full(count*3*128+32,0x3f25,dtype=np.uint16);out=guard[16:-16].reshape(count,3,128);seconds=np.zeros(3,dtype=np.float64)
  rc=self.lib.kimi_split_probe_query(self.ctx,past,count,q.ctypes.data,k.ctypes.data,v.ctypes.data,out.ctypes.data,out.size,seconds.ctypes.data,3)
  # Persist raw after the actual CAPI, including a floor/error RC after computation.
  self.record(out,dict(native_rc=int(rc),past=past,count=count,seconds=seconds.tolist(),output_CPU_canary=bool(np.all(guard[:16]==0x3f25)and np.all(guard[-16:]==0x3f25)),raw_on_failed_RC_not_qualified=bool(rc)))
  self.rc(rc)
  if not np.all(guard[:16]==0x3f25)or not np.all(guard[-16:]==0x3f25):raise AssertionError('CPU output canary')
  return out,seconds.tolist()
 def close(self):
  out=np.zeros(5,np.uint64);ctx=self.ctx;self.ctx=None
  if ctx:
   rc=self.lib.kimi_split_probe_destroy(ctx,out.ctypes.data,5);self.closed=dict(rc=int(rc),values=out.tolist());self.rc(rc)
  return out.tolist()
class DriverHistory:
 def __init__(self,cuda):self.cuda=cuda;self.samples=[];self.errors=[];self.stop=threading.Event();self.thread=None
 def start(self):
  def work():
   if self.cuda.cudaSetDevice(0):self.errors.append('sampler cudaSetDevice');return
   while not self.stop.is_set():
    free=C.c_size_t();total=C.c_size_t();rc=self.cuda.cudaMemGetInfo(C.byref(free),C.byref(total));self.samples.append([time.monotonic(),time.time(),int(rc),free.value,total.value])
    if rc:self.errors.append('cudaMemGetInfo '+str(rc));return
    self.stop.wait(.005)
  self.thread=threading.Thread(target=work,name='bounded-driver-memory-history',daemon=True);self.thread.start()
 def close(self):
  self.stop.set();self.thread.join(timeout=5)
  if self.thread.is_alive():self.errors.append('sampler not joined')
class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
def main():
 R=Path('/refactor');step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];epoch=float(os.environ['KIMI_SPLITV_GPU_START']);deadline=epoch+480
 result=R/'results'/('Kimi-split-V128-'+step+'.json');private=R/'results'/('Kimi-split-V128-'+step+'-private');assert not result.exists();private.mkdir()
 q=dict(completed=False,passed=False,numeric_agreement=None,normal_Runtime_close=False,Model=False,TP32_communication=False,synthetic_component=True,actual_step=step,PID=os.getpid(),UID=os.getuid(),node=socket.gethostname().split('.')[0],original_epoch=epoch,planned_phases=4,planned_queries=20,planned_querytokens=8196,events=[],outputs=[],source_manifest_SHA256=os.environ['KIMI_SPLITV_GPU_SOURCE_SHA'])
 history=None;api=None;torch=None;records=[];state={};q['driver_initialization_points']=[];publish(result,q)
 def clock():
  if time.time()>=deadline:raise TimeoutError('original epoch+480 work deadline before next operation')
 def driver_point(label):
  free=C.c_size_t();total=C.c_size_t();rc=cuda.cudaMemGetInfo(C.byref(free),C.byref(total));q['driver_initialization_points'].append(dict(label=label,unix=time.time(),monotonic=time.monotonic(),rc=int(rc),free=free.value,total=total.value));assert rc==0
 def snapshot_torch():
  if torch is None:return None
  return dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0),max_allocated=torch.cuda.max_memory_allocated(0),max_reserved=torch.cuda.max_memory_reserved(0),scope='monotonic process peaks; never reset_peak/empty_cache')
 def event(kind,phase,mode,value):
  if kind not in ['Runtime.closed']:clock()
  state.update(kind=kind,phase=phase,mode=mode)
  if isinstance(value,dict)and 'case'in value:state['case']=value['case']
  q['events'].append(dict(kind=kind,phase=phase,mode=mode,value=value,monotonic=time.monotonic(),unix=time.time(),Torch=snapshot_torch()));publish(result,q)
 def capture(out,metadata):
  clock_name='phase'+str(state['phase'])+'-'+str(state.get('case','unknown'))
  path=private/(clock_name+'.bf16');raw=out.astype('<u2',copy=False).tobytes()
  with path.open('xb')as f:f.write(raw)
  q['outputs'].append(dict(metadata,phase=state['phase'],mode=state['mode'],case=state.get('case'),path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),shape=list(out.shape),dtype='BF16',endianness='little',finite=bool(np.isfinite(f32(out)).all())));publish(result,q)
  if metadata['native_rc']==0 and not q['outputs'][-1]['finite']:raise FloatingPointError('actual nonfinite BF16 output')
 try:
  assert q['node']=='a14r2n08'and os.environ['SLURM_JOB_ID']=='910380';clock();mf=D/'GPU_SOURCE_MANIFEST.json';assert H(mf)==q['source_manifest_SHA256'];m=json.loads(mf.read_bytes());assert m['GPU_GO']and not m['Model_GO']
  for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
  assert H(SOURCE/'SOURCE_MANIFEST.json')==m['component_source_manifest_SHA256']
  for n,h in json.loads((SOURCE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert H(SOURCE/n)==h,n
  pin=Path(m['CPU_receipt']);assert H(pin)==m['CPU_receipt_SHA256'];cpu=json.loads(pin.read_bytes());assert cpu['passed']and cpu['actual_CPU']=='910407.1080'
  assert H(cpu['probe'])==m['probe_SHA256']==cpu['probe_SHA256'];assert H(cpu['parent_Native']['path'])==m['Native_SHA256']==cpu['parent_Native']['SHA256']
  for v in cpu['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  q['entry_affinity']=sorted(os.sched_getaffinity(0));q['allocator_intervention']='begin_mode clear+Runtime.trimMemory each OFF/ON phase; component only, not Model allocator state'
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12',mode=C.RTLD_GLOBAL)
  for name,args in [('cudaSetDevice',[C.c_int]),('cudaMemGetInfo',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]),('cudaGetDeviceCount',[C.POINTER(C.c_int)]),('cudaDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int])]:
   fn=getattr(cuda,name);fn.argtypes=args;fn.restype=C.c_int
  driver_point('first_self_mem_query_may_initialize_primary_context');count=C.c_int();assert cuda.cudaGetDeviceCount(C.byref(count))==0 and count.value==1;pci=C.create_string_buffer(32);assert cuda.cudaDeviceGetPCIBusId(pci,32,0)==0;q.update(visible_count=count.value,physical_PCI=pci.value.decode(),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'))
  driver_point('after_count_PCI_before_Torch');q['driver_sampling_coverage']='processstart/first CUDA context initialization and Torch import have only boundary points; transient peaks in those gaps UNKNOWN. Continuous5ms history starts after Torch import; subinterval peaks UNKNOWN'
  import torch as actual_torch
  torch=actual_torch;q['Torch_version']=torch.__version__;q['Torch_initial']=snapshot_torch();driver_point('after_Torch_import_allocator_stats');history=DriverHistory(cuda);history.start()
  lib=C.CDLL(cpu['probe'],mode=C.RTLD_LOCAL);dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int;q['symbols']={}
  mangled=m['baseline_symbols'];runtime_dep=next(v for k,v in cpu['actual_dependency_library_SHA256'].items()if k=='libinfinilm_runtime.so')
  for label,fn in [(n,getattr(lib,n))for n in cpu['CAPI']]+[(n,getattr(lib,s))for n,s in mangled.items()]:
   info=DlInfo();assert dl.dladdr(C.cast(fn,C.c_void_p),C.byref(info))and info.name;path=Path(info.name.decode()).resolve();q['symbols'][label]=dict(path=str(path),SHA256=H(path))
   if label in cpu['CAPI']:assert path==Path(cpu['probe']).resolve()
   else:assert path==Path(runtime_dep['path']).resolve()and H(path)==runtime_dep['SHA256'],(label,str(path))
  from binding import declare
  declare(lib)
  for i,name in enumerate(['allocate','forward','cache_store']):
   path=Path(lib.kimi_split_probe_symbol_path(i).decode()).resolve();assert path==Path(cpu['parent_Native']['path']).resolve();q['symbols'][name]=dict(path=str(path),SHA256=H(path))
  def words(shape,seed):
   a=np.arange(np.prod(shape),dtype=np.uint32);a=a*np.uint32(1664525)+np.uint32(seed);x=((a>>24).astype(np.int16)-128).astype(np.float32)/np.float32(256);return(x.view(np.uint32)>>16).astype(np.uint16).reshape(shape)
  data=(words((32768,3,192),1013904223),words((32768,3,192),12345),words((32768,3,128),98765));q['inputs']=[dict(shape=list(a.shape),dtype='BF16',SHA256=hashlib.sha256(a.tobytes()).hexdigest())for a in data];api=CaptureAPI(lib,capture,clock);publish(result,q)
  records,closed=run_schedule(api,data,event);q['normal_Runtime_close']=bool(api.closed and api.closed['rc']==0 and closed[2]==1);q['close']=api.closed
  # All raw outputs already persisted after each actual CAPI, even if later audit/close fails.
  sample_refs={};first={};phase_past0={};numeric=[]
  for r in records:
   clock();name=r['case'];past=r['past'];count=r['count'];a=f32(r['output']);rows=sorted(set([0,min(count-1,31),count//2,count-1]))
   if name not in sample_refs:
    ref=np.zeros((len(rows),3,128),np.float32);Q,K,V=map(f32,data)
    for j,row in enumerate(rows):
     total=past+row+1
     for head in range(3):
      score=K[:total,head]@Q[past+row,head]*np.float32(1/np.sqrt(192));prob=np.exp(score.astype(np.float64)-float(score.max()));prob/=prob.sum();ref[j,head]=prob@V[:total,head]
    sample_refs[name]=(rows,ref)
   if name not in first:first[name]=a.copy()
   rows,ref=sample_refs[name]
   if name=='past0':phase_past0[r['phase']]=a.copy()
   reset_agreement=compare(a,phase_past0[r['phase']])if name=='reset_reuse'else None
   numeric.append(dict(phase=r['phase'],mode=r['mode'],case=name,full_vs_first_OFF=compare(a,first[name]),CPU_samples=compare(a[rows],ref),reset_vs_samephase_past0=reset_agreement))
  q['numeric']=numeric;q['numeric_agreement']=len(numeric)==20 and all(x['full_vs_first_OFF']['agreement']and x['CPU_samples']['agreement']and(x['reset_vs_samephase_past0']is None or x['reset_vs_samephase_past0']['agreement'])for x in numeric)
  q['mechanical_passed']=len(records)==20 and len(q['outputs'])==20 and all(x['finite']and x['output_CPU_canary']and x['native_rc']==0 for x in q['outputs'])and q['normal_Runtime_close'];q['query_metric_scopes']=['Native FA/cache preparation dispatch+sync (baseline includes padding and update)','output D2H plus audit_inputs ALL GPU guard readbacks+sync','entire CABI including input H2D/metadata/FA/audits'];q['final_affinity']=sorted(os.sched_getaffinity(0));q['Torch_final']=snapshot_torch()
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),close=getattr(api,'closed',None))
 finally:
  if api and api.ctx:
   try:api.close();q['close']=api.closed
   except BaseException as e:q['cleanup_error']=repr(e)
  if api and api.closed:q['normal_Runtime_close']=bool(api.closed['rc']==0 and api.closed['values'][2]==1);q['close']=api.closed
  if history:
   history.close();q['memory_sampler_errors']=history.errors;q['driver_samples_count']=len(history.samples);q['driver_min_observed']=min((s[3]for s in history.samples if s[2]==0),default=None)
   samples=private/'driver-memory-history.json';samples.write_text(json.dumps(history.samples,separators=(',',':')));q['driver_history']=dict(path=str(samples),bytes=samples.stat().st_size,SHA256=H(samples),interval_target_s=.005,scope='continuous process driver observations; no reset; subinterval peaks unknown')
   phase_observations=[]
   for phase,mode in enumerate(ORDER):
    begins=[e for e in q['events']if e['phase']==phase and e['kind']=='phase.allocated'];ends=[e for e in q['events']if e['phase']==phase and e['kind']=='phase.complete']
    if begins and ends:
     values=[x[3]for x in history.samples if x[2]==0 and begins[0]['monotonic']<=x[0]<=ends[0]['monotonic']]
     census=begins[0]['value'];storage=[census[16+i*8:16+(i+1)*8]for i in range(24)];expected_bytes=64880640 if mode else 77856768
     assert census[2]==census[3]and census[5]==24 and all(x[1]==expected_bytes for x in storage)and len({x[0]for x in storage})==24
     phase_observations.append(dict(phase=phase,mode=mode,driver_observed_min=min(values)if values else None,samples=len(values),after_24allocation_driver_free=census[0],persistent_storage_declared_bytes=sum(x[1]for x in storage),scope='observed interval after24allocation through20query-stage metadata/prefix/FA/readback; begin clear+trim/allocation transient retained in globalhistory separately',Torch_begin=begins[0]['Torch'],Torch_end=ends[0]['Torch']))
   q['phase_memory']=phase_observations;mins={p['phase']:p['driver_observed_min']for p in phase_observations}
   q['observed_capacity_pairs']=[dict(OFF=a,ON=b,observed_ON_minus_OFF_minfree=mins[b]-mins[a],net_atleast_295698432=(mins[b]-mins[a]>=295698432))for a,b in [(0,1),(3,2)]if a in mins and b in mins and mins[a]is not None and mins[b]is not None]
   q['observed_warm_net_capacity_sufficient']=any(p['OFF']==3 and p['ON']==2 and p['net_atleast_295698432']for p in q['observed_capacity_pairs'])
   observed_points=[x['free']for x in q['driver_initialization_points']if x['rc']==0];q['all_observed_min_free']=min(observed_points+[q['driver_min_observed']]if q['driver_min_observed']is not None else observed_points,default=None)
   q['telemetry_floor_passed']=not history.errors and q['all_observed_min_free']is not None and q['all_observed_min_free']>=4294967296
   q['passed']=bool(q.get('mechanical_passed')and q['normal_Runtime_close']and q.get('numeric_agreement')and q['telemetry_floor_passed']and q['observed_warm_net_capacity_sufficient'])
   q['qualified']=False;q['capacity_scope']='component observed warmphase2ON/3OFF driver difference only, with monotonic unresetTorch/history; missed5ms/initialization peaks UNKNOWN. Even sufficient is not Model fit or global memory upper bound.'
  q.update(completed=True,elapsed_s=time.time()-epoch,Model_GO=False,Model_fit=False);q['maps']=[l for l in Path('/proc/self/maps').read_text().splitlines()if any(x in l for x in ['libKimi','libinfinilm_runtime','libinfiniops','libcudart','flash_attn'])];publish(result,q);print(json.dumps({k:q.get(k)for k in ['actual_step','passed','numeric_agreement','normal_Runtime_close','error','elapsed_s']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
