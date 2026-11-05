"""One real Runtime; two actual geometries, synthetic finite BF16, no Model."""
from pathlib import Path
import ctypes as C,hashlib,json,os,socket,time,traceback,sys,io,contextlib
import numpy as np
from driver import run,GEOMETRIES
D=Path(__file__).resolve().parent;R=Path('/refactor');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def publish(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def f32(x):return(x.astype(np.uint32)<<16).view(np.float32)
def metrics(x,y):
 a=x.astype(np.float64);b=y.astype(np.float64);delta=a-b
 rms=float(np.sqrt(np.mean(b*b)));ma=float(np.max(np.abs(delta)));scaled=ma/max(1.,float(np.max(np.abs(b))))
 nrms=float(np.sqrt(np.mean(delta*delta)))/max(rms,1e-20)
 return dict(NRMS=nrms,maxabs=ma,maxabs_scaled=scaled,agreement=nrms<=.02 and scaled<=.04)
class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
def main():
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi461-mixed-'+step+'.json');assert not out.exists()
 epoch=float(os.environ['KIMI_MIXED_GPU_START']);deadline=epoch+150
 q=dict(completed=False,passed=False,numeric_agreement=None,Model=False,actual_step=step,PID=os.getpid(),UID=os.getuid(),node=socket.gethostname().split('.')[0],original_epoch=epoch,source_manifest_SHA256=os.environ['KIMI_MIXED_GPU_SOURCE_SHA'],normal_destroy=False,real_checkpoint_weights=False,synthetic_geometry_component=True)
 publish(out,q)
 try:
  assert q['node']=='a14r2n08'and os.environ['SLURM_JOB_ID']=='910380'
  assert 'INFINILM_OWNER_CPU_WIDTH_R2392'not in os.environ
  assert H(D/'GPU_SOURCE_MANIFEST.json')==q['source_manifest_SHA256'];m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
  receipt=Path(m['CPU_receipt']);assert H(receipt)==m['CPU_receipt_SHA256'];build=json.loads(receipt.read_bytes())
  assert build['passed']and build['actual_CPU']=='910407.1063'and not build['SDK_device_execution']
  assert H(build['library'])==m['library_SHA256']==build['library_SHA256']
  assert H(build['compiled']['object'])==build['compiled']['object_SHA256']
  assert H(Path(build['compiled']['object']).parent/'actual.d')==build['compiled']['actual_MD_SHA256']
  for p,h in build['exact_dependency_libraries'].items():assert H(p)==h,p
  assert H(m['baseline_source']['path'])==m['baseline_source']['SHA256']
  assert H(m['BLAS_policy']['path'])==m['BLAS_policy']['SHA256']
  q['entry_affinity']=sorted(os.sched_getaffinity(0));assert len(q['entry_affinity'])>=4
  assert not any(k in sys.modules for k in ['torch','infinilm','_infinilm'])
  cfg=io.StringIO()
  with contextlib.redirect_stdout(cfg):np.show_config()
  q['CPU_reference_provider']=dict(numpy=np.__version__,configuration=cfg.getvalue(),operands='F32 exact expansion of BF16',dot='NumPy CPU F32 matrix multiply',metrics='CPU F64 reductions only')
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12',mode=C.RTLD_GLOBAL)
  count=C.c_int();assert cuda.cudaGetDeviceCount(C.byref(count))==0 and count.value==1
  pci=C.create_string_buffer(32);assert cuda.cudaDeviceGetPCIBusId(pci,32,0)==0
  q.update(visible_count=count.value,physical_PCI=pci.value.decode(),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),HIP_VISIBLE_DEVICES=os.environ.get('HIP_VISIBLE_DEVICES'))
  lib=C.CDLL(build['library'],mode=C.RTLD_LOCAL)
  rtpath=next(p for p in build['exact_dependency_libraries']if p.endswith('/libinfinilm_runtime.so'));rtlib=C.CDLL(rtpath,mode=C.RTLD_LOCAL)
  stats=rtlib.modern_dense_prefill_stats_r2475;stats.restype=C.c_int;stats.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t]
  dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int
  q['symbols']={}
  for name,fn in [(n,getattr(lib,n))for n in build['CAPI']]+[('modern_dense_prefill_stats_r2475',stats)]:
   v=DlInfo();assert dl.dladdr(C.cast(fn,C.c_void_p),C.byref(v))and v.name
   path=Path(v.name.decode()).resolve();expected=Path(rtpath if name=='modern_dense_prefill_stats_r2475'else build['library']).resolve()
   assert path==expected;q['symbols'][name]=dict(path=str(path),SHA256=H(path))
  before=(C.c_uint64*5)();assert stats(0,before,5)==0;q['baseline_stats_before']=list(before);assert list(before)==[0]*5
  refs={};first_OFF={};samples=[];sources=[];private=R/'results'/('Kimi461-mixed-'+step+'-raw');private.mkdir()
  def buffers(gid,g):
   mm,k,n=g
   def words(size,seed):
    # Exact finite BF16 fractions. Generation is explicit synthetic, not activation capture.
    v=np.arange(size,dtype=np.uint32);v=v*np.uint32(1664525)+np.uint32(seed)
    f=((v>>24).astype(np.int16)-128).astype(np.float32)/np.float32(256)
    return(f.view(np.uint32)>>16).astype(np.uint16)
   x=words(mm*k,1013904223).reshape(mm,k);w=words(n*k,12345).reshape(n,k)
   assert np.isfinite(f32(x)).all()and np.isfinite(f32(w)).all()
   rows=sorted(set([0,1,mm//2,mm-1]));cols=sorted(set([0,n//3,2*n//3,n-1]))
   refs[gid]=(rows,cols,f32(x[rows,:])@f32(w[cols,:]).T)
   sources.append(dict(geometry=gid,input_shape=[mm,k],weight_shape=[n,k],dtype='BF16',weight_rows_contiguous=True,helper_weight_view_shape=[k,n],helper_weight_view_strides=[1,k],input_SHA256=hashlib.sha256(x.tobytes()).hexdigest(),weight_SHA256=hashlib.sha256(w.tobytes()).hexdigest(),CPU_sample_rows=rows,CPU_sample_columns=cols))
   return x,w
  def compare(gid,mode,result,x,w):
   rows,cols,ref=refs[gid];actual=f32(result[np.ix_(rows,cols)])
   cpu=metrics(actual,ref)
   if mode==0 and gid not in first_OFF:first_OFF[gid]=result.copy()
   off=metrics(f32(result),f32(first_OFF[gid]));v=dict(geometry=gid,mode=mode,CPU_F32_sample=cpu,vs_original_OFF=off)
   samples.append(v);return v
  def persist(gid,ordinal,result):
   p=private/('g'+str(gid)+'-arm'+str(ordinal)+'.bf16');raw=result.astype('<u2',copy=False).tobytes()
   with p.open('xb')as f:f.write(raw)
   return dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),dtype='BF16',endianness='little',shape=list(result.shape))
  q['driver']=run(lib,buffers,compare,persist,time.monotonic()+max(0,deadline-time.time()))
  q['normal_destroy']=q['driver']['cleanup_return']==0
  q['sources']=sources;q['numeric_agreement']=bool(samples)and len(samples)==12 and all(v['CPU_F32_sample']['agreement']and v['vs_original_OFF']['agreement']for v in samples)
  after=(C.c_uint64*5)();assert stats(0,after,5)==0;q['baseline_stats_after']=list(after)
  assert list(after)==[6,15360,154183680,132167680,0]
  q['final_affinity']=sorted(os.sched_getaffinity(0));assert q['entry_affinity']==q['final_affinity']
  q['maps']=[l for l in Path('/proc/self/maps').read_text().splitlines()if any(s in l for s in ['libKimi_mixed','libinfinilm_runtime','libinfiniops','libcudart','libblas_policy'])]
  q['observed_floor_bytes']=min(v for r in q['driver']['records']for v in r['driver_memory'][::2]);assert q['observed_floor_bytes']>=4*1024**3
  q['memory_scope']='real driver points/cache retained+explicit component-only warmtrim; not instantaneous peak; no Model fit'
  q['passed']=q['driver']['passed']and q['normal_destroy']
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q['completed']=True;q['elapsed_s']=time.time()-epoch;publish(out,q);print(json.dumps({k:q.get(k)for k in ['actual_step','completed','passed','numeric_agreement','normal_destroy','error','elapsed_s']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
