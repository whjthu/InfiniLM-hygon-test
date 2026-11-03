"""Two independent current BF16 lowrank projections; no MoE, Model, target or HTTP."""
from pathlib import Path
import ctypes as C,hashlib,json,os,socket,sys,time,traceback,threading
D=Path(__file__).resolve().parent;R=Path('/refactor');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(path,q):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def relative(a,b,np):
 x=a.astype(np.float64);y=b.astype(np.float64);finite=bool(np.isfinite(x).all()and np.isfinite(y).all())
 if not finite:return dict(finite=False,NRMSE=None,max_abs=None,reference_RMS=None,elements=int(x.size))
 delta=x-y
 return dict(finite=True,NRMSE=float(np.sqrt(np.mean(delta*delta))/max(float(np.sqrt(np.mean(y*y))),1e-30)),max_abs=float(np.max(np.abs(delta))),reference_RMS=float(np.sqrt(np.mean(y*y))),elements=int(x.size))
def main():
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi432-'+step+'.json');assert not out.exists()
 start=time.monotonic();epoch=float(os.environ['KIMI_R432_GPU_START']);deadline=epoch+540
 q=dict(completed=False,passed=False,actual_step=step,PID=os.getpid(),node=socket.gethostname().split('.')[0],source_manifest_SHA256=os.environ['KIMI_R432_GPU_SOURCE_SHA'],records=[],diagnostics=[],normal_destroy=False,Model=False,performance_audit_required=True,original_epoch=epoch,real_activations=False)
 publish(out,q);lib=None;ctx=None;weights=[]
 try:
  assert q['node']=='a02r3n03'and os.environ['SLURM_JOB_ID']=='910371'
  assert 'INFINILM_OWNER_CPU_WIDTH_R2392'not in os.environ,'No model8owner affinity policy'
  assert sha(D/'GPU_SOURCE_MANIFEST.json')==q['source_manifest_SHA256'];m=json.loads((D/'GPU_SOURCE_MANIFEST.json').read_bytes())
  for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
  receipt=Path(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];build=json.loads(receipt.read_bytes());assert build['passed']and sha(build['library'])==m['library_SHA256']
  for v in build['compiled']:
   assert sha(v['object'])==v['object_SHA256']and sha(v['MD'])==v['MD_SHA256']
   for p,h in v['dependency_SHA256'].items():assert sha(p)==h,p
  assert sha(m['BLAS_policy']['path'])==m['BLAS_policy']['SHA256']
  q['entry_affinity']=sorted(os.sched_getaffinity(0));assert len(q['entry_affinity'])>=4
  assert not any(x in sys.modules for x in ['torch','transformers','infinilm','_infinilm'])
  import numpy as np
  import contextlib,io
  config_text=io.StringIO()
  with contextlib.redirect_stdout(config_text):np.show_config()
  q['CPU_reference_provider']=dict(numpy_version=np.__version__,numpy_configuration=config_text.getvalue(),input_dtype='F32 expanded exactly from BF16',matmul_output_dtype='F32',metric_reductions='CPU F64 only, no GPU F64')
  def f32(a):return(a.astype(np.uint32)<<16).view(np.float32)
  def bf16(a):
   u=np.ascontiguousarray(a,dtype=np.float32).view(np.uint32);return((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
  index=Path('/shared/Kimi-K3/model.safetensors.index.json');assert sha(index)==m['checkpoint_index_SHA256'];mapping=json.loads(index.read_bytes())['weight_map'];weights_meta=[];loaded=time.monotonic()
  for case in m['cases']:
   assert time.time()<deadline
   path=index.parent/mapping[case['weight_key']];assert path.name=='model-00002-of-000096.safetensors'
   with path.open('rb')as f:
    headn=int.from_bytes(f.read(8),'little');assert 0<headn<64*1024*1024;header=f.read(headn);v=json.loads(header)[case['weight_key']];assert v['dtype']=='BF16'and v['shape']==[case['N'],case['K']]
    a,b=v['data_offsets'];assert b-a==51380224;f.seek(8+headn+a);raw=f.read(b-a);assert len(raw)==b-a
   w=np.frombuffer(raw,dtype='<u2').reshape(case['N'],case['K']).copy();assert w.flags.c_contiguous and np.isfinite(f32(w)).all();weights.append(w)
   weights_meta.append(dict(key=case['weight_key'],shape=v['shape'],dtype=v['dtype'],shard=str(path),shard_header_SHA256=hashlib.sha256(header).hexdigest(),offsets=[8+headn+a,8+headn+b],payload_SHA256=hashlib.sha256(w.tobytes()).hexdigest()))
  q.update(checkpoint_load_s=time.monotonic()-loaded,weights=weights_meta)
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12',mode=C.RTLD_GLOBAL);count=C.c_int();assert cuda.cudaGetDeviceCount(C.byref(count))==0 and count.value==1;buf=C.create_string_buffer(32);assert cuda.cudaDeviceGetPCIBusId(buf,32,0)==0
  q.update(visible_count=count.value,physical_PCI=buf.value.decode(),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),HIP_VISIBLE_DEVICES=os.environ.get('HIP_VISIBLE_DEVICES'))
  lib=C.CDLL(build['library'],mode=C.RTLD_LOCAL)
  lib.kimi_lowrank_error_r432.restype=C.c_char_p
  lib.kimi_lowrank_create_r432.argtypes=[C.c_int,C.c_void_p,C.c_void_p];lib.kimi_lowrank_create_r432.restype=C.c_void_p
  lib.kimi_lowrank_run_r432.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.POINTER(C.c_double)];lib.kimi_lowrank_run_r432.restype=C.c_int
  lib.kimi_lowrank_counts_r432.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t];lib.kimi_lowrank_layout_r432.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t];lib.kimi_lowrank_destroy_r432.argtypes=[C.c_void_p]
  def check(rc):
   if rc:raise RuntimeError('component rc='+str(rc)+' error='+str(lib.kimi_lowrank_error_r432()))
  q['mapped_libraries']={}
  for line in Path('/proc/self/maps').read_text().splitlines():
   p=line.split()[-1]
   if p.startswith('/')and any(v in p for v in ['libKimi_lowrank_M1_r432','libinfiniops.so','libinfinilm_runtime.so','_infinilm.cpython','libcublas.so','libcudart.so']):q['mapped_libraries'][p]=sha(p)
  for p,h in build['library_pins'].items():
   if any(s in p for s in ['libinfiniops.so','libinfinilm_runtime.so','_infinilm.cpython','libcublas.so','libcudart.so']):assert sha(p)==h,p
  created=time.monotonic();ctx=lib.kimi_lowrank_create_r432(0,weights[0].ctypes.data,weights[1].ctypes.data);assert ctx,str(lib.kimi_lowrank_error_r432());q['create_s']=time.monotonic()-created;q['affinity_after_create']=sorted(os.sched_getaffinity(0));assert q['affinity_after_create']==q['entry_affinity']
  layout=(C.c_uint64*10)();check(lib.kimi_lowrank_layout_r432(ctx,layout,10));q['actual_weight_layout']=list(layout);assert list(layout)==[3584,7168,7168,1,1,7168,3584,3584,1,1]
  # Real owner-rejection call does not touch Runtime or poison the valid owner.
  wrong=[]
  def reject():wrong.append(lib.kimi_lowrank_counts_r432(ctx,(C.c_uint64*4)(),4))
  t=threading.Thread(target=reject);t.start();t.join(5);assert not t.is_alive()and wrong==[1];q['wrong_owner_rejected']=True
  references=[]
  for projection,case in enumerate(m['cases']):
   k,n=case['K'],case['N'];x=bf16(np.sin(np.arange(k,dtype=np.float32)*np.float32(.013+projection*.002))*np.float32(.1));xhash=hashlib.sha256(x.tobytes()).hexdigest()
   ref_start=time.monotonic();reference_f32=f32(weights[projection])@f32(x);reference=bf16(reference_f32);q.setdefault('independent_CPU_reference',[]).append(dict(projection=projection,F32_dot_then_BF16=True,GPU_F64=False,seconds=time.monotonic()-ref_start))
   artifacts={}
   for tag,array,dtype in [('input',x,'BF16'),('CPUref',reference,'BF16'),('CPUref_F32',reference_f32,'F32')]:
    path=R/'results'/('Kimi432-'+step+'-p%d-'%projection+tag+'.bin');raw=array.tobytes();path.write_bytes(raw);artifacts[tag]=dict(path=str(path),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),dtype=dtype,shape=list(array.shape),endianness='little')
   q.setdefault('input_reference_raw_artifacts',[]).append(artifacts);publish(out,q)
   base_output=None
   for slot,mode in enumerate([1]+m['modes']):
    assert time.time()<deadline,'Original driver540s deadline'
    guarded=int(slot==0);output=np.full(n,0x7fc1,dtype=np.uint16);metrics=(C.c_double*6)();host0=time.monotonic();rc=lib.kimi_lowrank_run_r432(ctx,projection,mode,x.ctypes.data,output.ctypes.data,guarded,metrics);host=time.monotonic()-host0
    # Preserve completed query/fields and original bytes before any diagnostic guard.
    fn=R/'results'/('Kimi432-'+step+'-p%d-slot%d-output.u16'%(projection,slot));raw=output.tobytes();fn.write_bytes(raw)
    record=dict(projection=projection,case=case['name'],slot=slot,mode=mode,guarded_diagnostic=bool(guarded),rc=rc,metrics=dict(zip(['input_seed_s','allocation_dispatch_drain_s','GPU_stream_event_ms','output_D2H_s','complete_native_call_s','canary_ok'],list(metrics))),ctypes_dispatch_checked_s=host,input_SHA256=xhash,output_SHA256=hashlib.sha256(raw).hexdigest(),output_path=str(fn),output_bytes=len(raw),output_dtype='BF16',output_shape=[n],output_endianness='little',vs_CPU_reference=relative(f32(output),f32(reference),np))
    record['output_valid_by_producer']=rc==0
    if rc:record['producer_error']=str(lib.kimi_lowrank_error_r432())
    if slot==1 and rc==0:base_output=output.copy()
    if base_output is not None:record['vs_original_OFF']=relative(f32(output),f32(base_output),np)
    (q['diagnostics']if guarded else q['records']).append(record);publish(out,q)
    check(rc);assert metrics[5]==1 and record['vs_CPU_reference']['finite'],'Canary/nonfinite stop'
    assert record['vs_CPU_reference']['NRMSE']<=.02,'F32reference numerical diagnostic threshold stop'
    if 'vs_original_OFF'in record:assert record['vs_original_OFF']['NRMSE']<=.02,'OFF relative numerical diagnostic threshold stop'
    assert hashlib.sha256(x.tobytes()).hexdigest()==xhash
   references.append(reference)
  policy=C.CDLL(None).modern_blas_policy_r243_json;policy.restype=C.c_char_p;q['actual_BLAS_policy']=json.loads(policy().decode());assert q['actual_BLAS_policy']['failures']==0 and q['actual_BLAS_policy']['owners']and all(v['created']==v['verified']for v in q['actual_BLAS_policy']['owners'])
  q['host_weights_after_SHA256']=[hashlib.sha256(w.tobytes()).hexdigest()for w in weights];assert q['host_weights_after_SHA256']==[v['payload_SHA256']for v in weights_meta]
  counts=(C.c_uint64*4)();check(lib.kimi_lowrank_counts_r432(ctx,counts,4));q['actual_call_counts']=list(counts);assert list(counts)==[3,4,3,4]
  assert len(q['records'])==12 and len(q['diagnostics'])==2
  q.update(passed=True,NUMERIC_scope='Two projections/one fixed synthetic BF16 input each/current actual checkpoint, diagnostic not Modelquality.',device_weight_bytes_after_readback='UNKNOWN; HOST checkpoint bytes immutable and device W only passed const to GEMV/read-only original GEMM.')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc(),passed=False)
 finally:
  if ctx:
   try:rc=lib.kimi_lowrank_destroy_r432(ctx);q['destroy_rc']=rc;q['normal_destroy']=rc==0;ctx=None;assert rc==0
   except BaseException as error:q.setdefault('cleanup_errors',[]).append(repr(error));q['passed']=False
  q.update(completed=True,complete_process_wall_s=time.monotonic()-start,driver_observer_cost='Raw output files and CPU diagnostics included in processwall; outside native CAPI and event interval',final_affinity=sorted(os.sched_getaffinity(0)));publish(out,q)
 print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','normal_destroy','error']}),flush=True);return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
