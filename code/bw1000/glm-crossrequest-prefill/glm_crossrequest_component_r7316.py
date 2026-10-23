"""Finite eight-owner component, deterministic synthetic QKV; no Model or weights."""
import pathlib,os,json,hashlib,ctypes,time,threading,concurrent.futures,traceback
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('glm-crossrequest-component-r7316-'+tag+'.json');assert not out.exists()
import torch,numpy as np
torch.set_num_threads(1);torch.cuda.set_device(0) # GPU context BEFORE any GPU DSO static constructors.
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
assert torch.cuda.device_count()==8
receipt=json.loads((r/'candidates/glm-crossrequest-attention-r7314/CPU-build.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert receipt['completed']and receipt['producer']=='910374.713'and receipt['library_SHA256']=='cb532ff5870415b450ed89dbfdf77f82430aaa6e947d6d2068df006d1236036f'
for p,h in receipt['Runtime_and_extension_SHA256'].items():assert sha(p)==h
for n,h in receipt['SDK_libraries_SHA256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/n)==h
assert sha(receipt['library'])==receipt['library_SHA256'];lib=ctypes.CDLL(receipt['library'])
lib.glm_crossrequest_create_r7314.argtypes=[ctypes.c_int];lib.glm_crossrequest_create_r7314.restype=ctypes.c_void_p
lib.glm_crossrequest_forward_r7314.argtypes=[ctypes.c_void_p,ctypes.c_int,*([ctypes.c_void_p]*6),ctypes.POINTER(ctypes.c_double)]
lib.glm_crossrequest_destroy_r7314.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];lib.glm_crossrequest_error_r7314.restype=ctypes.c_char_p
class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
providers={}
for name,expected in [('glm_crossrequest_forward_r7314',pathlib.Path(receipt['library'])),('modern_glm_dense_mode_r48_set',r/'candidates/glm-rows16-model-r5707/libglm_rows16_model_r5707.so'),('modern_glm_dense_causal_mask_r25',r/'candidates/glm-rows16-model-r5707/libglm_rows16_model_r5707.so'),('_ZN8infinilm6models11glm_moe_dsa31sparse_attention_mapped_boundedERNS_7runtime7RuntimeERKNS2_6TensorES7_S7_S7_S7_mf',r/'candidates/glm-rows16-model-r5707/libglm_rows16_model_r5707.so')]:
 info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.fname.decode()).resolve();assert p==expected.resolve(),(name,str(p));providers[name]={'path':str(p),'SHA256':sha(p)}
barrier=threading.Barrier(8,timeout=100);arms=[('warm-O',0),('warm-N',1),('O0',0),('N0',1),('N1',1),('O1',0),('O2',0),('N2',1),('N3',1),('O3',0),('restore',0)]
def worker(rank):
 torch.cuda.set_device(rank);report={'rank':rank,'owner_thread':threading.get_ident(),'completed':False,'arms':[]};ctx=None
 ownerpath=r/'results'/('glm-crossrequest-component-r7316-'+tag+'-rank'+str(rank)+'.json');assert not ownerpath.exists()
 def save():
  tmp=ownerpath.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(ownerpath)
 try:
  generator=torch.Generator(device='cpu').manual_seed(731600+rank)
  arrays=[(torch.randn((8,512,8,256),generator=generator)*s).to(torch.bfloat16).cuda(rank)for s in[.25,.25,.5]]
  # Eight distinct requests with full causal-set indices; supplied order differs by request.
  positions=torch.arange(512,dtype=torch.int64).cuda(rank);indices=torch.stack([torch.roll(torch.arange(512,dtype=torch.int64),b*17).repeat(512,1)for b in range(8)]).cuda(rank)
  output=torch.empty_like(arrays[0]);torch.cuda.synchronize(rank);ctx=lib.glm_crossrequest_create_r7314(rank);assert ctx,lib.glm_crossrequest_error_r7314().decode()
  report['input_layouts']=[{'shape':list(t.shape),'strides_elements':list(t.stride()),'dtype':str(t.dtype)}for t in[*arrays,indices,positions]]
  report['input_fingerprints']=[hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()for t in arrays]
  reference=None
  for label,on in arms:
   barrier.wait();seconds=ctypes.c_double(-1);before=time.perf_counter();code=lib.glm_crossrequest_forward_r7314(ctx,on,*[ctypes.c_void_p(t.data_ptr())for t in[*arrays,indices,positions,output]],ctypes.byref(seconds));callwall=time.perf_counter()-before
   assert code==0,(label,lib.glm_crossrequest_error_r7314().decode())
   # Observation is after both native clocks, included in driver loop elapsed only.
   actual=output.float().cpu().numpy();finite=bool(np.isfinite(actual).all());assert finite,label
   if reference is None:reference=actual.copy()
   diff=actual-reference;rmse=float(np.sqrt(np.mean(diff.astype(np.float64)**2)));scale=float(np.sqrt(np.mean(reference.astype(np.float64)**2)))
   record={'label':label,'candidate':bool(on),'native_attention_work_wall_seconds':seconds.value,'complete_CAPI_call_wall_seconds':callwall,'finite':finite,'exact_elements':int(np.count_nonzero(actual==reference)),'total_elements':actual.size,'max_abs_error':float(np.max(np.abs(diff))),'NRMSE':rmse/max(scale,1e-20),'component_lastaxis_argmax_equal':int(np.count_nonzero(actual.argmax(-1)==reference.argmax(-1))),'component_argmax_rows':int(actual[...,0].size),'output_F32_SHA256':hashlib.sha256(actual.tobytes()).hexdigest()}
   report['arms'].append(record);save();assert record['NRMSE']<.01,(label,record)
  report['completed']=True
 except BaseException as e:
  report.update(error=repr(e),traceback=traceback.format_exc());barrier.abort()
 finally:
  if ctx:
   expired=ctypes.c_int();report['normal_destroy_code']=lib.glm_crossrequest_destroy_r7314(ctx,ctypes.byref(expired));report['Runtime_expired']=expired.value
  report['final_GPU_memory']=list(torch.cuda.mem_get_info(rank))
  save()
 return report
started=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:owners=list(pool.map(worker,range(8)))
result={'producer':tag,'library_SHA256':receipt['library_SHA256'],'source_SHA256':receipt['source_SHA256'],'providers':providers,'owners':owners,'driver_wall_includes_CPU_observation_seconds':time.perf_counter()-started,'scope':'BW1000 single-node TP8-shaped independent attention components. No TP collective, weight, Engine, Model, final-token gate or service TPS. Synthetic unique BF16 QKV exact production shapes. Native clock excludes CAPI pointer wrapping/pre-drain, includes metadata D2H/validation, QKV pack, QK/PV, scale/round/mask/Softmax, outputscatter+completiondrain. Complete CAPI timer includes pointer checks/pre-drain. CPU F64 error statistics only; no GPU F64. Candidate changes headbatch8->64 and strides, may change provider algorithm despite unchanged per-head M/N/K; full numerical differences reported.'}
result['completed']=all(v['completed']and v.get('normal_destroy_code')==0 and v.get('Runtime_expired')==1 for v in owners)
out.write_text(json.dumps(result,indent=2));print(json.dumps({'path':str(out),'completed':result['completed'],'SHA256':sha(out)}));assert result['completed']
