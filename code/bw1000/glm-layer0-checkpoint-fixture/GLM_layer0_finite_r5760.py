"""NOT RUN. One GPU TP8rank0-local actualcheckpoint layer0; NOT historical479 QKV.
No selector/FFN/fullModel. CPU extraction exact checkpoint bytes and actual717 prepare.
"""
import os,pathlib,json,ctypes,time,hashlib
import torch
from GLM_layer0_extract_r5756 import extract
R=pathlib.Path('/refactor');V=ctypes.c_void_p;Z=ctypes.c_size_t
def main():
 torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=False
 for name in ['allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']:setattr(torch.backends.cuda.matmul,name,False)
 p=R/'results/glm-layer0-finite-r5760.json';assert not p.exists();q={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'scope':__doc__,'cohorts':[]}
 def save():t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
 base=R/'candidates/glm-layer0-fixture-r5756';proof=json.loads((base/'CPU-build.json').read_text());libpath=pathlib.Path(proof['library']);assert hashlib.sha256(libpath.read_bytes()).hexdigest()==proof['library_SHA256'];q['DSO_SHA256']=proof['library_SHA256'];q['CPU_receipt_SHA256']=hashlib.sha256((base/'CPU-build.json').read_bytes()).hexdigest()
 lib=ctypes.CDLL(str(libpath));lib.glm_l0_create_r5756.argtypes=[ctypes.c_int,ctypes.c_char_p];lib.glm_l0_create_r5756.restype=V
 lib.glm_l0_error_r5756.restype=ctypes.c_char_p;lib.glm_l0_keys_r5756.argtypes=[V];lib.glm_l0_keys_r5756.restype=ctypes.c_char_p
 lib.glm_l0_load_r5756.argtypes=[V,ctypes.c_char_p,V,Z,ctypes.POINTER(Z),ctypes.c_int];lib.glm_l0_finish_r5756.argtypes=[V];lib.glm_l0_prepare_r5756.argtypes=[V,Z,V,V,V];lib.glm_l0_append_check_r5756.argtypes=[V];lib.glm_l0_run_r5756.argtypes=[V,ctypes.c_int,V];lib.glm_l0_destroy_r5756.argtypes=[V,ctypes.POINTER(ctypes.c_int)]
 def ck(v):assert v==0,(v,lib.glm_l0_error_r5756().decode())
 class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',V),('sname',ctypes.c_char_p),('saddr',V)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[V,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int;q['actual_symbol_owners']={}
 for name in ['glm_l0_create_r5756','modern_glm_crossrequest_prefill_calls_r7328','modern_glm_dense_mode_r48_set']:
  a=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,name),V),ctypes.byref(a));assert pathlib.Path(a.fname.decode()).resolve()==libpath.resolve();q['actual_symbol_owners'][name]=a.fname.decode()
 lib.modern_glm_dense_mode_r48_set.argtypes=[ctypes.c_int];ck(lib.modern_glm_dense_mode_r48_set(2))
 cfg=json.loads(pathlib.Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_text());start=time.perf_counter();weights,embedding,ids=extract();q['CPU_extract_seconds']=time.perf_counter()-start
 # Only ~26MiB persistent parameters+compacttable, native workspace additional.
 assert torch.cuda.mem_get_info()[0]>=6*1024**3+64*1024**2
 ctx=lib.glm_l0_create_r5756(0,json.dumps(cfg).encode());assert ctx,lib.glm_l0_error_r5756().decode();closed=False
 try:
  assert set(json.loads(lib.glm_l0_keys_r5756(ctx)))==set(weights)
  for key,w in weights.items():
   blob=ctypes.create_string_buffer(w['bytes'],len(w['bytes']));shape=(Z*len(w['shape']))(*w['shape']);ck(lib.glm_l0_load_r5756(ctx,key.encode(),ctypes.cast(blob,V),len(shape),shape,{'F8_E4M3':0,'F32':1,'BF16':2}[w['dtype']]))
  ck(lib.glm_l0_finish_r5756(ctx));del weights
  table=torch.frombuffer(bytearray(embedding),dtype=torch.bfloat16).reshape(341,6144).cuda();positions=torch.arange(512,dtype=torch.int64).repeat(8).cuda();del embedding
  q['free_after_load']=torch.cuda.mem_get_info()[0];assert q['free_after_load']>=6*1024**3
  for cohort in range(2):
   ti=torch.tensor(ids[cohort*8:cohort*8+8],dtype=torch.int64,device='cuda').reshape(-1);torch.cuda.synchronize();t=time.perf_counter();ck(lib.glm_l0_prepare_r5756(ctx,cohort,V(table.data_ptr()),V(ti.data_ptr()),V(positions.data_ptr())));prep=time.perf_counter()-t
   ck(lib.glm_l0_append_check_r5756(ctx));outs=[torch.empty((4096,8,256),dtype=torch.bfloat16,device='cuda')for _ in range(3)]
   for mode in [0,1]:ck(lib.glm_l0_run_r5756(ctx,mode,V(outs[mode].data_ptr())))
   # Strict byte-copy contract for same old attention on identical-layout bank.
   assert torch.equal(outs[0].view(torch.uint8),outs[1].view(torch.uint8)), 'old fresh vsbank mismatch: stop before attributing candidate'
   ck(lib.glm_l0_run_r5756(ctx,2,V(outs[2].data_ptr())))
   a=outs[0].float().cpu();b=outs[2].float().cpu();err=b-a;metrics={'maxabs':float(err.abs().max()),'NRMSE':float((err.double().square().mean()/a.double().square().mean()).sqrt()),'exact_fraction':float((a==b).float().mean()),'finite':bool(torch.isfinite(a).all()and torch.isfinite(b).all()),'last_feature_argmax_differences':int((a.argmax(-1)!=b.argmax(-1)).sum())};assert metrics['finite']
   timings=[]
   for trial,mode in enumerate([0,2,2,0]):
    torch.cuda.synchronize();t=time.perf_counter();ck(lib.glm_l0_run_r5756(ctx,mode,V(outs[mode].data_ptr())));wall=time.perf_counter()-t;timings.append({'trial':trial,'mode':mode,'drained_wall_seconds':wall});assert torch.cuda.mem_get_info()[0]>=512*1024**2
   q['cohorts'].append({'cohort':cohort,'prepare_seconds':prep,'append_byteequal':True,'oldfresh_oldbank_byteequal':True,'numerics':metrics,'timings':timings,'free_after':torch.cuda.mem_get_info()[0]});save();del outs,a,b,err,ti
  expired=ctypes.c_int();closed=True;ck(lib.glm_l0_destroy_r5756(ctx,ctypes.byref(expired)));assert expired.value==1
  torch.cuda.synchronize();z=torch.ones(32,device='cuda');assert (z+1).sum().item()==64;q.update(completed=True,normal_close=True,Runtime_weak_expired=True,Torch_postclose=True);save()
 except BaseException as e:q['failure']=repr(e);save();raise
 finally:
  if not closed:
   expired=ctypes.c_int();rc=lib.glm_l0_destroy_r5756(ctx,ctypes.byref(expired));q['failure_close']={'return':rc,'expired':expired.value};save()
if __name__=='__main__':main()
