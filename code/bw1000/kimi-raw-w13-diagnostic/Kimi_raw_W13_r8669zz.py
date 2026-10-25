"""One synthetic caller-contract diagnosis; no checkpoint, CCL API or performance claim."""
import pathlib,json,hashlib,os,time
import numpy as np
from kimi_wfp4_repack_r8669zu import _permute
KS=[0,1,7,8,15,16,31,32,63,64,511,512,1023,2047,3582,3583]
E,H,I,N,R=896,3584,192,384,256
def code(e,n,k):
 with np.errstate(over='ignore'):
  v=np.asarray(e*np.uint32(1315423911),dtype=np.uint32)^np.asarray(n*np.uint32(2654435761),dtype=np.uint32)^np.asarray(k*np.uint32(2246822519),dtype=np.uint32)
 v=v^(v>>13);return ((v%7)+1+(((v>>7)&1)*8)).astype(np.uint8)
def scale(e,n,kblock):return np.asarray(122+(e*17+n*7+kblock*13)%11,dtype=np.uint8)
def construct():
 packed=np.empty((16,N,H//2),np.uint8);scales=np.empty((16,N,H//32),np.uint8);n=np.arange(N,dtype=np.uint32)[:,None];k=np.arange(H,dtype=np.uint32)[None,:]
 for e in range(16):
  c=code(np.uint32(e),n,k);packed[e]=c[:,::2]|(c[:,1::2]<<4);scales[e]=scale(e,n,np.arange(H//32)[None,:])
 _permute(packed.reshape(-1,H//2))
 # Independent scalar FP32 oracle: no qword load/decode/layout code reuse.
 lut=[0.,.5,1.,1.5,2.,3.,4.,6.];oracle=np.empty((16,16,N),np.float32)
 for t,kpoint in enumerate(KS):
  for e in range(16):
   for col in range(N):
    c=int(code(np.uint32(e),np.uint32(col),np.uint32(kpoint)));s=int(scale(e,col,kpoint//32));oracle[t,e,col]=np.float32((-1 if c&8 else 1)*lut[c&7]*2.**(s-127))
 assert len({tuple(oracle[:,:,col].reshape(-1))for col in range(N)})==N
 x=np.zeros((16,H),np.uint16);x[np.arange(16),KS]=0x3f80
 ids=np.tile(np.arange(16,dtype=np.int32),(16,1));assert set(ids.reshape(-1))==set(range(16))
 return dict(x=x,ids=ids,w13=packed,s13=scales,w2=np.zeros((16,H,I//2),np.uint8),s2=np.full((16,H,I//32),127,np.uint8),oracle=oracle.reshape(R,N))
def contract(v):
 assert v['w13'].shape==(16,N,H//2)and v['s13'].shape==(16,N,H//32)
 assert v['w2'].shape==(16,H,I//2)and v['s2'].shape==(16,H,I//32)
 assert v['ids'].dtype==np.int32 and v['oracle'].shape==(R,N)
 # CABI pack uses E896 counts and E896*T16*topk16 route slots; W13/W2
 # access expert IDs only from these16 active IDs. Never shrink E counts.
 return {'passed':True,'onehot_K':KS,'active_experts':16,'full_count_entries':896,'full_route_entries':896*256,'weight_and_scale_bytes':sum(v[n].nbytes for n in ['w13','s13','w2','s2']),'extra_required_bytes':410880,'raw_F32_bytes':R*N*4,'canary_bytes':256,'max_W13_byte_exclusive':16*N*(H//2),'max_W2_byte_exclusive':16*H*(I//2),'CPU_oracle':'Independent scalar E2M1*power2/onehot, no GPU F64 or dot-order tolerance','source_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
def run():
 import torch,ctypes
 p=pathlib.Path('/refactor/results')/('Kimi-raw-W13-r8669zz-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');assert not p.exists();out={'completed':False,'scope':'Synthetic directCABI rawW13 diagnosis only/no weights/CCL/Model/performance','phase':'CPU_construct'}
 def save():p.write_bytes((json.dumps(out,indent=2)+'\n').encode())
 try:
  v=construct();out['CPU_contract']=contract(v);save();torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);stream=torch.cuda.current_stream(0)
  libpath='/refactor/candidates/Kimi-official-WFP4-r8669zu/libKimi_official_WFP4_r8669zu.so';assert hashlib.sha256(pathlib.Path(libpath).read_bytes()).hexdigest()=='6b669a1a5f3e58caaaf78d441931e7098f56bb94cb80e558911fa5b402544d1d';lib=ctypes.CDLL(libpath);fn=lib.modern_kimi_official_WFP4_r8669zu;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*13;fn.restype=ctypes.c_int
  tensors={k:torch.from_numpy(v[k]).to('cuda')for k in ['x','ids','w13','s13','w2','s2']};counts=torch.zeros(E,dtype=torch.int32,device='cuda');routes=torch.full((E*R,),-1,dtype=torch.int32,device='cuda');extra=torch.full((410880+256,),0x5a,dtype=torch.uint8,device='cuda');activation=torch.zeros(R*I,dtype=torch.bfloat16,device='cuda');y=torch.zeros(16*H,dtype=torch.bfloat16,device='cuda');rw=torch.ones(R,dtype=torch.float32,device='cuda')
  torch.cuda.synchronize();out['phase']='single_direct_CABI';save()
  with torch.cuda.device(0),torch.cuda.stream(stream):
   assert torch.cuda.current_device()==0
   args=[y,activation,tensors['x'],tensors['ids'],rw,tensors['w13'],tensors['s13'],tensors['w2'],tensors['s2'],counts,routes,extra]
   out['returncode']=fn(0,*[z.data_ptr()for z in args],stream.cuda_stream);torch.cuda.synchronize();out['current_after']=torch.cuda.current_device()
  actual=extra[:R*N*4].view(torch.float32).cpu().numpy().reshape(R,N);expected=v['oracle'];error=np.abs(actual-expected);idx=np.unravel_index(int(error.argmax()),error.shape);out.update(raw_finite=bool(np.isfinite(actual).all()),maxabs=float(error.max()),RMSE=float(np.sqrt(np.mean(error**2,dtype=np.float32))),relativeRMS=float(np.sqrt(np.mean(error**2,dtype=np.float32))/np.sqrt(np.mean(expected**2,dtype=np.float32))),first_largest={'route':int(idx[0]),'token':int(idx[0]//16),'expert':int(idx[0]%16),'N':int(idx[1]),'K':KS[idx[0]//16],'expected':float(expected[idx]),'actual':float(actual[idx])},canary_intact=bool(torch.all(extra[410880:]==0x5a).item()),output_W2_zero=bool(torch.all(y==0).item()),counts=counts.cpu().tolist(),group_count=int(extra[410624:410628].view(torch.int32).cpu()[0]))
  # A single ready-to-interpret raw slice, tiny and synthetic only.
  out['selected_actual_expected']=[{'token':t,'K':KS[t],'expert':e,'N':n,'actual':float(actual[t*16+e,n]),'expected':float(expected[t*16+e,n])}for t in [0,1,7,15]for e in [0,7,15]for n in [0,1,7,8,15,16,31,32,63,64,191,192,383]]
  reference=expected.reshape(16,16,N);got=actual.reshape(16,16,N);signatures={tuple(reference[:,e,n]):(e,n)for e in range(16)for n in range(N)};matches=[]
  for e in range(16):
   for n in range(N):
    mapped=signatures.get(tuple(got[:,e,n]))
    if mapped is not None:matches.append({'expert':e,'N':n,'mapped_expert':mapped[0],'mapped_N':mapped[1]})
  out['single_signature_mapping_diagnostic']={'basis':'16onehotK signature/column/expert, exact discrete coefficients for classification only; not Modelqualitygate','matched_columns':len(matches),'total_columns':16*N,'identity_columns':sum(z['expert']==z['mapped_expert']and z['N']==z['mapped_N']for z in matches),'first_nonidentity':next((z for z in matches if z['expert']!=z['mapped_expert']or z['N']!=z['mapped_N']),None),'scope':'One dictionary lookup hypothesis only; unmatched means inconclusive, no grid/retry'}
  assert out['returncode']==0 and out['current_after']==0 and out['canary_intact'];assert out['counts']==[16]*16+[0]*(E-16)and out['group_count']==16
  del args,tensors,counts,routes,extra,activation,y,rw;torch.cuda.synchronize();torch.cuda.empty_cache();out['normal_owner_close']=True;out['completed']=True
 except BaseException as e:out['error']=repr(e);raise
 finally:out['phase']='completed'if out['completed']else'failed';save()
if __name__=='__main__':
 import sys
 if '--CPU-check'in sys.argv:
  q=contract(construct());print(json.dumps(q))
 else:run()
