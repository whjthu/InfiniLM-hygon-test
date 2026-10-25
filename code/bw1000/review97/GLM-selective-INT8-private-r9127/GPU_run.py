import pathlib,json,hashlib,ctypes,os,time,traceback,statistics,socket
import numpy as np,torch
from conversion import fp8_blocks_to_i8_channels,per_row_i8,error_summary,i8_gemm_oracle
r=pathlib.Path('/refactor');b=r/'candidates/GLM-selective-INT8-r9127';receipt=json.loads((b/'CPU-closure-b.json').read_text());assert receipt['completed'];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':socket.gethostname(),'cases':[],'scope':'actual checkpoint TP8 expert0 layer3 weights, synthetic seeded activations, operator only not model/task quality/performance','baseline':'Torch BF16 resident dequantized weight GEMM; not original GLM FP8 provider','CPU_closure_SHA256':sha(b/'CPU-closure-b.json'),'library_SHA256':sha(receipt['library']),'normal_owner_cleanup':False};assert q['library_SHA256']==receipt['library_SHA256'];lib=None
outpath=r/'results'/('GLM-I8-Q1-r9127-'+q['producer']+'.json')
def save():outpath.write_text(json.dumps(q,indent=2))
try:
 torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 lib=ctypes.CDLL(receipt['library']);lib.glm_i8_error_r9127.restype=ctypes.c_char_p;lib.glm_i8_run_r9127.argtypes=[ctypes.c_size_t]*3+[ctypes.c_void_p]*7;lib.glm_i8_run_r9127.restype=ctypes.c_int;lib.glm_i8_shutdown_r9127.restype=ctypes.c_int
 model=pathlib.Path(receipt['identity']['path']);assert sha(model/'config.json')==receipt['identity']['config_SHA256'] and sha(model/'model.safetensors.index.json')==receipt['identity']['index_SHA256'];headers=receipt['selected_headers'];prefix='model.layers.3.mlp.experts.0.'
 # Direct bounded reads from frozen safetensors headers; no unsupported CPU float8 tensor operations or whole shard reads.
 def tensor(key,rows,cols):
  h=headers[key];path=model/h['file'];assert path.stat().st_size==h['file_size']
  with path.open('rb')as f:hlen=int.from_bytes(f.read(8),'little');raw=f.read(hlen);assert hashlib.sha256(raw).hexdigest()==h['header_SHA256']
  md=h['tensor'];dtype=np.uint8 if md['dtype']=='F8_E4M3'else np.float32;mm=np.memmap(path,dtype=dtype,mode='r',offset=8+hlen+md['data_offsets'][0],shape=tuple(md['shape']));return np.array(mm[rows,cols],copy=True)
 weights=[]
 for name,k,n in [('w13',6144,512),('down',256,6144)]:
  parts=[]
  for proj in (['gate_proj','up_proj']if name=='w13'else ['down_proj']):
   rows=slice(0,256)if name=='w13'else slice(None);cols=slice(None)if name=='w13'else slice(0,256);codes=tensor(prefix+proj+'.weight',rows,cols);sc=tensor(prefix+proj+'.weight_scale_inv',slice(None),slice(None));packed,scale,truth=fp8_blocks_to_i8_channels(codes,sc);parts.append((packed,scale,truth))
  wi=np.concatenate([x[0]for x in parts]);ws=np.concatenate([x[1]for x in parts]);wf=np.concatenate([x[2]for x in parts]);assert wi.shape==(n,k);weights.append((name,k,n,wi,ws,wf))
  q.setdefault('weights',[]).append({'name':name,'K':k,'N':n,'TP_rank':0,'weight_error':error_summary(wi.astype(np.float32)*ws,wf),'sha_packed':hashlib.sha256(wi.tobytes()).hexdigest(),'sha_original_reconstructed':hashlib.sha256(wf.tobytes()).hexdigest()})
 q['actual_map_before']=sorted(set(x.split()[-1]for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if '.so'in x and ('infini'in x or 'cublas'in x or 'cudart'in x)));save()
 rng=np.random.default_rng(20261026)
 for name,k,n,wi,ws,wf in weights:
  w=torch.from_numpy(wi).cuda();scale=torch.from_numpy(ws).cuda();wb=torch.from_numpy(wf).to(device='cuda',dtype=torch.bfloat16);wcopy=w.clone();scopy=scale.clone()
  for m in [1,4,8,16,128]:
   a=torch.from_numpy(rng.normal(size=(m+1,k)).astype(np.float32)).to(device='cuda',dtype=torch.bfloat16);a[-1].fill_(17);acopy=a.clone();qa=torch.full((m+1,k),71,device='cuda',dtype=torch.int8);asc=torch.full((m+1,1),37,device='cuda',dtype=torch.float32);y=torch.full((m+1,n),19,device='cuda',dtype=torch.bfloat16);yb=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
   def cand():
    z=lib.glm_i8_run_r9127(m,n,k,a.data_ptr(),qa.data_ptr(),asc.data_ptr(),w.data_ptr(),scale.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
    if z:raise RuntimeError(lib.glm_i8_error_r9127().decode())
   def base():torch.mm(a[:m],wb.T,out=yb);torch.cuda.synchronize()
   for _ in range(3):base();cand()
   times={'baseline':[],'candidate':[]}
   for pair in range(4):
    for label,fn in ([('baseline',base),('candidate',cand)]if pair%2==0 else [('candidate',cand),('baseline',base)]):
     torch.cuda.synchronize();t=time.perf_counter()
     for _ in range(20):fn()
     times[label].append((time.perf_counter()-t)*1000/20)
   assert torch.equal(a,acopy)and torch.equal(w,wcopy)and torch.equal(scale,scopy)and bool((qa[-1]==71).all())and bool((asc[-1]==37).all())and bool((y[-1]==19).all())and bool(torch.isfinite(y[:m]).all())
   ap=a[:m].float().cpu().numpy();actual=y[:m].float().cpu().numpy();qi,qas=per_row_i8(ap);ref_quant=(qi.astype(np.float32)*qas)@(wi.astype(np.float32)*ws).T;ref_float=ap@wf.T;err=error_summary(actual,ref_quant);assert err['relative_RMS']<.01
   item={'name':name,'M':m,'K':k,'N':n,'wall_ms':times,'candidate_mean_ms':statistics.mean(times['candidate']),'baseline_mean_ms':statistics.mean(times['baseline']),'speedup_vs_BF16_reference':statistics.mean(times['baseline'])/statistics.mean(times['candidate']),'error_vs_quantized_float_oracle':err,'error_vs_original_FP8_weight_FP32_matmul':error_summary(actual,ref_float),'readonly_canary_finite':True,'dynamic_activation_error':error_summary(qi.astype(np.float32)*qas,ap)}
   if m==1:
    exact,acc=i8_gemm_oracle(qi,qas,wi,ws);item['error_vs_exact_I32_oracle']=error_summary(actual,exact);item['I32_absmax']=int(np.abs(acc).max());assert item['error_vs_exact_I32_oracle']['relative_RMS']<.01
   q['cases'].append(item);save();print(json.dumps({k:item[k]for k in ['name','M','candidate_mean_ms','baseline_mean_ms','speedup_vs_BF16_reference']}),flush=True)
 q['actual_dependency_SHA256']={path:sha(path)for path in receipt['dependencies']};assert q['actual_dependency_SHA256']==receipt['dependencies'];q['actual_map_after']=sorted(set(x.split()[-1]for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if '.so'in x and ('infini'in x or 'cublas'in x or 'cudart'in x)));assert lib.glm_i8_shutdown_r9127()==0;q['normal_owner_cleanup']=True;torch.cuda.synchronize();q['completed']=True
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
finally:
 if lib is not None and not q['normal_owner_cleanup']:q['cleanup_return']=lib.glm_i8_shutdown_r9127()
 save()
print(json.dumps({'completed':q['completed'],'cases':len(q['cases']),'error':q.get('error'),'normal_owner_cleanup':q['normal_owner_cleanup']}),flush=True);assert q['completed']
