import pathlib,json,hashlib,ctypes,numpy as np,subprocess,time,os,traceback
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-projection-manifest-r9030.json').read_text());job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Kimi-projection-real-r9030-'+job+'.json');assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'status':'FAILED','normal_close':False,'scope':m['fixture']['scope'],'GPU_F64':False,'dense_prefill_env':os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')};ctx=None;failure=None
try:
 assert q['dense_prefill_env']=='1'
 for k in ['weight','input']:assert sha(m['fixture'][k])==m['fixture'][k+'_SHA256']
 w=np.fromfile(m['fixture']['weight'],dtype='<u2');x=np.fromfile(m['fixture']['input'],dtype='<u2').reshape(8,7168);wh=hashlib.sha256(w.tobytes()).hexdigest();xh=hashlib.sha256(x.tobytes()).hexdigest()
 assert sha(m['native'])==m['native_SHA256'];lib=ctypes.CDLL(m['native']);lib.proj_error.restype=ctypes.c_char_p;lib.proj_create.argtypes=[ctypes.c_void_p];lib.proj_create.restype=ctypes.c_void_p;lib.proj_run.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p];lib.proj_close.argtypes=lib.proj_readonly.argtypes=[ctypes.c_void_p];lib.proj_counts.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
 class Info(ctypes.Structure):_fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];bindings={}
 symbols=m['provider_symbols'];assert any('7forward'in s for s in symbols)and any('C1E'in s for s in symbols)
 for name in ['proj_create','proj_run','proj_close','proj_readonly',*symbols]:
  info=Info();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.file.decode());bindings[name]={'path':str(p),'SHA256':sha(p)}
 for symbol in symbols:assert bindings[symbol]['SHA256']==m['provider_DSO_SHA256']
 q['actual_bindings']=bindings
 ctx=lib.proj_create(w.ctypes.data);assert ctx,lib.proj_error();assert lib.proj_readonly(ctx)==0,lib.proj_error()
 def run(a):
  n=len(a);guard=np.full(n*2304+64,0x7fc1,dtype=np.uint16);y=guard[32:-32].reshape(n,2304);t=time.perf_counter();rc=lib.proj_run(ctx,n,a.ctypes.data,y.ctypes.data);wall=time.perf_counter()-t;assert rc==0,lib.proj_error();assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((y&0x7f80)==0x7f80);return y.copy(),wall
 serial=[];walls=[]
 for j in range(2):
  a,t=run(x[j:j+1]);b,u=run(x[j:j+1]);assert np.array_equal(a,b),'M1 ownrepeat';serial.append(a[0]);walls.extend([t,u])
 eight,t=run(x);repeat,u=run(x);assert np.array_equal(eight,repeat),'M8 ownrepeat';assert lib.proj_readonly(ctx)==0,lib.proj_error();assert hashlib.sha256(w.tobytes()).hexdigest()==wh and hashlib.sha256(x.tobytes()).hexdigest()==xh
 q['comparisons']=[]
 for j,pos in enumerate([536,537]):
  a=serial[j];b=eight[j];af=(a.astype(np.uint32)<<16).view(np.float32).astype(np.float64);bf=(b.astype(np.uint32)<<16).view(np.float32).astype(np.float64);delta=af-bf;q['comparisons'].append({'actual_input_position':pos,'BF16_different_elements':int(np.count_nonzero(a!=b)),'max_abs':float(np.max(abs(delta))),'relative_RMS':float(np.sqrt(np.sum(delta*delta)/max(float(np.sum(af*af)),1e-30))),'M1_SHA256':hashlib.sha256(a.tobytes()).hexdigest(),'M8_SHA256':hashlib.sha256(b.tobytes()).hexdigest()})
 counts=(ctypes.c_uint64*3)();assert lib.proj_counts(ctx,counts)==0 and list(counts)==[4,2,20];q.update(status='QKV_PROJECTION_COMPONENT_DIAGNOSTIC_COMPLETE',counts_M1_M8_rows=list(counts),M1_CAPI_walls_s=walls,M8_CAPI_walls_s=[t,u],ownrepeat=True,finite=True,canary=True,host_input_readonly=True,GPU_weight_readonly=True,raw_outputs='remote-only',Model_cause='NOT_PROVEN',final_IDs_EOS='NOT_TESTED')
 np.stack(serial).tofile(r/'results'/('Kimi-projection-'+job+'-M1.bf16'));eight.tofile(r/'results'/('Kimi-projection-'+job+'-M8.bf16'))
 maps=pathlib.Path('/proc/self/maps').read_text();libs=sorted({x.split()[-1]for x in maps.splitlines()if any(n in x for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so'])});q['actual_runtime_dependencies_SHA256']={p:sha(p)for p in libs}
except BaseException as e:failure=e;q['exception']=repr(e);q['traceback']=traceback.format_exc()
finally:
 if ctx:
  try:assert lib.proj_close(ctx)==0,lib.proj_error();q['normal_close']=True
  except BaseException as e:q['close_error']=repr(e);failure=failure or e
 q['source_SHA256']=sha(__file__);out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
if failure:raise failure
