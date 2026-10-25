import pathlib,json,hashlib,ctypes,numpy as np,time,os,traceback
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-gate-manifest-r9032.json').read_text());job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Kimi-gate-real-r9032-'+job+'.json');assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'status':'FAILED','normal_close':False,'contexts':{},'comparisons':{},'CPU_binding_step':m['producer'],'scope':'Two realnorm0 rows536/537 +6geometry-onlyzero companions, TP16rank0 actual weights; noModel/KDA/o_proj/CCL','GPU_F64':False};ctx=None;failure=None
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def diff(a,b):
 af=(a.astype(np.uint32)<<16).view(np.float32).astype(np.float64);bf=(b.astype(np.uint32)<<16).view(np.float32).astype(np.float64);delta=af-bf
 return {'BF16_different_elements':int(np.count_nonzero(a!=b)),'max_abs':float(np.max(abs(delta))),'relative_RMS':float(np.sqrt(np.sum(delta*delta)/max(float(np.sum(af*af)),1e-30))),'M1_SHA256':digest(a),'M8_SHA256':digest(b)}
try:
 assert os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')=='1';assert sha(m['native'])==m['native_SHA256'];assert sha(m['norm_fixture']['input'])==m['norm_fixture']['input_SHA256'];x=np.fromfile(m['norm_fixture']['input'],dtype='<u2').reshape(8,7168);xh=digest(x)
 lib=ctypes.CDLL(m['native']);lib.proj_error.restype=ctypes.c_char_p;lib.gate_proj_create_r9032.argtypes=[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p];lib.gate_proj_create_r9032.restype=ctypes.c_void_p;lib.proj_run.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p];lib.proj_close.argtypes=lib.proj_readonly.argtypes=[ctypes.c_void_p];lib.proj_counts.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
 class Info(ctypes.Structure):_fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];q['actual_bindings']={}
 for name in ['gate_proj_create_r9032','proj_run','proj_close','proj_readonly',*m['provider_symbols']]:
  info=Info();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.file.decode());q['actual_bindings'][name]={'path':str(p),'SHA256':sha(p)}
 for name in m['provider_symbols']:assert q['actual_bindings'][name]['SHA256']==m['provider_DSO_SHA256']
 fa1=fa8=None
 for name in ['f_a_proj','b_proj','g_proj','f_b_proj']:
  spec=m['weights'][name];assert sha(spec['path'])==spec['SHA256'];n,k=spec['shape'];w=np.fromfile(spec['path'],dtype='<u2');wh=digest(w);ctx=lib.gate_proj_create_r9032(k,n,w.ctypes.data);assert ctx,lib.proj_error();entry={'closed_normally':False,'weight_SHA256':wh,'shape':[n,k],'walls_s':[]};q['contexts'][name]=entry
  try:
   assert lib.proj_readonly(ctx)==0,lib.proj_error()
   def run(a):
    a=np.ascontiguousarray(a);assert a.dtype==np.uint16 and a.shape[1]==k;before=digest(a);guard=np.full(len(a)*n+64,0x7fc1,dtype=np.uint16);y=guard[32:-32].reshape(len(a),n);t=time.perf_counter();assert lib.proj_run(ctx,len(a),a.ctypes.data,y.ctypes.data)==0,lib.proj_error();entry['walls_s'].append(time.perf_counter()-t);assert digest(a)==before;assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((y&0x7f80)==0x7f80);return y.copy()
   real=x[:2]if name!='f_b_proj'else fa1
   serial=[]
   for j in range(2):
    a=run(real[j:j+1]);b=run(real[j:j+1]);assert np.array_equal(a,b),'M1 ownrepeat';serial.append(a[0])
   serial=np.stack(serial)
   if name=='f_b_proj':
    same=np.zeros((8,k),np.uint16);same[:2]=fa1;full=same;entry['SAME_faM1_operand_SHA256']=digest(fa1)
   else:full=x
   eight=run(full);again=run(full);assert np.array_equal(eight,again),'M8 ownrepeat';q['comparisons'][name]=[{'position':pos,**diff(serial[j],eight[j])}for j,pos in enumerate([536,537])]
   if name=='f_a_proj':fa1=serial.copy();fa8=eight.copy()
   if name=='f_b_proj':
    chain=run(fa8);repeat=run(fa8);assert np.array_equal(chain,repeat),'chain ownrepeat';q['comparisons']['fa_to_fb_actual_chain']=[{'position':pos,**diff(serial[j],chain[j])}for j,pos in enumerate([536,537])];entry['actual_faM8_operand_SHA256']=digest(fa8)
   assert lib.proj_readonly(ctx)==0,lib.proj_error();assert digest(w)==wh;counts=(ctypes.c_uint64*3)();assert lib.proj_counts(ctx,counts)==0;entry['counts_M1_M8_rows']=list(counts);assert list(counts)==([4,4,36]if name=='f_b_proj'else[4,2,20]);entry.update(ownrepeat=True,finite=True,canary=True,host_input_readonly=True,GPU_weight_readonly=True)
   serial.tofile(r/'results'/('Kimi-gate-'+job+'-'+name+'-M1.bf16'));eight.tofile(r/'results'/('Kimi-gate-'+job+'-'+name+'-M8.bf16'))
  except BaseException as e:
   entry['original_exception']=repr(e);entry['original_traceback']=traceback.format_exc();raise
  finally:
   counts=(ctypes.c_uint64*3)()
   if lib.proj_counts(ctx,counts)==0:entry['actual_counts_before_close']=list(counts)
   assert lib.proj_close(ctx)==0,lib.proj_error();entry['closed_normally']=True;ctx=None
 assert digest(x)==xh;q['counts_all']=[sum(e['counts_M1_M8_rows'][j]for e in q['contexts'].values())for j in range(3)];assert q['counts_all']==[16,10,96];q['all_projection_comparisons_same']=all(e['BF16_different_elements']==0 for rows in q['comparisons'].values()for e in rows);q['rowlocal_transform_scope']='Only exact view fb[1,T,6,128]/beta[1,T,6]/g[1,T,768]; no approximate sigmoid, fabricatedKDA or gated-pre-oproj tensor. OriginalSDK nonlinear entry NOT_EXECUTED.'
 maps=pathlib.Path('/proc/self/maps').read_text();libs=sorted({x.split()[-1]for x in maps.splitlines()if any(n in x for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so'])});q['actual_runtime_dependencies_SHA256']={p:sha(p)for p in libs};q.update(status='GATE_PROJECTION_COMPONENT_DIAGNOSTIC_COMPLETE',normal_close=True,final_IDs_EOS='NOT_TESTED',Model_cause='NOT_PROVEN')
except BaseException as e:failure=e;q['exception']=repr(e);q['traceback']=traceback.format_exc()
finally:
 if ctx:
  try:lib.proj_close(ctx)
  except BaseException as e:q['close_error']=repr(e)
 q['source_SHA256']=sha(__file__);out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
if failure:raise failure
