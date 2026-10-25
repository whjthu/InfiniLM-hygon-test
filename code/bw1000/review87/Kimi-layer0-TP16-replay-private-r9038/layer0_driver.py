"""Bounded original layer0 A/C replay; raw activations remain remote."""
import pathlib,json,hashlib,sys,os,time,ctypes,threading,concurrent.futures,traceback
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def frontend(manifest):
 m=json.loads(pathlib.Path(manifest).read_text());assert m['TP']==16 and m['PP']==1 and len(m['nodes'])==2;assert m['arms']=={'A':[1]*32,'C':[1]*24+[8]};assert sha(m['native'])==m['native_SHA256'];assert sha(m['actual678_DSO'])==m['actual678_DSO_SHA256'];assert sha(m['provider_DSO'])==m['provider_DSO_SHA256'];assert sha(m['norm']['path'])==m['norm']['SHA256'];assert m['norm']['shape']==[544,7168];assert len(m['weights'])==16
 for rank in range(16):
  entries=m['weights'][str(rank)];assert len(entries)==14 and sum(x['bytes']for x in entries.values())==57208344
  for key,meta in entries.items():assert sha(meta['path'])==meta['SHA256'],(rank,key)
 for arm,widths in m['arms'].items():assert 512+sum(widths)==544
 return m,{'status':'CPU_FRONTEND_PASS_NO_GPU_OR_DLOPEN','TP':16,'PP':1,'ranks':16,'weights_per_rank':14,'weight_bytes_per_rank':57208344,'planned_native_calls_per_rank':59,'planned_native_calls_allranks':944,'actual_arms':m['arms'],'state_pool':{'conv':[17,2304,3],'KDA':[17,6,128,128]},'row_contract':'A inplace1; C prefill1 then alternate1/9 per fullaccept','norm_source_SHA256':m['norm']['SHA256']}
def main(manifest):
 import numpy as np,torch,infinilm
 from stage_control_r2386 import StageControl
 m,contract=frontend(manifest);r=pathlib.Path('/refactor');node=int(os.environ['SLURM_PROCID']);assert node in [0,1];torch.set_num_threads(1);assert torch.cuda.device_count()==8;job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('Kimi-layer0-real-r9038-'+job+'-node'+str(node)+'.json');assert not path.exists();report={'node':node,'job':job,'status':'RUNNING','scope':'OriginalTP16layer0only A/C actualnorm, nootherlayer/fullModel/finalIDs/servingspeed','CPU_frontend':contract,'workers':[{'rank':node*8+i,'normal_close':False}for i in range(8)],'raw_storage':'REMOTE_ONLY','GPU_F64':False};lock=threading.Lock()
 def save():
  with lock:p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(path)
 save();control=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],2,500);control.collect('frontend_ready',{'node':node,'contract':contract});norm=np.fromfile(m['norm']['path'],dtype='<u2').reshape(544,7168);config=pathlib.Path(m['config']).read_text();lib=ctypes.CDLL(m['native']);lib.layer0_error_r9038.restype=ctypes.c_char_p;lib.layer0_id_size_r9038.restype=ctypes.c_size_t;lib.layer0_id_r9038.argtypes=[ctypes.c_void_p];lib.layer0_create_r9038.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.layer0_create_r9038.restype=ctypes.c_void_p;lib.layer0_load_local_r9038.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
 for n in ['finish','reset','close']:getattr(lib,'layer0_'+n+'_r9038').argtypes=[ctypes.c_void_p]
 lib.layer0_run_r9038.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
 class Info(ctypes.Structure):_fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];report['actual_bindings']={}
 anchor=Info();assert dl.dladdr(ctypes.cast(lib.layer0_run_r9038,ctypes.c_void_p),ctypes.byref(anchor));native_base=anchor.base;assert sha(anchor.file.decode())==m['native_SHA256']
 for name,expected in m['provider_expected_SHA256'].items():
  address=ctypes.c_void_p.from_address(native_base+m['provider_relocation_slots'][name]).value;assert address;inf=Info();assert dl.dladdr(ctypes.c_void_p(address),ctypes.byref(inf));p=pathlib.Path(inf.file.decode());report['actual_bindings'][name]={'path':str(p),'SHA256':sha(p),'method':'actualnativeGOTrelocation','relocation_offset':m['provider_relocation_slots'][name]};assert sha(p)==expected,(name,str(p))
 def check(code):
  if code:raise RuntimeError(lib.layer0_error_r9038().decode())
 uid=(ctypes.c_ubyte*lib.layer0_id_size_r9038())();assert len(uid)>0 and len(uid)<1024
 if node==0:check(lib.layer0_id_r9038(uid))
 uids=control.collect('realTP16_uniqueID',{'id':bytes(uid).hex()if node==0 else None});assert uids[0]['id'];ctypes.memmove(uid,bytes.fromhex(uids[0]['id']),len(uid));barrier=threading.Barrier(8,timeout=480);root=r/'results'/('Kimi-layer0-raw-r9038-'+job)
 if node==0:
  root.mkdir(exist_ok=False)
  for arm in m['arms']:(root/arm).mkdir()
 control.collect('rawdirs_ready',{'node':node});os.environ['INFINILM_KIMI_KDA_TRACE_LAYER']='0';os.environ['INFINILM_KIMI_TRACE_STATE_ROWS_MODE_R8516']='all';os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/'A');report['raw_root']=str(root);save();shared_outputs={}
 def worker(device):
  rank=node*8+device;own=report['workers'][device];ctx=None
  try:
   affinity=sorted(os.sched_getaffinity(0));assert len(affinity)>=32;chosen=affinity[device*4:device*4+4];os.sched_setaffinity(threading.get_native_id(),chosen);own['owner_affinity']=sorted(os.sched_getaffinity(0));assert own['owner_affinity']==chosen;torch.cuda.set_device(device);ctx=lib.layer0_create_r9038(config.encode(),rank,device,uid);assert ctx,lib.layer0_error_r9038().decode();specs=m['weights'][str(rank)]
   for key,meta in sorted(specs.items()):
    w=np.fromfile(meta['path'],dtype='<u2'if meta['dtype']=='BF16'else'<f4').reshape(meta['shape']);assert sha(meta['path'])==meta['SHA256'];shape=(ctypes.c_size_t*w.ndim)(*w.shape);check(lib.layer0_load_local_r9038(ctx,key.encode(),w.ctypes.data,w.ndim,shape,1 if meta['dtype']=='BF16'else 0))
   check(lib.layer0_finish_r9038(ctx));own['loaded_weight_bytes']=sum(x['bytes']for x in specs.values());save();barrier.wait()
   if device==0:control.collect('all16_weights_ready',{'node':node,'loaded':8})
   barrier.wait()
   for arm,widths in m['arms'].items():
    if device==0:
     os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/arm);control.collect('arm:'+arm,{'node':node});report['phase']=arm;save()
    barrier.wait();check(lib.layer0_reset_r9038(ctx));barrier.wait();calls=[];outs={};active,shadow=1,9;past=0
    for width in [512,*widths]:
     read=1 if arm=='A'or past==0 else active;write=1 if arm=='A'or past==0 else shadow;x=np.ascontiguousarray(norm[past:past+width]);guard=np.full(width*7168+64,0x7fc1,np.uint16);out=guard[32:-32].reshape(width,7168);before=hashlib.sha256(x.tobytes()).hexdigest();t=time.perf_counter();check(lib.layer0_run_r9038(ctx,width,read,write,x.ctypes.data,out.ctypes.data));wall=time.perf_counter()-t;assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1),'canary';assert not np.any((out&0x7f80)==0x7f80),'nonfinite';assert hashlib.sha256(x.tobytes()).hexdigest()==before,'hostinputmutated';calls.append({'past':past,'width':width,'read':read,'write':write,'wall_s':wall,'output_SHA256':hashlib.sha256(out.tobytes()).hexdigest()})
     for p in [536,537]:
      if past<=p<past+width:outs[p]=out[p-past].copy()
     if arm!='A'and past>0:active,shadow=shadow,active
     past+=width
    assert past==544;own[arm]={'calls':calls,'native_calls':len(calls),'final_active':active,'final_shadow':shadow};shared_outputs[(device,arm)]=outs;save();barrier.wait()
    if device==0:control.collect('arm_closed:'+arm,{'node':node})
    barrier.wait()
   own['normal_close']=lib.layer0_close_r9038(ctx)==0;ctx=None;assert own['normal_close'];save()
  except BaseException as e:own['error']=repr(e);own['traceback']=traceback.format_exc();barrier.abort();save();raise
  finally:
   if ctx:
    try:own['normal_close_after_error']=lib.layer0_close_r9038(ctx)==0
    except BaseException as e:own['close_error']=repr(e)
   save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(worker,range(8)))
 assert all(z['normal_close']for z in report['workers']);report['phase']='analyze_existing_raw';save()
 def decode(raw,dtype):
  a=np.frombuffer(raw,dtype={'bf16':'<u2','f32':'<f4','i32':'<i4','i64':'<i8'}[dtype]);return (a.astype(np.uint32)<<16).view(np.float32).astype(np.float64)if dtype=='bf16'else a.astype(np.float64)
 def compare(ra,rb,dtype):
  a,b=decode(ra,dtype),decode(rb,dtype);assert a.shape==b.shape;d=a-b;return {'different_elements':int(np.count_nonzero(a!=b)),'byte_equal':ra==rb,'max_abs':float(np.max(abs(d)))if len(d)else 0,'relative_RMS':float(np.sqrt(np.sum(d*d)/max(float(np.sum(a*a)),1e-30))),'A_SHA256':hashlib.sha256(ra).hexdigest(),'C_SHA256':hashlib.sha256(rb).hexdigest()}
 report['boundary_comparisons']=[];report['trace_finite']=True
 for device in range(8):
  def records(arm,call):
   p=root/arm/f'stage{node}-rank{device}-layer0-kda-call{call}.jsonl';v=[json.loads(x)for x in p.read_text().splitlines()];assert all(x['nonfinite']==0 for x in v);return v
  # At536: A call25 is W1, C call25 is W8. Both precede prior same24W1prefix.
  aa=[records('A',25),records('A',26)];cc=records('C',25);cm={z['point']:z for z in cc}
  for j,pos in enumerate([536,537]):
   for rec in aa[j]:
    point=rec['point'];other=cm.get(point)
    if other is None:continue
    shape=rec['shape'];osh=other['shape'];dtype=rec['dtype'];item=2 if dtype=='bf16'else 8 if dtype=='i64'else 4
    is_token=len(shape)>=2 and shape[0]==1 and shape[1]==1 and len(osh)==len(shape)and osh[0]==1 and osh[1]==8 and shape[2:]==osh[2:]
    is_headnorm=point=='normalized_f32'and shape==[6,128]and osh==[48,128]
    if not(is_token or is_headnorm):continue
    rawA=(root/'A'/rec['file']).read_bytes();rawC=(root/'C'/other['file']).read_bytes();rowbytes=len(rawA);rawC=rawC[j*rowbytes:(j+1)*rowbytes];report['boundary_comparisons'].append({'rank':node*8+device,'position':pos,'point':point,'dtype':dtype,**compare(rawA,rawC,dtype)})
  # Compare actual active init pool1 before536, not stale shadow9.
  beforeA={z['point']:z for z in aa[0]};beforeC=cm
  for point in ['conv_before_row_1','ssm_before_row_1']:
   a,bb=beforeA[point],beforeC[point];report['boundary_comparisons'].append({'rank':node*8+device,'position':536,'point':point,'dtype':a['dtype'],**compare((root/'A'/a['file']).read_bytes(),(root/'C'/bb['file']).read_bytes(),a['dtype'])})
 report['recorded678_output_diagnostic']=[]
 if node==0:
  for record in m['recorded_outputs']:
   ref=np.fromfile(record['path'],dtype='<u2').reshape(-1,7168)[record['row']];actual=shared_outputs[(0,record['arm'])][record['position']];report['recorded678_output_diagnostic'].append({'arm':record['arm'],'position':record['position'],'reference_SHA256':record['SHA256'],**compare(ref.tobytes(),actual.tobytes(),'bf16')})
 maps=pathlib.Path('/proc/self/maps').read_text();libs=sorted({x.split()[-1]for x in maps.splitlines()if any(n in x for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so','libinfiniccl.so'])});report['actual_runtime_dependencies_SHA256']={p:sha(p)for p in libs};report.update(status='LAYER0_REPLAY_COMPONENT_DIAGNOSTIC_COMPLETE',normal_close=True,finite=True,canary=True,host_input_readonly=True,native_calls_thisnode=sum(z[a]['native_calls']for z in report['workers']for a in ['A','C']),Model_finalIDs='NOT_TESTED',samegeometry_repeat='NOT_RUN; oneA/Cmatrix only');assert report['native_calls_thisnode']==472;save();control.collect('normalclose_complete',{'node':node,'normal_close':True});print(json.dumps({'node':node,'status':report['status'],'receipt':str(path),'native_calls':472}))
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--cpu-contract':m,q=frontend(sys.argv[2]);print(json.dumps(q))
 elif len(sys.argv)==2:
  try:main(sys.argv[1])
  except BaseException as e:
   p=pathlib.Path('/refactor/results')/('Kimi-layer0-real-r9038-'+os.environ.get('SLURM_JOB_ID','none')+'.'+os.environ.get('SLURM_STEP_ID','none')+'-node'+os.environ.get('SLURM_PROCID','none')+'-fatal.json');p.write_text(json.dumps({'exception':repr(e),'traceback':traceback.format_exc()},indent=2));raise
 else:raise SystemExit('usage: driver [--cpu-contract] manifest')
