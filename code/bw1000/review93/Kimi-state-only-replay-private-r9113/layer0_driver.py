"""Real TP16 Delta prefix-state replay component; no full-Model TPS claim."""
import pathlib,json,hashlib,os,ctypes,threading,time,copy,concurrent.futures
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main(manifest):
 import numpy as np,torch,infinilm
 from stage_control_r2386 import StageControl
 m=json.loads(pathlib.Path(manifest).read_bytes());r=pathlib.Path('/refactor');node=int(os.environ['SLURM_PROCID']);job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('Kimi-state-only-r9113-'+job+'-node'+str(node)+'.json');assert not path.exists();assert m['TP']==16 and m['nodes']==['a02r3n03','a02r3n07'];assert sha(m['native'])==m['native_SHA256'] and sha(m['Main'])==m['Main_SHA256'];assert sha(m['norm']['path'])==m['norm']['SHA256'];assert m['norm']['shape']==[544,7168]
 torch.set_num_threads(1);report={'status':'RUNNING','job':job,'node':node,'scope':'ActualTP16singleDelta layer0 real544norm, cachedConv/KDA partialstate replay. Component only; no Model benchmark/token gate.','workers':[{'rank':node*8+i,'normal_close':False}for i in range(8)],'reference_token_equality_required':False,'GPU_F64':False,'raw_tensors':'NOT_SAVED_LOCAL_OR_GIT'};lock=threading.Lock();local=threading.local()
 def save():
  snap=copy.deepcopy(local.own)if hasattr(local,'own')else None
  with lock:
   if snap is not None:report['workers'][local.device]=snap
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 save();ctl=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],2,500);ctl.collect('stateonly_frontend',{'node':node,'native_SHA256':m['native_SHA256']});norm=np.fromfile(m['norm']['path'],dtype='<u2').reshape(544,7168);cfg=pathlib.Path(m['config']).read_text();lib=ctypes.CDLL(m['native']);lib.layer0_error_r9038.restype=ctypes.c_char_p
 lib.layer0_id_size_r9038.restype=ctypes.c_size_t;lib.layer0_id_r9038.argtypes=[ctypes.c_void_p];lib.layer0_create_r9038.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.layer0_create_r9038.restype=ctypes.c_void_p;lib.layer0_load_local_r9038.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
 for n in ['finish','reset','close']:getattr(lib,'layer0_'+n+'_r9038').argtypes=[ctypes.c_void_p]
 lib.layer0_run_r9038.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.layer0_states_r9113.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];lib.layer0_epoch_r9113.argtypes=[ctypes.c_void_p];lib.layer0_epoch_r9113.restype=ctypes.c_uint64;lib.layer0_replay_epoch_r9113.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t];lib.layer0_stats_r9113.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
 def check(code):
  if code:raise RuntimeError(lib.layer0_error_r9038().decode())
 uid=(ctypes.c_ubyte*lib.layer0_id_size_r9038())()
 if node==0:check(lib.layer0_id_r9038(uid))
 got=ctl.collect('stateonly_UID',{'id':bytes(uid).hex()if node==0 else None});ctypes.memmove(uid,bytes.fromhex(got[0]['id']),len(uid));barrier=threading.Barrier(8,timeout=480)
 def worker(device):
  rank=node*8+device;own={'rank':rank,'normal_close':False,'timings':[],'checks':[],'layer_calls':0};local.device=device;local.own=own;ctx=None
  try:
   torch.cuda.set_device(device);ctx=lib.layer0_create_r9038(cfg.encode(),rank,device,uid);assert ctx,lib.layer0_error_r9038().decode()
   for key,meta in sorted(m['weights'][str(rank)].items()):
    assert sha(meta['path'])==meta['SHA256'];x=np.fromfile(meta['path'],dtype='<u2'if meta['dtype']=='BF16'else'<f4').reshape(meta['shape']);shape=(ctypes.c_size_t*x.ndim)(*x.shape);check(lib.layer0_load_local_r9038(ctx,key.encode(),x.ctypes.data,x.ndim,shape,1 if meta['dtype']=='BF16'else 0))
   check(lib.layer0_finish_r9038(ctx));save();barrier.wait()
   if device==0:ctl.collect('stateonly_allweights',{'node':node,'owners':8})
   barrier.wait()
   def run(past,width,read=1,write=1):
    x=np.ascontiguousarray(norm[past:past+width]);before=hashlib.sha256(x.tobytes()).hexdigest();guard=np.full(width*7168+64,0x7fc1,np.uint16);out=guard[32:-32];tick=time.perf_counter();check(lib.layer0_run_r9038(ctx,width,read,write,x.ctypes.data,out.ctypes.data));elapsed=time.perf_counter()-tick;own['layer_calls']+=1;assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((out&0x7f80)==0x7f80);assert hashlib.sha256(x.tobytes()).hexdigest()==before;return elapsed
   def states():
    conv=np.empty((17,2304,3),np.uint16);ssm=np.empty((17,6,128,128),np.uint16);check(lib.layer0_states_r9113(ctx,conv.ctypes.data,ssm.ctypes.data));assert not np.any((conv&0x7f80)==0x7f80)and not np.any((ssm&0x7f80)==0x7f80);return conv,ssm
   def prep():
    barrier.wait()
    if device==0:os.environ['KIMI_STATE_CAPTURE_R9113']='0'
    barrier.wait();check(lib.layer0_reset_r9038(ctx));run(0,512)
    for p in range(512,536):run(p,1)
    barrier.wait()
   def set_capture(on):
    barrier.wait()
    if device==0:os.environ['KIMI_STATE_CAPTURE_R9113']=str(int(on))
    barrier.wait()
   def full(n):
    prep();active_before=states();set_capture(False);verify=run(536,8,1,9);barrier.wait();prefix=run(536,n,1,9);snap=states();assert np.array_equal(snap[0][1],active_before[0][1])and np.array_equal(snap[1][1],active_before[1][1]);return verify,prefix,snap
   def replay(n,negative=False):
    prep();active_before=states();set_capture(True);verify=run(536,8,1,9);epoch=int(lib.layer0_epoch_r9113(ctx));assert epoch>0;set_capture(False)
    if negative:
     for bad_epoch,bad_n in [(epoch-1,n),(epoch,0),(epoch,9)]:assert lib.layer0_replay_epoch_r9113(ctx,bad_epoch,bad_n)!=0
    barrier.wait();tick=time.perf_counter();check(lib.layer0_replay_epoch_r9113(ctx,epoch,n));elapsed=time.perf_counter()-tick;out=states();assert lib.layer0_replay_epoch_r9113(ctx,epoch,n)!=0,'already-consumed replay accepted';assert np.array_equal(out[0][1],active_before[0][1])and np.array_equal(out[1][1],active_before[1][1]);others=[i for i in range(17)if i!=9];assert np.array_equal(out[0][others],active_before[0][others])and np.array_equal(out[1][others],active_before[1][others]);return verify,elapsed,out
   def difference(a,b):
    x=(a.astype(np.uint32)<<16).view(np.float32);y=(b.astype(np.uint32)<<16).view(np.float32);diff=x-y;return {'elements':x.size,'different':int(np.count_nonzero(a!=b)),'maxabs':float(np.max(np.abs(diff))),'relative_RMS':float(np.sqrt(np.mean(diff*diff))/(np.sqrt(np.mean(x*x))+1e-20))}
   for n in [1,4,8]:
    if device==0:ctl.collect('stateonly_QA:'+str(n),{'node':node})
    a=full(n);b=replay(n,negative=True);row={'retained':n,'conv':difference(a[2][0][9],b[2][0][9]),'ssm':difference(a[2][1][9],b[2][1][9]),'active_and_other_rows_unchanged':True,'stale_epoch_bad_width_duplicate_rejected':True,'full_accept_statecheck_only':n==8};own['checks'].append(row);save()
   for n in [1,4]:
    if device==0:ctl.collect('stateonly_timing:'+str(n),{'node':node})
    for pair in range(5):
     values={}
     for mode in (['full','state']if pair%2==0 else['state','full']):
      x=full(n)if mode=='full'else replay(n);values[mode]={'verify_seconds':x[0],'replay_seconds':x[1],'verify_plus_replay_seconds':x[0]+x[1]}
     own['timings'].append({'retained':n,'pair':pair,'warmup':pair==0,**values});save()
   stat=(ctypes.c_uint64*7)();check(lib.layer0_stats_r9113(ctx,stat,7));own['actual_cache_counters']=list(stat);assert stat[1]>0 and stat[2]>0 and stat[3]==stat[4]==stat[2];own['finite_readonly_canary']=True;save();barrier.wait()
  except BaseException as e:
   own['error']=repr(e);save()
   try:barrier.abort()
   except Exception:pass
   raise
  finally:
   if ctx is not None:
    code=lib.layer0_close_r9038(ctx);own['normal_close']=code==0
   save()
  return own
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:results=list(pool.map(worker,range(8)))
 assert all(v['normal_close']and'error'not in v for v in results);report.update(status='STATE_ONLY_COMPONENT_COMPLETE',completed=True,normal_close=True,owners=8,Model_gain=False,raw_tensors='NO_RAW_TENSOR_SAVED',native_SHA256=m['native_SHA256']);save();ctl.collect('stateonly_normal',{'node':node,'normal':True});print('KIMI_STATE_ONLY_R9113_NORMAL',flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
