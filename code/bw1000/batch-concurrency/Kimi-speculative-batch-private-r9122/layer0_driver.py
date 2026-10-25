"""Real TP16 Delta prefix-state replay component; no full-Model TPS claim."""
import pathlib,json,hashlib,os,ctypes,threading,time,copy,concurrent.futures
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main(manifest):
 import numpy as np,torch,infinilm
 from stage_control_r2386 import StageControl
 m=json.loads(pathlib.Path(manifest).read_bytes());r=pathlib.Path('/refactor');node=int(os.environ['SLURM_PROCID']);job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];path=r/'results'/('Kimi-batch-r9122-'+job+'-node'+str(node)+'.json');assert not path.exists();assert m['TP']==16 and m['nodes']==['a02r3n03','a02r3n07'];assert sha(m['native'])==m['native_SHA256'] and sha(m['Main'])==m['Main_SHA256'];assert sha(m['norm']['path'])==m['norm']['SHA256'];assert m['norm']['shape']==[544,7168]
 torch.set_num_threads(1);report={'status':'RUNNING','job':job,'node':node,'scope':'ActualTP16singleDelta layer0 realweights544norm; native packedC2W4/C4W2 verify, independent statepairs/commit/EOS and nextforward. Duplicated prefill trajectory, component only, no Model quality/TPS claim.','workers':[{'rank':node*8+i,'normal_close':False}for i in range(8)],'reference_token_equality_required':False,'GPU_F64':False,'raw_tensors':'NOT_SAVED_LOCAL_OR_GIT'};lock=threading.Lock();local=threading.local()
 def save():
  snap=copy.deepcopy(local.own)if hasattr(local,'own')else None
  with lock:
   if snap is not None:report['workers'][local.device]=snap
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 save();ctl=StageControl(node,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],2,500);ctl.collect('batch_frontend',{'node':node,'native_SHA256':m['native_SHA256']});norm=np.fromfile(m['norm']['path'],dtype='<u2').reshape(544,7168);cfg=pathlib.Path(m['config']).read_text();lib=ctypes.CDLL(m['native']);lib.layer0_error_r9038.restype=ctypes.c_char_p
 lib.layer0_id_size_r9038.restype=ctypes.c_size_t;lib.layer0_id_r9038.argtypes=[ctypes.c_void_p];lib.layer0_create_r9038.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.layer0_create_r9038.restype=ctypes.c_void_p;lib.layer0_load_local_r9038.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
 for n in ['finish','reset','close']:getattr(lib,'layer0_'+n+'_r9038').argtypes=[ctypes.c_void_p]
 lib.layer0_batch_run_r9122.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]+[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.layer0_states_r9113.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];lib.layer0_epoch_r9113.argtypes=[ctypes.c_void_p];lib.layer0_epoch_r9113.restype=ctypes.c_uint64
 lib.layer0_batch_prepare_r9122.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t];lib.layer0_batch_commit_r9122.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p];lib.layer0_batch_replay_r9122.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.c_void_p];lib.layer0_batch_row_r9122.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.layer0_batch_stats_r9122.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
 def check(code):
  if code:raise RuntimeError(lib.layer0_error_r9038().decode())
 uid=(ctypes.c_ubyte*lib.layer0_id_size_r9038())()
 if node==0:check(lib.layer0_id_r9038(uid))
 got=ctl.collect('batch_UID',{'id':bytes(uid).hex()if node==0 else None});ctypes.memmove(uid,bytes.fromhex(got[0]['id']),len(uid));barrier=threading.Barrier(8,timeout=480)
 def worker(device):
  rank=node*8+device;own={'rank':rank,'normal_close':False,'timings':[],'checks':[],'layer_calls':0};local.device=device;local.own=own;ctx=None
  try:
   torch.cuda.set_device(device);ctx=lib.layer0_create_r9038(cfg.encode(),rank,device,uid);assert ctx,lib.layer0_error_r9038().decode()
   for key,meta in sorted(m['weights'][str(rank)].items()):
    assert sha(meta['path'])==meta['SHA256'];x=np.fromfile(meta['path'],dtype='<u2'if meta['dtype']=='BF16'else'<f4').reshape(meta['shape']);shape=(ctypes.c_size_t*x.ndim)(*x.shape);check(lib.layer0_load_local_r9038(ctx,key.encode(),x.ctypes.data,x.ndim,shape,1 if meta['dtype']=='BF16'else 0))
   check(lib.layer0_finish_r9038(ctx));save();barrier.wait()
   if device==0:ctl.collect('batch_allweights',{'node':node,'owners':8})
   barrier.wait()
   def run(parts,logical,past,read,write,capture=False):
    x=np.ascontiguousarray(np.concatenate(parts,axis=0));widths=[len(y)for y in parts];arrays=[np.asarray(z,np.int32)for z in [logical,past,np.cumsum([0]+widths),read,write]];before=hashlib.sha256(x.tobytes()).hexdigest();guard=np.full(len(x)*7168+64,0x7fc1,np.uint16);out=guard[32:-32];tick=time.perf_counter();check(lib.layer0_batch_run_r9122(ctx,len(x),len(parts),*[z.ctypes.data for z in arrays],int(capture),x.ctypes.data,out.ctypes.data));elapsed=time.perf_counter()-tick;own['layer_calls']+=1
    assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((out&0x7f80)==0x7f80);assert hashlib.sha256(x.tobytes()).hexdigest()==before;return elapsed
   def states():
    conv=np.empty((17,2304,3),np.uint16);ssm=np.empty((17,6,128,128),np.uint16);check(lib.layer0_states_r9113(ctx,conv.ctypes.data,ssm.ctypes.data));assert not np.any((conv&0x7f80)==0x7f80)and not np.any((ssm&0x7f80)==0x7f80);return conv,ssm
   def row(logical):
    out=np.full(4,-1,np.int32);check(lib.layer0_batch_row_r9122(ctx,logical,out.ctypes.data));return out.tolist()
   def prep(C):
    barrier.wait();check(lib.layer0_reset_r9038(ctx));run([norm[:512]]*C,list(range(1,C+1)),[0]*C,list(range(1,C+1)),list(range(1,C+1)));check(lib.layer0_batch_prepare_r9122(ctx,C,512));barrier.wait()
   def baseline(C,W,ret,cont):
    prep(C);before=states();parts=[norm[512+i*W:512+(i+1)*W]for i in range(C)];logical=list(range(1,C+1));verify=run(parts,logical,[512]*C,logical,[x+8 for x in logical]);partial=[i for i in range(C)if cont[i]and ret[i]<W];barrier.wait();replay=0
    if partial:replay=run([parts[i][:ret[i]]for i in partial],[i+1 for i in partial],[512]*len(partial),[i+1 for i in partial],[i+9 for i in partial])
    after=states();assert np.array_equal(after[0][1:C+1],before[0][1:C+1])and np.array_equal(after[1][1:C+1],before[1][1:C+1]);return verify,replay,after
   def candidate(C,W,ret,cont,negative=False,next_step=False):
    prep(C);before=states();parts=[norm[512+i*W:512+(i+1)*W]for i in range(C)];logical=list(range(1,C+1));verify=run(parts,logical,[512]*C,logical,[x+8 for x in logical],True);epoch=int(lib.layer0_epoch_r9113(ctx));accepted=np.asarray([x-1 for x in ret],np.uintp);continuation=np.asarray(cont,np.int32)
    if negative:
     bad=np.full(C,W,np.uintp);assert lib.layer0_batch_commit_r9122(ctx,epoch,C,bad.ctypes.data,continuation.ctypes.data)!=0
    tick=time.perf_counter();check(lib.layer0_batch_commit_r9122(ctx,epoch,C,accepted.ctypes.data,continuation.ctypes.data));commitwall=time.perf_counter()-tick;kept=np.asarray([ret[i]if cont[i]and ret[i]<W else W for i in range(C)],np.uintp)
    if negative:
     assert lib.layer0_batch_replay_r9122(ctx,epoch-1,C,kept.ctypes.data)!=0;bad=kept.copy();bad[0]=0;assert lib.layer0_batch_replay_r9122(ctx,epoch,C,bad.ctypes.data)!=0
    barrier.wait();tick=time.perf_counter();check(lib.layer0_batch_replay_r9122(ctx,epoch,C,kept.ctypes.data));replaywall=time.perf_counter()-tick;after=states();assert lib.layer0_batch_replay_r9122(ctx,epoch,C,kept.ctypes.data)!=0
    assert np.array_equal(after[0][1:C+1],before[0][1:C+1])and np.array_equal(after[1][1:C+1],before[1][1:C+1]);others=[i for i in range(17)if i not in range(9,9+C)];assert np.array_equal(after[0][others],before[0][others])and np.array_equal(after[1][others],before[1][others]);rows=[row(i+1)for i in range(C)]
    for i,z in enumerate(rows):assert z==([i+9,i+1,512+ret[i],1]if cont[i]else[i+1,i+9,512,4]),(i,z,ret,cont)
    stat=(ctypes.c_uint64*8)();check(lib.layer0_batch_stats_r9122(ctx,stat,8));stats=list(stat);assert stats[7]==C and stats[1]==1 and stats[2]==sum(bool(cont[i]and ret[i]<W)for i in range(C))and stats[3]==stats[4]==stats[2]
    if next_step:
     active=[i for i in range(C)if cont[i]];run([norm[520+i:521+i]for i in active],[i+1 for i in active],[rows[i][2]for i in active],[rows[i][0]for i in active],[rows[i][1]for i in active],True);epoch2=int(lib.layer0_epoch_r9113(ctx));aa=np.zeros(len(active),np.uintp);cc=np.ones(len(active),np.int32);check(lib.layer0_batch_commit_r9122(ctx,epoch2,len(active),aa.ctypes.data,cc.ctypes.data))
     for i in active:assert row(i+1)==[i+1,i+9,513+ret[i],1]
    return verify,replaywall,after,commitwall,stats,rows
   def diff(a,b):
    x=(a.astype(np.uint32)<<16).view(np.float32);y=(b.astype(np.uint32)<<16).view(np.float32);d=x-y;return {'maxabs':float(np.max(np.abs(d))),'relative_RMS':float(np.sqrt(np.mean(d*d))/(np.sqrt(np.mean(x*x))+1e-20))}
   for C in [2,4]:
    W=8//C
    for scenario in ['mixed','all','EOS']:
     ret=[W if scenario=='all'else(1 if i%2==0 else max(2,W-1))for i in range(C)];cont=[0 if scenario=='EOS'and i==0 else 1 for i in range(C)]
     if device==0:ctl.collect('batch_QA:'+str(C)+scenario,{'node':node})
     a=baseline(C,W,ret,cont);b=candidate(C,W,ret,cont,negative=True,next_step=True);own['checks'].append({'C':C,'W':W,'flattenT':C*W,'scenario':scenario,'retained':ret,'continuation':cont,'conv':diff(a[2][0][9:9+C],b[2][0][9:9+C]),'ssm':diff(a[2][1][9:9+C],b[2][1][9:9+C]),'rows_after_replay':b[5],'stats8':b[4],'active_and_other_rows_unchanged':True,'wrong_epoch_retention_double_replay_invalid_commit_rejected':True,'next_batched_W1_forward_actual_CPU_ledger_pass':True});save()
    ret=[1 if i%2==0 else max(2,W-1)for i in range(C)];cont=[1]*C
    for pair in range(5):
     values={}
     for mode in (['full','state']if pair%2==0 else['state','full']):
      x=baseline(C,W,ret,cont)if mode=='full'else candidate(C,W,ret,cont);values[mode]={'verify_seconds':x[0],'replay_seconds':x[1],'commit_seconds':x[3]if mode=='state'else 0,'verify_plus_replay_seconds':x[0]+x[1]+(x[3]if mode=='state'else 0)}
     own['timings'].append({'C':C,'W':W,'flattenT':C*W,'retained':ret,'pair':pair,'warmup':pair==0,**values});save()
   own['finite_readonly_canary']=True;own['true_packed_batch_verify']=True;save();barrier.wait()
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
 assert all(v['normal_close']and'error'not in v for v in results);report.update(status='BATCH_COMPONENT_COMPLETE',completed=True,normal_close=True,owners=8,Model_gain=False,raw_tensors='NO_RAW_TENSOR_SAVED',native_SHA256=m['native_SHA256']);save();ctl.collect('batch_normal',{'node':node,'normal':True});print('KIMI_BATCH_R9122_NORMAL',flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
