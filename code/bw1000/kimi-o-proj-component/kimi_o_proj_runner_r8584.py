"""Finite faithful TP8 actual RowParallel component; no Model or service TPS."""
import pathlib,json,ctypes,hashlib,threading,time,concurrent.futures,os
import numpy as np

def resolve(value):
 p=pathlib.PurePosixPath(value);host=pathlib.PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
 try:return pathlib.Path('/refactor')/p.relative_to(host)
 except ValueError:return pathlib.Path(value)
sha=lambda b:hashlib.sha256(b).hexdigest()
def compare(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype==np.dtype('<u2')
 af=(a.astype(np.uint32)<<16).view(np.float32);bf=(b.astype(np.uint32)<<16).view(np.float32);d=af.astype(np.float64)-bf.astype(np.float64);den=float(np.mean(bf.astype(np.float64)**2))
 return dict(changed=int(np.count_nonzero(a!=b)),elements=a.size,max_abs=float(np.max(np.abs(d))),nrmse=float(np.sqrt(np.mean(d*d)/den))if den else float(np.sqrt(np.mean(d*d))),finite=bool(np.isfinite(af).all()),SHA256=sha(a.tobytes()),expected_SHA256=sha(b.tobytes()))

def run(manifest,library,output):
 path=pathlib.Path(output);assert not path.exists()
 import torch,infinilm
 torch.set_num_threads(1);assert torch.cuda.device_count()==8
 m=json.loads(pathlib.Path(manifest).read_text());assert len(m['ranks'])==8
 closure=json.loads(pathlib.Path('/refactor/results/Kimi-o-proj-component-CPU-closure-r8586.json').read_text());assert sha(pathlib.Path(library).read_bytes())==closure['library_SHA256']=='cd02455424fba7fe9965769c4c7f9c8084d0d4f7e858061daa10b91c7bb26eb9'
 lib=ctypes.CDLL(str(library));lib.kimi_oproj_error_r8582.restype=ctypes.c_char_p
 lib.kimi_oproj_id_bytes_r8582.restype=ctypes.c_size_t;lib.kimi_oproj_id_r8582.argtypes=[ctypes.c_void_p]
 lib.kimi_oproj_create_r8582.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.kimi_oproj_create_r8582.restype=ctypes.c_void_p
 lib.kimi_oproj_run_r8582.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];lib.kimi_oproj_destroy_r8582.argtypes=[ctypes.c_void_p]
 class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
 expected=pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so');providers={}
 symbols=closure['critical_mangled_symbols'];assert len(symbols)>=4
 for name in symbols:
  d=Dl();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(d));actual=pathlib.Path(d.fname.decode()).resolve();assert actual==expected.resolve();assert sha(actual.read_bytes())=='2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2';providers[name]=dict(path=str(actual),SHA256=sha(actual.read_bytes()))
 data=[];cache={}
 def load(record):
  shape=record.get('shape',record.get('record',{}).get('shape'));key=(record['SHA256'],tuple(shape))
  if key not in cache:
   raw=resolve(record['path']).read_bytes();assert sha(raw)==record['SHA256'];cache[key]=np.frombuffer(raw,dtype='<u2').reshape(shape)
  return cache[key]
 for rank in m['ranks']:
  cases={c['label']:dict(x=load(c['input']),expected=load(c['expected']))for c in rank['cases']}
  data.append(dict(weight=load(rank['weight']),cases=cases))
 result=dict(scope='TP8 layer0 faithful RowParallel component; NOT Model/finaltokens/TPS',source_manifest_SHA256=sha(pathlib.Path(manifest).read_bytes()),library_SHA256=sha(pathlib.Path(library).read_bytes()),actualproviders=providers,memory_gate_scope='Host actualancestor/MemAvailable checked in the same fresh before-KFD admission step, before this numericPopen',workers=[dict(rank=i,normal_close=False,baseline=[],diagnostic=[])for i in range(8)],all8normal=False,final_tokens='NOT_MEASURED',performance='N/A_CAUSAL_DIAGNOSTIC',raw_storage='REMOTE_ONLY')
 lock=threading.Lock();validation_lock=threading.Lock();barrier=threading.Barrier(8,timeout=120);ready=[False]*8;different=[None]*8;choice={'mode':None};started=time.monotonic()
 def inspect(a,b):
  # Limit transient CPU float comparison storage to one full2048x7168 pair.
  with validation_lock:return compare(a,b)
 def save():
  with lock:
   temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
 def check(code):
  if code:raise RuntimeError(lib.kimi_oproj_error_r8582().decode())
 uid=(ctypes.c_ubyte*lib.kimi_oproj_id_bytes_r8582())();check(lib.kimi_oproj_id_r8582(uid));save()
 def worker(rank):
  torch.cuda.set_device(rank);ctx=None;own=result['workers'][rank]
  try:
   item=data[rank];ctx=lib.kimi_oproj_create_r8582(rank,rank,uid,item['weight'].ctypes.data);assert ctx,lib.kimi_oproj_error_r8582();barrier.wait()
   local={};base={};diag={}
   def call(label,mode):
    case=item['cases'][label];x=case['x'];y=np.empty_like(case['expected']);pre=np.empty_like(y);span=ctypes.c_double();begin=time.monotonic();check(lib.kimi_oproj_run_r8582(ctx,x.shape[1],x.ctypes.data,pre.ctypes.data,y.ctypes.data,mode,ctypes.byref(span)))
    return y,pre,dict(full_CAPI_wall_s=time.monotonic()-begin,native_diagnostic_span_s=span.value,contains_preTP_D2H=mode!=0)
   for label in ('detail-O','detail-N'):
    y,pre,t=call(label,0);base[label]=y;own['baseline'].append(dict(label=label,**inspect(y,item['cases'][label]['expected']),**t));save()
   ready[rank]=all(v['changed']==0 and v['finite']for v in own['baseline']);barrier.wait()
   if not all(ready):own['candidate_skipped']='Any original fullO/N RowParallel output notfaithful';save();return
   for label in ('detail-O','detail-N'):
    y,pre,t=call(label,1);local[label]=pre;diag[label]=y;own['diagnostic'].append(dict(label=label,against_actual_Row=inspect(y,base[label]),**t));save()
   ready[rank]=all(v['against_actual_Row']['changed']==0 and v['against_actual_Row']['finite']for v in own['diagnostic']);different[rank]=inspect(local['detail-N'][:,:512],local['detail-O']);own['preTP_target_N_vs_O']=different[rank];barrier.wait()
   if rank==0 and all(ready):choice['mode']=2 if any(v['changed']for v in different)else 3;result['singlefactor_choice']=dict(mode=choice['mode'],meaning='Split localGEMM512, original fullM CCL'if choice['mode']==2 else'Original fullM localGEMM, splitCCL512',not_claimed='Other factor may also contribute; component not finalModel acceptance')
   barrier.wait()
   if not all(ready):own['candidate_skipped']='Instrumented local+originalCCL notfaithful toactualRow';save();return
   y,pre,t=call('detail-N',choice['mode']);own['singlefactor']=dict(mode=choice['mode'],target_vs_captured_O=inspect(y[:,:512],item['cases']['detail-O']['expected']),full_vs_captured_N=inspect(y,item['cases']['detail-N']['expected']),preTP_target_vs_O=inspect(pre[:,:512],local['detail-O']),**t);save();barrier.wait()
  except BaseException as error:
   own['error']=repr(error);barrier.abort();save()
  finally:
   if ctx:
    own['normal_close']=lib.kimi_oproj_destroy_r8582(ctx)==0
    if not own['normal_close']:own['close_error']=lib.kimi_oproj_error_r8582().decode()
   save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(worker,range(8)))
 result.update(elapsed_s=time.monotonic()-started,all8normal=all(v['normal_close']for v in result['workers']),all_original_full_O_N_faithful=all(len(v['baseline'])==2 and all(x['changed']==0 for x in v['baseline'])for v in result['workers']),all_diagnostic_full_O_N_faithful=all(len(v['diagnostic'])==2 and all(x['against_actual_Row']['changed']==0 for x in v['diagnostic'])for v in result['workers']))
 save();print(json.dumps({k:v for k,v in result.items()if k!='workers'}))
 if not result['all8normal']or any('error'in v for v in result['workers']):raise SystemExit(1)
if __name__=='__main__':
 import sys
 run(*sys.argv[1:])
