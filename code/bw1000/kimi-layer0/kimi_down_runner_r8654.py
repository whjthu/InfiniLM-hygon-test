"""Finite faithful TP8 actual RowParallel component; no Model or service TPS."""
import pathlib,json,ctypes,hashlib,threading,time,concurrent.futures,os
import numpy as np
from kimi_owner_report_r8643 import OwnerSnapshotReport

def resolve(value):
 p=pathlib.PurePosixPath(value);host=pathlib.PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
 try:return pathlib.Path('/refactor')/p.relative_to(host)
 except ValueError:return pathlib.Path(value)
sha=lambda b:hashlib.sha256(b).hexdigest()
def compare(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype==np.dtype('<u2')
 af=(a.astype(np.uint32)<<16).view(np.float32);bf=(b.astype(np.uint32)<<16).view(np.float32);d=af.astype(np.float64)-bf.astype(np.float64);den=float(np.mean(bf.astype(np.float64)**2))
 return dict(changed=int(np.count_nonzero(a!=b)),elements=a.size,max_abs=float(np.max(np.abs(d))),nrmse=float(np.sqrt(np.mean(d*d)/den))if den else float(np.sqrt(np.mean(d*d))),finite=bool(np.isfinite(af).all()),SHA256=sha(a.tobytes()),expected_SHA256=sha(b.tobytes()))

def validate_manifest(m):
 assert m['source_handle']=='910374.757'and len(m['ranks'])==8 and [v['rank']for v in m['ranks']]==list(range(8))
 w=m['checkpoint_weight'];assert w['shape']==[7168,33792]and w['dtype']=='BF16'and w['bytes']==484442112 and w['checkpoint_key']=='language_model.model.layers.0.mlp.down_proj.weight'and w['native_FQN']=='layers.0.mlp.down_proj.weight'and w['Row_leaf']=='weight'
 for rank in m['ranks']:
  assert [v['label']for v in rank['cases']]==['detail-O','detail-N']
  for case,M in zip(rank['cases'],[512,2048]):
   for key,width in [('input',4224),('expected',7168)]:
    rec=case[key];assert rec['dtype']=='BF16'and rec['shape']==[1,M,width]and rec['strides']==[M*width,width,1]and rec['bytes']==2*M*width and len(rec['SHA256'])==64
    assert pathlib.PurePosixPath(rec['path']).is_relative_to('/refactor/results/Kimi-MLP-layer0-r8648-910374.757-raw')
 return {'ranks':8,'realcompanions':4,'global_weight_bytes':484442112,'new_GPU':False}

def run(manifest,library,output):
 path=pathlib.Path(output);assert not path.exists()
 import torch,infinilm
 torch.set_num_threads(1);assert torch.cuda.device_count()==8
 m=json.loads(pathlib.Path(manifest).read_text());validate_manifest(m)
 closure_path=pathlib.Path('/refactor')/m['CPU_receipt'];assert sha(closure_path.read_bytes())==m['CPU_receipt_raw_SHA256'];closure=json.loads(closure_path.read_text());assert closure['CPU_only']and not closure['GPU_dlopen'];assert sha(pathlib.Path(library).read_bytes())==closure['library_SHA256']==m['library_SHA256']
 lib=ctypes.CDLL(str(library));lib.kimi_down_error_r8653.restype=ctypes.c_char_p
 lib.kimi_down_id_bytes_r8653.restype=ctypes.c_size_t;lib.kimi_down_id_r8653.argtypes=[ctypes.c_void_p]
 lib.kimi_down_create_r8653.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.kimi_down_create_r8653.restype=ctypes.c_void_p
 lib.kimi_down_run_r8653.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];lib.kimi_down_destroy_r8653.argtypes=[ctypes.c_void_p]
 class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
 expected=pathlib.Path('/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/_infinilm.cpython-310-x86_64-linux-gnu.so');providers={}
 symbols=m['critical_mangled_symbols'];assert len(symbols)>=4
 for name in symbols:
  d=Dl();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(d));actual=pathlib.Path(d.fname.decode()).resolve();assert actual==(pathlib.Path('/refactor/candidates/Kimi-CCL512-r8606/libKimi_CCL512_r8606.so').resolve()if 'create_kimi_k3_model_config'in symbols[name]else expected.resolve());assert sha(actual.read_bytes())==('f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'if 'create_kimi_k3_model_config'in symbols[name]else'2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2');providers[name]=dict(path=str(actual),SHA256=sha(actual.read_bytes()))
 from safetensors import safe_open
 model=pathlib.Path('/shared/Kimi-K3');meta=m['checkpoint_weight'];index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert index[meta['checkpoint_key']]==meta['shard'];weightpath=model/meta['shard']
 import struct
 with weightpath.open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))[meta['checkpoint_key']]
 assert all(header[k]==meta[k]for k in ['shape','dtype','data_offsets'])
 with safe_open(weightpath,framework='pt',device='cpu')as f:globalweight=f.get_tensor(meta['checkpoint_key']).contiguous()
 assert list(globalweight.shape)==[7168,33792]and globalweight.dtype==torch.bfloat16 and sha(globalweight.view(torch.uint8).numpy().tobytes())==meta['loaded_SHA256']
 config=(model/'config.json').read_text()
 data=[];cache={}
 def load(record):
  shape=record.get('shape',record.get('record',{}).get('shape'));key=(record['SHA256'],tuple(shape))
  if key not in cache:
   raw=resolve(record['path']).read_bytes();assert sha(raw)==record['SHA256']and len(raw)==record['bytes']==int(np.prod(shape))*2;cache[key]=np.frombuffer(raw,dtype='<u2').reshape(shape)
  return cache[key]
 for rank in m['ranks']:
  cases={c['label']:dict(x=load(c['input']),expected=load(c['expected']))for c in rank['cases']}
  data.append(dict(weight=globalweight,cases=cases))
 result=dict(scope='TP8 layer0 faithful RowParallel component; NOT Model/finaltokens/TPS',source_manifest_SHA256=sha(pathlib.Path(manifest).read_bytes()),library_SHA256=sha(pathlib.Path(library).read_bytes()),actualproviders=providers,memory_gate_scope='Host actualancestor/MemAvailable checked in the same fresh before-KFD admission step, before this numericPopen',workers=[dict(rank=i,normal_close=False,baseline=[],diagnostic=[])for i in range(8)],all8normal=False,final_tokens='NOT_MEASURED',performance='N/A_CAUSAL_DIAGNOSTIC',raw_storage='REMOTE_ONLY')
 reporter=OwnerSnapshotReport(path,result);validation_lock=threading.Lock();barrier=threading.Barrier(8,timeout=120);ready=[False]*8;different=[None]*8;choice={'mode':None};started=time.monotonic()
 def inspect(a,b):
  # Limit transient CPU float comparison storage to one full2048x7168 pair.
  with validation_lock:return compare(a,b)
 def save():reporter.flush()
 def check(code):
  if code:raise RuntimeError(lib.kimi_down_error_r8653().decode())
 uid=(ctypes.c_ubyte*lib.kimi_down_id_bytes_r8653())();check(lib.kimi_down_id_r8653(uid));save()
 def worker(rank):
  torch.cuda.set_device(rank);ctx=None;own=dict(rank=rank,normal_close=False,baseline=[],diagnostic=[])
  def save():reporter.publish_worker(rank,own)
  try:
   item=data[rank];ctx=lib.kimi_down_create_r8653(config.encode(),rank,rank,uid,item['weight'].data_ptr());assert ctx,lib.kimi_down_error_r8653();barrier.wait()
   local={};base={};diag={}
   def call(label,mode):
    case=item['cases'][label];x=case['x'];y=np.empty_like(case['expected']);pre=np.empty_like(y);span=ctypes.c_double();begin=time.monotonic();check(lib.kimi_down_run_r8653(ctx,x.shape[1],x.ctypes.data,pre.ctypes.data,y.ctypes.data,mode,ctypes.byref(span)))
    return y,pre,dict(full_CAPI_wall_s=time.monotonic()-begin,native_diagnostic_span_s=span.value,contains_preTP_D2H=mode!=0)
   for label in ('detail-O','detail-N'):
    y,pre,t=call(label,0);base[label]=y;own['baseline'].append(dict(label=label,**inspect(y,item['cases'][label]['expected']),**t));save()
   ready[rank]=all(v['changed']==0 and v['finite']for v in own['baseline']);barrier.wait()
   if not all(ready):own['candidate_skipped']='Any original fullO/N RowParallel output notfaithful';save();return
   for label in ('detail-O','detail-N'):
    y,pre,t=call(label,1);local[label]=pre;diag[label]=y;own['diagnostic'].append(dict(label=label,against_actual_Row=inspect(y,base[label]),**t));save()
   ready[rank]=all(v['against_actual_Row']['changed']==0 and v['against_actual_Row']['finite']for v in own['diagnostic']);different[rank]=inspect(local['detail-N'][:,:512],local['detail-O']);own['preTP_target_N_vs_O']=different[rank];barrier.wait()
   if rank==0 and all(ready):choice['mode']=2 if any(v['changed']for v in different)else 3;reporter.publish_top(singlefactor_choice=dict(mode=choice['mode'],meaning='Split localGEMM512, original fullM CCL'if choice['mode']==2 else'Original fullM localGEMM, splitCCL512',not_claimed='Other factor may also contribute; component not finalModel acceptance'))
   barrier.wait()
   if not all(ready):own['candidate_skipped']='Instrumented local+originalCCL notfaithful toactualRow';save();return
   y,pre,t=call('detail-N',choice['mode']);own['singlefactor']=dict(mode=choice['mode'],target_vs_captured_O=inspect(y[:,:512],item['cases']['detail-O']['expected']),full_vs_captured_N=inspect(y,item['cases']['detail-N']['expected']),preTP_target_vs_O=inspect(pre[:,:512],local['detail-O']),**t);save();barrier.wait()
  except BaseException as error:
   own['error']=repr(error);barrier.abort();save()
  finally:
   if ctx:
    own['normal_close']=lib.kimi_down_destroy_r8653(ctx)==0
    if not own['normal_close']:own['close_error']=lib.kimi_down_error_r8653().decode()
   save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(worker,range(8)))
 result=reporter.snapshot();reporter.publish_top(elapsed_s=time.monotonic()-started,all8normal=all(v['normal_close']for v in result['workers']),all_original_full_O_N_faithful=all(len(v['baseline'])==2 and all(x['changed']==0 for x in v['baseline'])for v in result['workers']),all_diagnostic_full_O_N_faithful=all(len(v['diagnostic'])==2 and all(x['against_actual_Row']['changed']==0 for x in v['diagnostic'])for v in result['workers']))
 result=reporter.snapshot();print(json.dumps({k:v for k,v in result.items()if k!='workers'}))
 if not result['all8normal']or any('error'in v for v in result['workers']):raise SystemExit(1)
if __name__=='__main__':
 import sys
 run(*sys.argv[1:])
