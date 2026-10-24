"""Five real captured projections × eight ranks. Finite component evidence only."""
import pathlib,json,hashlib,ctypes,time,os,subprocess,sys
import numpy as np
from kimi_projection_memory_r8559 import read_memory

def resolve(value):
 p=pathlib.PurePosixPath(value);host=pathlib.PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
 try:return pathlib.Path('/refactor')/p.relative_to(host)
 except ValueError:return pathlib.Path(value)
def sha(data):return hashlib.sha256(data).hexdigest()
def compare(actual,expected):
 assert actual.shape==expected.shape;mask=actual!=expected
 af=(actual.astype(np.uint32)<<16).view(np.float32).astype(np.float64);ef=(expected.astype(np.uint32)<<16).view(np.float32).astype(np.float64);delta=af-ef;den=float(np.mean(ef*ef))
 return dict(elements=actual.size,changed=int(mask.sum()),max_abs=float(np.max(np.abs(delta))),nrmse=float(np.sqrt(np.mean(delta*delta)/den))if den else float(np.sqrt(np.mean(delta*delta))),finite=bool(np.all((actual&0x7f80)!=0x7f80)),SHA256=sha(actual.tobytes()),expected_SHA256=sha(expected.tobytes()),firstdiff=None if not mask.any()else list(map(int,np.argwhere(mask)[0])))
def read_array(record,shape=None):
 raw=resolve(record['path']).read_bytes();assert sha(raw)==record['SHA256'];shape=shape or record['record']['shape'];array=np.frombuffer(raw,dtype='<u2').reshape(shape)
 if array.ndim==3:assert array.shape[0]==1;array=array[0]
 assert array.flags.c_contiguous;return array

def worker(manifest,library,output,rank,device):
 import torch  # Same known actual provider initialization as successful608.
 torch.set_num_threads(1)
 assert torch.cuda.device_count()==2, "Two allocated visible cards required"
 torch.cuda.set_device(device)
 m=json.loads(pathlib.Path(manifest).read_text());spec=m['ranks'][rank];assert spec['rank']==rank
 result=dict(rank=rank,device=device,scope='ACTUAL_CAPTURE_COMPONENT_ONLY; no Model/TPS',phase='CPU_operand_load',baseline=[],candidate=[],ABBA=[],normal_close=False,raw_storage='REMOTE_ONLY',source_manifest_SHA256=sha(pathlib.Path(manifest).read_bytes()),library_SHA256=sha(pathlib.Path(library).read_bytes()),timing_contract='Validation outside timers. full_CAPI includes H2D/D2H/alloc; native span includes Cast/Gemm/sync, not pure kernel profiler. No new per-op synchronization added.',resolved_torch_lib=str(pathlib.Path(torch.__file__).parent/'lib'))
 last_phase=None
 def save():
  nonlocal last_phase
  if result['phase']!=last_phase:
   result.setdefault('memory_snapshots',[]).append(read_memory(result['phase']));last_phase=result['phase']
  target=pathlib.Path(output);temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(target)
 save();items=[];contexts=[];lib=None;read_cache={}
 def load(record,shape=None):
  key=(record['path'],record['SHA256'],tuple(shape or record['record']['shape']))
  if key not in read_cache:read_cache[key]=read_array(record,shape)
  return read_cache[key]
 try:
  for op in spec['operations']:
   weight=load(op['weight'],op['weight']['shape']);cases={c['label']:dict(x=load(c['input']),y=load(c['expected']))for c in op['cases']}
   assert weight.shape==(op['n'],op['k'])and all(v['x'].shape==(512 if label=='Oconfirm'else 2048,op['k'])for label,v in cases.items())
   items.append(dict(spec=op,weight=weight,cases=cases,A=np.empty_like(cases['Wmissing']['y']),B=np.empty_like(cases['Wmissing']['y']),O=np.empty_like(cases['Oconfirm']['y'])))
  lib=ctypes.CDLL(str(library));lib.kimi_qkv_error_r8553.restype=ctypes.c_char_p
  lib.kimi_qkv_create_r8553.argtypes=[ctypes.c_int,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p];lib.kimi_qkv_create_r8553.restype=ctypes.c_void_p
  lib.kimi_qkv_run_r8553.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];lib.kimi_qkv_destroy_r8553.argtypes=[ctypes.c_void_p]
  for item in items:
   op=item['spec'];ctx=lib.kimi_qkv_create_r8553(device,op['k'],op['n'],item['weight'].ctypes.data);assert ctx,lib.kimi_qkv_error_r8553();item['ctx']=ctx;contexts.append(ctx)
  result['phase']='faithful_original';save()
  def call(item,label,chunk,destination,source=None):
   x=item['cases'][label]['x']if source is None else source;span=ctypes.c_double();begin=time.monotonic();rc=lib.kimi_qkv_run_r8553(item['ctx'],len(x),x.ctypes.data,destination.ctypes.data,chunk,ctypes.byref(span));wall=time.monotonic()-begin
   if rc:raise RuntimeError(lib.kimi_qkv_error_r8553().decode())
   return dict(full_CAPI_wall_s=wall,native_Cast_Gemm_sync_span_s=span.value,CAPI_minus_native_span_s=wall-span.value)
  for item in items:
   for label,buffer in [('Oconfirm','O'),('Wmissing','A')]:
    timing=call(item,label,0,item[buffer]);observation=compare(item[buffer],item['cases'][label]['y']);result['baseline'].append(dict(op=item['spec']['name'],label=label,**timing,**observation));save()
    if not observation['finite']:raise RuntimeError('nonfinite original '+item['spec']['name'])
  result['all_original_full_outputs_faithful']=all(v['changed']==0 for v in result['baseline']);save()
  if not result['all_original_full_outputs_faithful']:result['candidate_skipped']='Original full O/W output replay notfaithful';return result
  result['phase']='singlefactor512';save()
  for item in items:
   timing=call(item,'Wmissing',512,item['B']);target=compare(item['B'][:512],item['cases']['Oconfirm']['y']);full=compare(item['B'],item['cases']['Wmissing']['y']);result['candidate'].append(dict(op=item['spec']['name'],target_vs_original_O=target,full_vs_original_W=full,**timing));item['B_first_SHA256']=sha(item['B'].tobytes());save()
   if not target['finite']:raise RuntimeError('nonfinite candidate '+item['spec']['name'])
  byname={v['spec']['name']:v for v in items};fb=byname['f_b'];chain=np.empty_like(fb['B']);call(fb,'Wmissing',512,chain,byname['f_a']['B']);result['f_a_to_f_b_candidate_chain_target']=compare(chain[:512],fb['cases']['Oconfirm']['y']);save()
  # Both alternatives warm their real shapes. Host arrays are preallocated.
  for _ in range(2):
   for policy in ('A','B'):
    for item in items:call(item,'Wmissing',0 if policy=='A'else 512,item[policy])
  result['phase']='ABBA';save()
  for repetition in range(2):
   for policy in ('A','B','B','A'):
    timings=[];begin=time.monotonic()
    for item in items:timings.append(call(item,'Wmissing',0 if policy=='A'else 512,item[policy]))
    bundle_wall=time.monotonic()-begin
    # Hash/numeric validation deliberately starts after the timed bundle.
    validation=[]
    for item in items:
     actual=sha(item[policy].tobytes());expected=item['spec']['cases'][1]['expected']['SHA256']if policy=='A'else item['B_first_SHA256'];finite=bool(np.all((item[policy]&0x7f80)!=0x7f80));validation.append(dict(op=item['spec']['name'],finite=finite,stable_expected_SHA256=expected,actual_SHA256=actual,equal=actual==expected,reference_scope='Actual captured W'if policy=='A'else'Own firstB stability only; B target correctness separately compared to capturedO'))
    result['ABBA'].append(dict(repetition=repetition,policy=policy,bundle_full_CAPI_wall_s=bundle_wall,sum_native_span_s=sum(v['native_Cast_Gemm_sync_span_s']for v in timings),per_op=[dict(op=item['spec']['name'],**v)for item,v in zip(items,timings)],validation_outside_timing=validation));save()
    if not all(v['finite']for v in validation):raise RuntimeError('nonfinite repeated component')
  result['all_candidate_targets_match_original_O']=all(v['target_vs_original_O']['changed']==0 for v in result['candidate'])and result['f_a_to_f_b_candidate_chain_target']['changed']==0
  result['all_repeats_stable']=all(v['equal']for s in result['ABBA']for v in s['validation_outside_timing']);result['phase']='completed';save()
 except BaseException as exc:result['error']=repr(exc);result['phase']='failed';save()
 finally:
  if lib is not None:
   closed=[lib.kimi_qkv_destroy_r8553(ctx)==0 for ctx in reversed(contexts)];result['normal_close']=len(closed)==5 and all(closed)
  result.setdefault('memory_snapshots',[]).append(read_memory('after_close'));result['final_token_acceptance']='NOT_MEASURED';result['performance_promotion']=False;save()
 return result

def parent(manifest,library,output):
 base=pathlib.Path(output);assert not base.exists();children=[];started=time.monotonic()
 covered=[4,6]
 for device,rank in enumerate(covered):
  path=base.with_name(base.stem+'-rank'+str(rank)+'.json');log=base.with_name(base.stem+'-rank'+str(rank)+'.log');assert not path.exists()and not log.exists()
  with log.open('x')as f:p=subprocess.Popen([sys.executable,__file__,manifest,library,str(path),'--rank',str(rank),'--device',str(device)],stdout=f,stderr=subprocess.STDOUT)
  children.append((rank,p,path))
  if device==0:
   # Cold context/kernel initialization serialized; at most two workers exist.
   while p.poll()is None:
    if path.exists():
     try:
      phase=json.loads(path.read_text())['phase']
      if phase in('singlefactor512','ABBA','completed','failed'):break
     except (OSError,ValueError,KeyError):pass
    if time.monotonic()-started>360:raise TimeoutError('first requested worker initialization')
    time.sleep(.1)
 results=[]
 for rank,p,path in children:
  code=p.wait(timeout=max(1,480-(time.monotonic()-started)));value=json.loads(path.read_text())if path.exists()else dict(error='Missing worker result',normal_close=False);results.append(dict(rank=rank,exitcode=code,path=str(path),result=value))
 out=dict(scope='Only missing actualprojection ranks4/6; noModel/noTPS',covered_ranks=covered,requested_rank_count=2,device_mapping={'4':0,'6':1},initialization_serialized=True,max_concurrent_workers=2,elapsed_s=time.monotonic()-started,workers=results,all_requested_normal=all(v['result'].get('normal_close')for v in results),all_requested_original_O_W_faithful=all(v['result'].get('all_original_full_outputs_faithful')for v in results),all_requested_candidate_targets_match_O=all(v['result'].get('all_candidate_targets_match_original_O')for v in results),all_repeats_stable=all(v['result'].get('all_repeats_stable')for v in results),final_token_acceptance='NOT_MEASURED',performance_promotion=False)
 base.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items()if k!='workers'}));return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('library');p.add_argument('output');p.add_argument('--rank',type=int);p.add_argument('--device',type=int);a=p.parse_args()
 if a.rank is None:parent(a.manifest,a.library,a.output)
 else:worker(a.manifest,a.library,a.output,a.rank,a.device)
