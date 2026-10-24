"""Finite faithful actual O/W component replay. No model, no generated operands."""
import pathlib,json,hashlib,ctypes,time,os
import numpy as np

def resolve(value):
 p=pathlib.PurePosixPath(value);host=pathlib.PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
 try:return pathlib.Path('/refactor')/p.relative_to(host)
 except ValueError:return pathlib.Path(value)

def run(manifest,library,output):
 m=json.loads(pathlib.Path(manifest).read_text());weight=m['weight'];p=resolve(weight['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==weight['SHA256']
 w=np.frombuffer(raw,dtype='<u2').reshape(weight['shape']);assert w.flags.c_contiguous
 lib=ctypes.CDLL(str(library));lib.kimi_qkv_error_r8545.restype=ctypes.c_char_p
 lib.kimi_qkv_create_r8545.argtypes=[ctypes.c_int,ctypes.c_void_p];lib.kimi_qkv_create_r8545.restype=ctypes.c_void_p
 lib.kimi_qkv_run_r8545.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)]
 lib.kimi_qkv_destroy_r8545.argtypes=[ctypes.c_void_p]
 result=dict(scope='ACTUAL_CAPTURE_COMPONENT_ONLY; no fullmodel or service performance',baseline=[],candidate=None,normal_close=False,source_manifest_SHA256=hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest(),library_SHA256=hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),historical_weight_readback='NOT_CAPTURED; exact baseline output replay is prerequisite')
 def save():pathlib.Path(output).write_text(json.dumps(result,indent=2))
 def compare(actual,expected):
  assert actual.shape==expected.shape;mask=actual!=expected;af=(actual.astype(np.uint32)<<16).view(np.float32).astype(np.float64);ef=(expected.astype(np.uint32)<<16).view(np.float32).astype(np.float64);delta=af-ef;den=float(np.mean(ef*ef));return dict(max_abs=float(np.max(np.abs(delta))),nrmse=float(np.sqrt(np.mean(delta*delta)/den))if den else float(np.sqrt(np.mean(delta*delta))),elements=actual.size,changed=int(mask.sum()),firstdiff=None if not mask.any()else list(map(int,np.argwhere(mask)[0])),SHA256=hashlib.sha256(actual.tobytes()).hexdigest(),expected_SHA256=hashlib.sha256(expected.tobytes()).hexdigest(),finite=bool(np.all((actual&0x7f80)!=0x7f80)))
 cases=[]
 for case in m['cases']:
  arrays={}
  for key in ['input','qkv_projected']:
   q=case['points'][key];data=resolve(q['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==q['SHA256'];arrays[key]=np.frombuffer(data,dtype='<u2').reshape(q['record']['shape'])[0]
  assert arrays['input'].shape[0]in(512,2048)and arrays['input'].shape[1]==7168
  cases.append((case['label'],arrays))
 ctx=lib.kimi_qkv_create_r8545(0,w.ctypes.data);assert ctx,lib.kimi_qkv_error_r8545()
 try:
  outputs={}
  for label,a in cases:
   y=np.empty_like(a['qkv_projected']);elapsed=ctypes.c_double()
   begin=time.monotonic();rc=lib.kimi_qkv_run_r8545(ctx,len(y),a['input'].ctypes.data,y.ctypes.data,0,ctypes.byref(elapsed));capi_wall=time.monotonic()-begin
   item=dict(label=label,return_code=rc,actual_full_token_range=[0,len(y)],target_token_range=[0,512],full_CAPI_wall_s=capi_wall,component_wall_s=elapsed.value,error=None if rc==0 else lib.kimi_qkv_error_r8545().decode())
   if rc==0:item.update(compare(y,a['qkv_projected']));outputs[label]=y.copy()
   result['baseline'].append(item);save()
   if rc or not item['finite']:return result
  result['faithful_original_O_W_reproduced']=all(v['changed']==0 for v in result['baseline']);save()
  if not result['faithful_original_O_W_reproduced']:return result
  label,a=cases[1];assert label=='Wmissing';y=np.empty_like(a['qkv_projected']);elapsed=ctypes.c_double()
  begin=time.monotonic();rc=lib.kimi_qkv_run_r8545(ctx,len(y),a['input'].ctypes.data,y.ctypes.data,512,ctypes.byref(elapsed));capi_wall=time.monotonic()-begin
  result['candidate']=dict(single_factor='Same actual2048 rows/weight; four512-row Gemm calls via unchanged actual2475 helper',return_code=rc,full_CAPI_wall_s=capi_wall,component_wall_s=elapsed.value,actual_full_token_range=[0,len(y)],target_token_range=[0,512])
  if rc==0:result['candidate'].update(vs_original_W=compare(y,a['qkv_projected']),target_vs_original_O=compare(y[:512],outputs['Oconfirm']),final_token_acceptance='NOT_MEASURED',performance_promotion=False)
  else:result['candidate']['error']=lib.kimi_qkv_error_r8545().decode()
 finally:
  result['normal_close']=lib.kimi_qkv_destroy_r8545(ctx)==0;save()
 return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('library');p.add_argument('output');a=p.parse_args();run(a.manifest,a.library,a.output)
