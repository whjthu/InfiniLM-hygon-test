"""Actual libffi pointer transport through the production declare function.

No compiler, Native DSO, Torch, device, or Slurm. The system CDLL only holds real
in-memory C function-pointer endpoints, not Python methods accepting arbitrary
argtypes. The callbacks never dereference a synthetic context address.
"""
from pathlib import Path
import ctypes,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_glm_native.checkpoint_loader import declare

def main():
 assert ctypes.sizeof(ctypes.c_void_p)==8
 high=0x1234567812345678;seen=[];keep=[];lib=ctypes.CDLL('kernel32.dll'if sys.platform=='win32'else None)
 names=['modern_glm_model_owned_prefix_policy_r1669','modern_glm_model_feature_bypass_r1406',
        'modern_glm_model_indexer_policy_r998','modern_glm_model_decode_leases_policy_r1356']
 prototype=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int)
 for index,name in enumerate(names):
  def callback(ctx,mode,index=index,name=name):
   seen.append(dict(name=name,ctx=ctx,mode=mode));return 20+index if ctx==high else -71
  target=prototype(callback);keep.append(target)
  entry=ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(target,ctypes.c_void_p).value);entry.argtypes=None
  setattr(lib,name,entry)
 old=getattr(lib,names[0]);old_behavior={}
 try:
  rc=old(high,0);old_behavior.update(auto_result=rc,auto_ctx=seen[-1]['ctx'])
  assert seen[-1]['ctx']!=high
 except ctypes.ArgumentError as error:
  # Current Windows Python refuses overflowing implicit integers. Linux3.10 in
  # original862 reached the actual CAPI and SIGSEGV. Do not invent a platform
  # equivalence: separately exercise the default c_int representation below.
  old_behavior.update(auto_rejected=repr(error),auto_callback_invoked=False)
 old(ctypes.c_int(high),0);assert seen[-1]['ctx']!=high
 old_behavior.update(explicit_default_c_int_ctx=seen[-1]['ctx'],high_ctx=high,loss_demonstrated=True)
 # Base declarations are real C function pointers too, but not invoked here.
 base=['modern_glm_model_error','modern_glm_model_id_size','modern_glm_model_id','modern_glm_model_create',
       'modern_glm_model_keys','modern_glm_model_load_batch','modern_glm_model_finish','modern_glm_model_forward',
       'modern_glm_model_control','modern_glm_model_reset','modern_glm_model_destroy']
 for name in base:
  target=ctypes.CFUNCTYPE(ctypes.c_int)(lambda:0);keep.append(target)
  setattr(lib,name,ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(target,ctypes.c_void_p).value))
 binding=declare(lib);assert binding.lib is lib;results=[]
 for index,name in enumerate(names):
  entry=getattr(lib,name);assert entry.argtypes==[ctypes.c_void_p,ctypes.c_int]and entry.restype is ctypes.c_int
  rc=entry(high,index%2);assert rc==20+index and seen[-1]==dict(name=name,ctx=high,mode=index%2)
  results.append(dict(name=name,ctx_readback=seen[-1]['ctx'],mode=seen[-1]['mode'],returncode=rc,passed=True))
 p=Path(__file__).resolve().parent
 q=dict(completed=True,passed=True,Python=sys.version,platform=sys.platform,pointer_bits=64,
  production_declare_SHA256=hashlib.sha256((p/'public_glm_native/checkpoint_loader.py').read_bytes()).hexdigest(),
  test_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  real_libffi_callbacks=True,old_default_int=old_behavior,setters=results,
  Native_DSO_opened=False,Native_compiled=False,GPU=0,Slurm=0,weights=0,
  old_leaf_suites_rerun=False,scope='Four real in-memory C ABI calls through actual declare; pointer transport, not Native/model/runtime qualification')
 (p/'FFI-CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
