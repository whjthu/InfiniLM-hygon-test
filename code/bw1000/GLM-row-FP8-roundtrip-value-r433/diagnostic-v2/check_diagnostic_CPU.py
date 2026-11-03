"""New diagnostic FFI/file/error glue only; old872 suites not executed."""
from pathlib import Path
import ctypes,hashlib,json,struct,tempfile
from binding import bind_diagnostic
from diagnostic_io import read_snapshot,validate_descriptor
def run():
 context=ctypes.create_string_buffer(32);ctx=ctypes.addressof(context);assert ctx>2**32
 payload=struct.pack('<HH',0x3f80,0x4000);desc=dict(quiesced=True,mode=1,detail=1,ticket=1,query_CAPI_completed=True,total_bytes=4,slots=[dict(key=0,name='actual_attention_BF16',shape=[1,2],dtype='BF16',nbytes=4)])
 desc_buffer=ctypes.create_string_buffer(json.dumps(desc).encode());seen=[];state=dict(copyfail=False,clearfail=False)
 @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int)
 def arm(p,d):seen.append(('arm',p,d));return 0 if p==ctx and d in [0,1,2]else 1
 @ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p)
 def descriptor(p):seen.append(('desc',p));return ctypes.addressof(desc_buffer)if p==ctx else None
 @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t)
 def read(p,k,out,n):
  seen.append(('read',p,k,n));
  if p!=ctx or k!=0 or not out or n!=4 or state['copyfail']:return 1
  ctypes.memmove(out,payload,4);return 0
 @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p)
 def clear(p):seen.append(('clear',p));return 1 if p!=ctx or state['clearfail']else 0
 class Lib:pass
 lib=Lib()
 for name,callback in [('arm',arm),('desc',descriptor),('read',read),('clear',clear)]:setattr(lib,'glm_attention_diag_'+name+'_r433',ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(callback,ctypes.c_void_p).value))
 bind_diagnostic(lib);assert lib.glm_attention_diag_arm_r433(ctx,1)==0 and lib.glm_attention_diag_arm_r433(ctx,3)==1
 with tempfile.TemporaryDirectory()as temp:
  q=read_snapshot(lib,ctx,temp,'success',mode=1,detail=1);assert q['completed']and q['collection_passed']and q['clear_RC']==0 and not q['query_qualification']
  assert Path(q['files'][0]['path']).read_bytes()==payload and q['files'][0]['SHA256']==hashlib.sha256(payload).hexdigest()
  state.update(copyfail=True,clearfail=True)
  try:read_snapshot(lib,ctx,temp,'copy-and-clear-error',mode=1,detail=1)
  except RuntimeError as e:assert 'copy failed'in str(e)
  else:raise AssertionError('missing copyfailure')
  q=json.loads((Path(temp)/'copy-and-clear-error.json').read_bytes());assert q['completed']and not q['collection_passed']and q['clear_error']and 'copy failed'in q['error']
  state.update(copyfail=False,clearfail=False);desc['query_CAPI_completed']=False
  desc_buffer=ctypes.create_string_buffer(json.dumps(desc).encode())
  q=read_snapshot(lib,ctx,temp,'partial-written-tensors',mode=1,detail=1)
  assert q['collection_passed']and not q['descriptor']['query_CAPI_completed']and not q['query_qualification']and q['clear_RC']==0
 # Independent schema refusals happen before allocating/copying device data.
 bad=json.loads(json.dumps(desc));bad['slots'][0]['nbytes']=3
 try:validate_descriptor(bad,1,1)
 except AssertionError:pass
 else:raise AssertionError('bad bytecount accepted')
 source=(Path(__file__).parent/'attention_component_r433.cpp').read_text();header=(Path(__file__).parent/'value_roundtrip_r433.hpp').read_text()
 assert 'diagnostic_tickets>=2'in source and 'if(own)drain(c);return 1;'in source and 'query_completed=true'in source
 assert 'identity_device_check_passed=true'in header and 'identity_check_host' in header and 'Diagnostic*diagnostic=nullptr'in header
 return dict(passed=True,actual_new_typed_FFI_highcontext=True,actual_rawfile_and_SHA=True,copyfailure_saved_and_cleared=True,clear_error_preserves_copy_primary=True,source_bounded_owner_snapshots=True,recorded_only_Native_leaves=True,old872_geometry_FFI_suites_rerun=False,Native_dlopen=False,Runtime=False,GPU=0,weights=False,math_qualified=False)
if __name__=='__main__':print(json.dumps(run(),indent=2))
