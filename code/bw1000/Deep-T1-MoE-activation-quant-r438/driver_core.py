"""Actual thin complete-forward caller; device/transport leaves explicit in CPU."""
import ctypes as C,time
from interface import check

def owner_session(lib,device,config,uid,load_weights,body):
 """Shared actual owner entry. Loader and device body are explicit CPU leaves."""
 ctx=None;primary=None;cleanup=[];result=None
 try:
  ctx=lib.deep_actquant_create_r438(device,config,uid)
  if not ctx:raise RuntimeError(lib.deep_actquant_error_r438())
  load_weights(lib,ctx);check(lib,lib.deep_actquant_finish_r438(ctx));result=body(lib,ctx)
 except BaseException as exc:primary=exc
 finally:
  if ctx:
   try:check(lib,lib.deep_actquant_destroy_r438(ctx))
   except BaseException as exc:cleanup.append(repr(exc))
  try:check(lib,lib.deep_actquant_shutdown_r438())
  except BaseException as exc:cleanup.append(repr(exc))
 if primary is not None:
  if cleanup:primary.add_note('r438 cleanup errors: '+repr(cleanup)) if hasattr(primary,'add_note')else None
  raise primary
 if cleanup:raise RuntimeError('r438 cleanup failed: '+repr(cleanup))
 return result
def forward(lib,ctx,input_ptr,producer,*,tokens,mode,capture=False,counts=None):
 if counts is None:counts={'attempted':0,'completed':0}
 counts['attempted']+=1;start=time.perf_counter()
 code=lib.deep_actquant_forward_r438(ctx,tokens,input_ptr,producer,mode,int(capture));check(lib,code)
 counts['completed']+=1;ended=time.perf_counter();seconds=(C.c_double*2)();stats=(C.c_uint64*7)();check(lib,lib.deep_actquant_stats_r438(ctx,seconds,stats))
 for role in range(2):
  calls,hits,fallbacks=stats[role*3:role*3+3];assert calls==1 and hits==int(tokens==1 and mode==1)and fallbacks==1-hits
 return dict(forward_CAPI_s=ended-start,CPP_returned_s=seconds[0],CPP_complete_s=seconds[1],counts=dict(counts),routed=list(stats[:3]),shared=list(stats[3:6]),actual_owner_thread=stats[6],capture=bool(capture))
def proof(lib,ctx,which):
 shape=(C.c_size_t*5)();size=C.c_size_t();dtype=C.c_int();check(lib,lib.deep_actquant_proof_info_r438(ctx,which,shape,C.byref(size),C.byref(dtype)))
 if size.value==0:assert shape[0]==0 and dtype.value==-1;return None
 assert 1<=shape[0]<=4;dims=tuple(shape[1:1+shape[0]]);assert all(v>0 for v in dims)
 raw=(C.c_ubyte*size.value)();check(lib,lib.deep_actquant_proof_read_r438(ctx,which,raw,size.value))
 # Consumer must decode by the returned actual dtype, shape and byte count.
 return dict(shape=dims,dtype=dtype.value,raw=bytes(raw),bytes=size.value)
