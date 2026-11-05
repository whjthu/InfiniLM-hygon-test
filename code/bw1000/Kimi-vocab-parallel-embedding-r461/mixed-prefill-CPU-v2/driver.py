"""Actual component entry; Native/weight/numeric leaves explicit in CPU recording."""
import ctypes as C,time
import numpy as np
GEOMETRIES=((4608,7168,1),(512,7168,3584))
MODES=(0,1,0,1,1,0) # two cold/cache observations + fixed ABBA

def bind(lib):
 for name,restype,args in [
  ('create',C.c_void_p,[C.c_int]),
  ('prepare',C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]),
  ('run',C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_size_t,C.POINTER(C.c_double),C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t]),
  ('trim',C.c_int,[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t]),
  ('counts',C.c_int,[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t]),('inspect',C.c_int,[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t]),
  ('destroy',C.c_int,[C.c_void_p]),('error',C.c_char_p,[])]:
  fn=getattr(lib,'kimi_mixed_'+name+'_r461');fn.restype=restype;fn.argtypes=args
 return lib

def run(lib,buffers,compare,persist,deadline):
 lib=bind(lib);ctx=None
 q=dict(passed=False,completed=False,planned_arms=12,records=[],inspections=[],counts=None,cleanup_return=None)
 def remaining():
  if time.monotonic()>=deadline:raise TimeoutError('fixed component deadline before next operation')
 def rc(value):
  if value:raise RuntimeError('actual component return '+str(value))
 try:
  remaining();ctx=lib.kimi_mixed_create_r461(0)
  if not ctx:raise RuntimeError('actual Runtime constructor returned null')
  for gid,(m,k,n) in enumerate(GEOMETRIES):
   remaining();x,w=buffers(gid,(m,k,n))
   assert x.dtype==np.uint16 and w.dtype==np.uint16 and x.flags.c_contiguous and w.flags.c_contiguous
   assert x.shape==(m,k)and w.shape==(n,k)
   rc(lib.kimi_mixed_prepare_r461(ctx,gid,x.ctypes.data,x.size,w.ctypes.data,w.size))
   def inspect(phase):
    layout=(C.c_uint64*25)();rc(lib.kimi_mixed_inspect_r461(ctx,layout,25));v=list(layout)
    assert v[0]==0 and v[1]>0 and v[2]==1 and v[3]==gid
    assert v[4:16]==[m,k,k,1,n,k,k,1,k,n,1,k]and all(v[i]for i in [16,17,18])and v[19:]==[1]*6
    q['inspections'].append(dict(geometry=gid,phase=phase,values=v))
   inspect('after_prepare_before_warm')
   for ordinal,mode in enumerate(MODES):
    remaining();out=np.empty((m,n),dtype=np.uint16);metrics=(C.c_double*3)();memory=(C.c_uint64*6)()
    rc(lib.kimi_mixed_run_r461(ctx,mode,out.ctypes.data,out.size,metrics,3,memory,6))
    finite=bool(np.all((out&0x7f80)!=0x7f80))
    rec=dict(geometry=gid,ordinal=ordinal,mode=mode,warm=ordinal<2,shape=[m,n],dtype='BF16',finite=finite,metrics=list(metrics),driver_memory=list(memory),output=persist(gid,ordinal,out))
    q['records'].append(rec)
    if not finite:raise FloatingPointError('nonfinite actual BF16 output')
    rec['CPU_sample_reference']=compare(gid,mode,out,x,w)
    if ordinal<2:
     cold=(C.c_uint64*4)();rc(lib.kimi_mixed_trim_r461(ctx,cold,4));rec['component_only_after_warm_trim']=list(cold)
   inspect('after_ABBA_before_next_prepare')
   del x,w
  counts=(C.c_uint64*4)();rc(lib.kimi_mixed_counts_r461(ctx,counts,4));q['counts']=list(counts)
  assert q['counts']==[3,3,3,3]
  q['passed']=True
 except BaseException as error:q['error']=repr(error)
 finally:
  if ctx:
   q['cleanup_return']=int(lib.kimi_mixed_destroy_r461(ctx))
   if q['cleanup_return']:q['passed']=False
  q['completed']=True;q['missing_arms']=q['planned_arms']-len(q['records'])
 return q
