"""Only new real driver/ctypes entry; all Native/data/numeric leaves RECORDING."""
import ctypes as C,json,time,hashlib
from types import SimpleNamespace
import numpy as np
from driver import run,GEOMETRIES
CTX=0x1234567887654321

def scenario(fail=False,expired=False):
 lib=SimpleNamespace();keepers=[];calls=[];errors=[];counts=[0]*4;state={'gid':None,'destroy':0,'samples':0,'persist':0}
 def add(name,restype,types,fn):
  def checked(*args):
   try:return fn(*args)
   except BaseException as error:errors.append(repr(error));return 0 if name=='create' else 1
  f=C.CFUNCTYPE(restype,*types)(checked);keepers.append(f);setattr(lib,'kimi_mixed_'+name+'_r461',f)
 def create(dev):calls.append(['create',dev]);return CTX
 def prepare(ctx,gid,x,nx,w,nw):
  assert ctx==CTX and x and w;m,k,n=GEOMETRIES[gid];assert nx==m*k and nw==n*k
  state['gid']=gid;calls.append(['prepare',gid]);return 0
 def execute(ctx,mode,out,ny,metrics,nm,mem,nb):
  assert ctx==CTX and nm==3 and nb==6;gid=state['gid'];m,k,n=GEOMETRIES[gid];assert ny==m*n
  calls.append(['run',gid,mode])
  if fail and mode==1:return 7
  np.ctypeslib.as_array((C.c_uint16*ny).from_address(out)).fill(0x3f80)
  for i in range(3):metrics[i]=0.001
  for i in range(6):mem[i]=1000000+i
  counts[gid*2+mode]+=1;return 0
 def trim(ctx,out,n):
  assert ctx==CTX and n==4
  for i in range(4):out[i]=i
  calls.append(['trim']);return 0
 def census(ctx,out,n):
  assert ctx==CTX and n==4
  for i,v in enumerate(counts):out[i]=v
  return 0
 def destroy(ctx):assert ctx==CTX;state['destroy']+=1;calls.append(['destroy']);return 0
 add('create',C.c_void_p,[C.c_int],create)
 add('prepare',C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],prepare)
 add('run',C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_size_t,C.POINTER(C.c_double),C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],execute)
 add('trim',C.c_int,[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t],trim)
 add('counts',C.c_int,[C.c_void_p,C.POINTER(C.c_uint64),C.c_size_t],census)
 add('destroy',C.c_int,[C.c_void_p],destroy)
 add('error',C.c_char_p,[],lambda:b'recording')
 def buffers(gid,g):
  m,k,n=g;return np.zeros((m,k),np.uint16),np.zeros((n,k),np.uint16)
 def compare(*args):state['samples']+=1;return {'recording_only':True,'actual_math_executed':False}
 def persist(gid,ordinal,out):state['persist']+=1;return {'bytes':out.nbytes,'SHA256':hashlib.sha256(out.tobytes()).hexdigest(),'recording_output':True}
 result=run(lib,buffers,compare,persist,time.monotonic()+(-1 if expired else 30))
 assert not errors,errors
 assert result['completed']and result['passed']==(not fail and not expired)
 assert state['destroy']==(0 if expired else 1)
 assert len(result['records'])==(0 if expired else 1 if fail else 12)
 assert state['samples']==state['persist']==len(result['records'])
 return dict(passed=True,driver=result,calls=calls,Native_SDK_weights_numeric_leaves='recording only')
if __name__=='__main__':
 result={'passed':True,'success':scenario(),'run_failure':scenario(fail=True),'expired_before_create':scenario(expired=True),'Native_dlopen':False,'GPU':0,'SDK_execution':False}
 print(json.dumps(result))
