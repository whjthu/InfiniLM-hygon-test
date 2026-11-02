from pathlib import Path
import sys,json,ctypes as C,copy
import numpy as np
import interface as api,consumer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cpu-v2'))
import reference as oracle,driver_core
def main():
 # Deliberately artificial enum codes: metadata, not guessed BF16/F32 shapes.
 dtype=dict(F32=123,BF16=124,I64=125,U8=126)
 def field(a,k):return dict(raw=np.ascontiguousarray(a).reshape(-1).view(np.uint8).copy(),shape=a.shape,dtype=dtype[k])
 raw=np.zeros((128,8448),np.uint8)
 for slot in range(2124):
  p,r=divmod(slot,64);raw[p,r*128:(r+1)*128]=56;raw[p,8192+r*4:8196+r*4]=np.array([1],np.float32).view(np.uint8)
 fields={0:field(np.full((1,8,128),56,np.uint8),'U8'),1:field(np.ones((1,8),np.float32),'F32'),2:None,3:None,4:field(np.full((1,2124),8192,np.float32),'F32'),5:field(np.arange(512,dtype=np.int64)[None,:],'I64'),6:field(raw,'U8'),7:field(np.ones((8,512),np.float32),'F32'),8:field(raw.copy(),'U8'),9:field(np.zeros((8,512),np.float32),'F32'),10:field(np.zeros((8,512),np.float32),'F32')}
 class Fn:
  def __init__(self,f):self.f=f
  def __call__(self,*args):return self.f(*args)
 class Lib:
  def __init__(self):
   for name in ['error','id_size','id','create','keys','load','finish','reset','forward','proof_info','proof_read','stats','dtype','destroy','shutdown']:setattr(self,'deep_c4_status_'+name+'_r406',Fn(getattr(self,name,lambda *a:0)))
  def proof_info(self,ctx,k,shape,size,dt):
   f=fields[k]
   if f is None:C.cast(size,C.POINTER(api.Z))[0]=0;shape[0]=0;C.cast(dt,C.POINTER(api.I))[0]=-1;return 0
   shape[0]=len(f['shape'])
   for i,n in enumerate(f['shape']):shape[i+1]=n
   C.cast(size,C.POINTER(api.Z))[0]=f['raw'].nbytes;C.cast(dt,C.POINTER(api.I))[0]=f['dtype'];return 0
  def proof_read(self,ctx,k,out,size):assert size==fields[k]['raw'].nbytes;C.memmove(out,int(fields[k]['raw'].ctypes.data),size);return 0
 f=api.configure(Lib());read={k:api.proof(f,lambda rc:None,None,k)for k in range(11)};assert consumer.equal(read[7],fields[7])
 tested=0;previous=None
 for item in driver_core.plan():
  st=dict(counts=[1,1,1,0,0],errors=[0]*5,clocks_valid=True,clock_reads=12 if item['timing']else 0,status_wall_ns=[20,30,40,0,0]if item['timing']else[0]*5,status_thread_CPU_ns=[10,15,20,0,0]if item['timing']else[0]*5)
  packets=[dict(rank=i,item=item,stats=copy.deepcopy(st),fields=read)for i in range(8)];q=consumer.qualify(packets,oracle,dtype,previous);assert len(q)==8 and all(v['CPU_FP8_score_NRMSE']==0 for v in q);previous=packets;tested+=1
 for key,change in [('clock',lambda p:p[0]['stats'].update(clock_reads=12)),('state_dtype',lambda p:p[0]['fields'][7].update(dtype=-999)),('shape',lambda p:p[0]['fields'][10].update(shape=(64,64))),('count',lambda p:p[0]['stats'].update(counts=[0,0,0,0,0]))]:
  packets=copy.deepcopy(previous);change(packets)
  try:consumer.qualify(packets,oracle,dtype)
  except AssertionError:continue
  raise AssertionError(('Negative not rejected',key))
 return dict(passed=True,actual_ctypes_proof_info_and_read=True,real_metadata_dtype_and_shape=True,five_query_CPU_oracle_calls=tested,TopK_cutoff_tie_accepted=True,OFF_clock_ON_count_wrongdtype_wrongshape_negative=True,GPU_qualification=False)
if __name__=='__main__':print(json.dumps(main()))
