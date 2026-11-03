"""Actual Python owner plan/interface/proof consumer with fake device leaves only."""
from pathlib import Path
import sys,json,ctypes as C,copy,hashlib,ast,threading,concurrent.futures
import numpy as np
import interface as api,consumer,reference as oracle,driver_core,freeze_control

def main():
 S=Path(__file__).resolve().parent;assert hashlib.sha256((S/'PARENT_PINS.json').read_bytes()).hexdigest()=='2663d8df9fd1cb0759c868570bfcf0dadd52ceab7b3b99fb551d6424b1d6cfe8'
 assert [p['mode']for p in driver_core.plan()]==[0,1,0,1,1,0,1,0,0,1]
 dtype=dict(F32=123,BF16=124,I64=125,U8=126)
 def field(a,k):return dict(raw=np.ascontiguousarray(a).reshape(-1).view(np.uint8).copy(),shape=a.shape,dtype=dtype[k])
 raw=np.zeros((128,8448),np.uint8)
 for slot in range(2124):
  p,r=divmod(slot,64);raw[p,r*128:(r+1)*128]=56;raw[p,8192+r*4:8196+r*4]=np.array([1],np.float32).view(np.uint8)
 fields={0:field(np.full((1,8,128),56,np.uint8),'U8'),1:field(np.ones((1,8),np.float32),'F32'),2:None,3:None,4:field(np.full((1,2124),8192,np.float32),'F32'),5:field(np.arange(512,dtype=np.int64)[None,:],'I64'),6:field(raw,'U8'),7:field(np.full((8,512),0x3f80,np.uint16),'BF16'),8:field(raw.copy(),'U8'),9:field(np.zeros((8,512),np.uint16),'BF16'),10:field(np.zeros((8,512),np.uint16),'BF16'),11:field(np.full((1,512),8192,np.float32),'F32')}
 class Fn:
  def __init__(self,f):self.f=f
  def __call__(self,*args):return self.f(*args)
 class Lib:
  mode=0
  def __init__(self):
   self.fields=copy.deepcopy(fields);self.captured=None;self.restores=0;self.generation=0
   for name in ['error','id_size','id','create','keys','load','finish','reset','forward','proof_info','proof_read','proof_pair','stats','freeze','restore','freeze_stats','dtype','destroy','shutdown']:setattr(self,'deep_c4_topk_'+name+'_r416',Fn(getattr(self,name,lambda *a:0)))
  def proof_info(self,ctx,k,shape,size,dt):
   f=self.fields[k]
   if f is None:C.cast(size,C.POINTER(api.Z))[0]=0;shape[0]=0;C.cast(dt,C.POINTER(api.I))[0]=-1;return 0
   shape[0]=len(f['shape'])
   for i,n in enumerate(f['shape']):shape[i+1]=n
   C.cast(size,C.POINTER(api.Z))[0]=f['raw'].nbytes;C.cast(dt,C.POINTER(api.I))[0]=f['dtype'];return 0
  def proof_read(self,ctx,k,out,size):assert k!=5 and size==self.fields[k]['raw'].nbytes;C.memmove(out,int(self.fields[k]['raw'].ctypes.data),size);return 0
  def proof_pair(self,ctx,ids,ids_size,values,values_size):
   assert ids_size==self.fields[5]['raw'].nbytes and values_size==self.fields[11]['raw'].nbytes
   C.memmove(ids,int(self.fields[5]['raw'].ctypes.data),ids_size);C.memmove(values,int(self.fields[11]['raw'].ctypes.data),values_size);return 0
  def freeze(self,ctx):
   assert self.captured is None;self.captured=copy.deepcopy({k:self.fields[k]for k in [6,7]});return 0
  def restore(self,ctx):
   assert self.captured is not None
   for k in [6,7]:self.fields[k]=copy.deepcopy(self.captured[k])
   self.restores+=1;return 0
  def freeze_stats(self,ctx,t,a):
   t[0]=.001;t[1]=.0001
   for i,n in enumerate([1,self.restores,280368,8192,4,self.generation,self.generation,100+self.generation*10]):a[i]=n
   return 0
  def stats(self,ctx,t,a):t[0]=.002;t[1]=.00001;a[0]=1;a[1]=self.mode;a[2]=1-self.mode;a[3]=threading.get_ident();return 0
 packets=[[None]*8 for _ in range(10)];completed=[]
 def owner(rank,fail=None):
  calls=[];control={};tid=threading.get_ident();lib=Lib();f=api.configure(lib)
  def create(r):calls.append(('create',r));return tid
  def trial(ctx,item):
   assert ctx==threading.get_ident();calls.append(('reset',item['index']))
   for past in range(0,8496,1024):calls.append(('prefix',past,min(1024,8496-past)))
   calls.append(('attempt',item['index']))
   if item['index']==fail:raise RuntimeError('original status=2')
   lib.generation=item['index']+1
   stage=freeze_control.stage(f,lambda rc:None,ctx,item,api,oracle,dtype,control,{'hidden':'sameH','q':'sameQ','positions':'8496'});assert stage['same_frozen_effective_payload']
   lib.mode=item['mode'];read={k:api.proof(f,lambda rc:None,ctx,k)for k in range(11)if k!=5};read[5],read[11]=api.proof_pair(f,lambda rc:None,ctx);st=api.stats(f,lambda rc:None,ctx);r=dict(rank=rank,item=item,fields=read,stats=st);packets[item['index']][rank]=r;calls.append(('completed',item['index']));return r
  def destroy(ctx):assert ctx==tid;calls.append(('destroy',rank))
  def shutdown():calls.append(('shutdown',rank))
  try:rows=driver_core.run_owner(rank,create,trial,destroy,shutdown,lambda:calls.append(('abort',rank)))
  except RuntimeError as e:
   assert fail is not None and str(e)=='original status=2'and calls[-1][0]=='shutdown'and len([c for c in calls if c[0]=='completed'])==fail;return
  assert len(rows)==10 and sum(c[0]=='prefix'for c in calls)==90 and calls[-2][0]=='destroy'and calls[-1][0]=='shutdown';completed.append(rank)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
 assert sorted(completed)==list(range(8));previous=None
 for row in packets:assert len(consumer.qualify(row,oracle,dtype,previous))==8;previous=row
 for key,change in [('hit',lambda p:p[0]['stats'].update(topk_hits=0)),('value',lambda p:p[0]['fields'][11]['raw'].__setitem__(0,1)),('state_dtype',lambda p:p[0]['fields'][7].update(dtype=-999))]:
  rows=copy.deepcopy(packets[-1]);change(rows)
  try:consumer.qualify(rows,oracle,dtype)
  except AssertionError:continue
  raise AssertionError(('negative_not_rejected',key))
 for fail in [0,1,2,9]:owner(0,fail)
 # Isolated fresh lib, actual stage rejects changedinput and actual raw/state proof after restore.
 lib=Lib();f=api.configure(lib);control={};item=driver_core.plan()[0];lib.generation=1
 freeze_control.stage(f,lambda rc:None,1,item,api,oracle,dtype,control,{'hidden':'same'})
 lib.generation=2
 try:freeze_control.stage(f,lambda rc:None,1,driver_core.plan()[1],api,oracle,dtype,control,{'hidden':'different'})
 except AssertionError:pass
 else:raise AssertionError('changed input accepted')
 original_restore=lib.restore
 def corrupt_restore(ctx):
  rc=original_restore(ctx);lib.fields[7]['raw'][0]^=1;return rc
 lib.deep_c4_topk_restore_r416=Fn(corrupt_restore)
 try:freeze_control.stage(f,lambda rc:None,1,driver_core.plan()[1],api,oracle,dtype,control,{'hidden':'same'})
 except AssertionError:pass
 else:raise AssertionError('changed restored state accepted')
 for p in S.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 return dict(passed=True,CPU_only=True,Native_dlopen=False,GPU=False,actual_shared_ten_query_owner_driver=True,actual_freeze_restore_stage=True,owner_records=8,query_records=80,prefix_transactions_per_owner=90,actual_ctypes_info_pair_read_stats=True,three_semantic_negative_cases=True,primary_error_and_close=True,first_ON_warm_before_formal=True,original_FP8_score_oracle=True,scope='Fake device leaves; actual CABI configure/proof-pair consumer and owner runner. C++ dispatch/header validated separately; no device qualification.')
if __name__=='__main__':print(json.dumps(main()))
