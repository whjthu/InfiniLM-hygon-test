import ctypes as C,time,json
import numpy as np
from batch_wire_r9128 import send,recv
P=C.c_void_p;S=C.c_size_t;I=C.c_int
class Native:
 def __init__(self,g):
  self.lib,self.ctx,self.control,self.node,self.tail,self.emitter,self.vocab=[g[k]for k in ['lib','ctx','control','node','tail','emitter','vocab']];self.card_count=int(g.get("pp",2))*16+int(__import__("os").environ.get("KIMI_DRAFT_CARD_COUNT_R9142","1"));self.memory_guard=g.get("memory_guard");self.observer=g.get("host_observer");self.peer=None;self.peers={};self.tp_nodes=int(g.get("tp_nodes",2));self.pp=int(g.get("pp",2));self.serial=0;self.epoch=0;self.calls=[];L=self.lib
  sig={'reset':[P,I],'mode':[P,I],'prefill':[P,S,S]+[P]*7,'forward':[P,S,S,I]+[P]*11,'commit':[P,S,P,P,P],'replay':[P,I,P,P,P],'rows':[P,S,P,P],'stats':[P,P,S],'taps_begin':[P,C.c_uint64,I,S,P,P,P],'taps_count':[P,P],'taps_info':[P,S,P,S],'taps_copy':[P,S,P,S]}
  for name,args in sig.items():getattr(L,'modern_kimi_batch_'+name+'_r9122').argtypes=args
  L.modern_kimi_batch_capabilities_r9122.argtypes=[P];L.modern_kimi_batch_capabilities_r9122.restype=C.c_char_p;self.cap=json.loads(L.modern_kimi_batch_capabilities_r9122(self.ctx));self.maxT=int(self.cap['implemented_max_flatten_tokens']);self.maxC=min(int(self.cap.get('logical_capacity',8)),int(self.cap['implemented_max_requests']));assert self.cap['native_batch']and self.maxT==64 and self.maxC in [8,16,32] and self.cap['pool_rows']==2*self.maxC+1 and self.cap['flags']==3
  L.modern_kimi_batch_taps_capabilities_r9122.argtypes=[];L.modern_kimi_batch_taps_capabilities_r9122.restype=C.c_uint64;assert L.modern_kimi_batch_taps_capabilities_r9122()==0x01200500
 def ck(self,v):
  if v:raise RuntimeError(self.lib.modern_kimi_engine_error_r2475().decode())
 @staticmethod
 def ptr(a):return P(a.ctypes.data)
 def reset(self,state):
  self.ck(self.lib.modern_kimi_batch_reset_r9122(self.ctx,int(state)));self.ck(self.lib.modern_kimi_batch_mode_r9122(self.ctx,int(state)))
 def rows(self,ids):
  a=np.asarray(ids,np.int32);out=np.empty((len(a),4),np.int32);self.ck(self.lib.modern_kimi_batch_rows_r9122(self.ctx,len(a),self.ptr(a),self.ptr(out)));return out.tolist()
 def stats(self):
  a=(C.c_uint64*64)();self.ck(self.lib.modern_kimi_batch_stats_r9122(self.ctx,a,64));return list(a)
 def I192(self):
  f=self.lib.modern_kimi_I192_counts_r8669y;f.argtypes=[I,P,S];out=[]
  for d in range(8):a=(C.c_uint64*2)();self.ck(f(d,a,2));out.append(list(a))
  return out
 def Tstats(self):
  f=self.lib.modern_kimi_I192_T_counts_r9129;f.argtypes=[I,P,S];out=[]
  for device in range(8):a=(C.c_uint64*65)();self.ck(f(device,a,65));out.append(list(a))
  return out
 def arm(self,ids,past,width,on):
  self.epoch+=1
  if not on:self.ck(self.lib.modern_kimi_batch_taps_begin_r9122(self.ctx,self.epoch,0,0,None,None,None));return
  a,b,c=[np.asarray(x,np.int32)for x in [ids,past,width]];self.ck(self.lib.modern_kimi_batch_taps_begin_r9122(self.ctx,self.epoch,1,len(a),self.ptr(a),self.ptr(b),self.ptr(c)))
 def harvest(self,ids,past,width):
  n=C.c_size_t();self.ck(self.lib.modern_kimi_batch_taps_count_r9122(self.ctx,C.byref(n)));T=sum(width);layers={};representative=self.node%self.tp_nodes==0
  assert representative or n.value==0
  if representative and self.node!=0:send(self.peer,{'serial':self.serial,'node':self.node,'count':n.value})
  for index in range(n.value):
   v=(C.c_int64*(5+3*len(ids)))();self.ck(self.lib.modern_kimi_batch_taps_info_r9122(self.ctx,index,v,len(v)));layer=int(v[1]);assert layer in [19,37,66,78,90] and list(v)==[self.epoch,layer,T,7168,len(ids)]+[x for triple in zip(ids,past,width)for x in triple];raw=(C.c_uint8*(T*7168*2))();self.ck(self.lib.modern_kimi_batch_taps_copy_r9122(self.ctx,index,raw,len(raw)));body=bytes(raw);bits=np.frombuffer(body,'<u2');assert not np.any((bits&0x7f80)==0x7f80)
   if self.node!=0:send(self.peer,{'serial':self.serial,'layer':layer,'tokens':T},body)
   else:assert layer not in layers;layers[layer]=bits.reshape(T,7168)
  if self.node!=0:return None
  for rank,peer in sorted(self.peers.items()):
   h,body=recv(peer);assert h['serial']==self.serial and h['node']==rank and not body and 0<=h['count']<=5
   for _ in range(h['count']):
    h,body=recv(peer);layer=h['layer'];assert h['serial']==self.serial and h['tokens']==T and layer in [19,37,66,78,90] and layer not in layers and len(body)==T*7168*2;layers[layer]=np.frombuffer(body,'<u2').reshape(T,7168)
  assert set(layers)=={19,37,66,78,90}
  result={};offset=0
  for key,N in zip(ids,width):result[key]=np.concatenate([layers[k][offset:offset+N]for k in [19,37,66,78,90]],axis=-1).tobytes();offset+=N
  return result
 def call(self,kind,ids,past,tokens,capture,on_selected=None):
  widths=list(map(len,tokens));T=sum(widths);a=np.asarray(ids,np.int32);pp=np.asarray(past,np.int32);total=pp+np.asarray(widths,np.int32);offset=np.asarray([0]+list(np.cumsum(widths)),np.int32);flat=np.asarray([x for row in tokens for x in row],np.int64);pos=np.asarray([p+j for p,N in zip(past,widths)for j in range(N)],np.int64);nout=len(a)if kind=='prefill'else T;logits=np.full((nout,self.vocab),0x7fc0,np.uint16);out=np.full(nout,-1,np.int64);flags=np.asarray([-1],np.int32);self.arm(ids,past,widths,capture);t=time.perf_counter()
  if kind=='prefill':self.ck(self.lib.modern_kimi_batch_prefill_r9122(self.ctx,T,len(a),*[self.ptr(x)for x in [a,offset,flat,pos,logits,out,flags]]))
  elif kind=='verify':
   rows=self.rows(ids);assert all(r[2]==p and r[3]==1 for r,p in zip(rows,past));read=np.asarray([r[0]for r in rows],np.int32);write=np.asarray([r[1]for r in rows],np.int32);assert T<=self.maxT and all(1<=x<=8 for x in widths);self.ck(self.lib.modern_kimi_batch_forward_r9122(self.ctx,T,len(a),3,*[self.ptr(x)for x in [read,write,a,pp,total,offset,flat,pos,logits,out,flags]]))
  else:raise ValueError(kind)
  return self.finish(kind,ids,past,widths,capture,logits,out,flags,time.perf_counter()-t,on_selected,tokens if kind=="verify"else None)
 def finish(self,kind,ids,past,width,capture,logits,out,flags,wall,callback,token_rows=None):
  assert flags[0]==int(self.tail)+2*int(self.node==self.emitter)
  if self.tail:assert not np.any((logits&0x7f80)==0x7f80)
  gap=None
  if kind=='verify'and self.node==self.emitter and getattr(self,'relaxed_gap_epsilon',0)>0:
   from host_gap_r9258 import scalar_gaps
   gap=scalar_gaps(logits,out.tolist(),token_rows)
  p=self.control.collect('batch-call:'+str(self.serial),{'wall':wall,'selected':out.tolist()if self.node==self.emitter else None,'relaxed_gap':gap});selected=p[self.emitter]['selected'];gap=p[self.emitter]['relaxed_gap'];assert all(0<=x<self.vocab for x in selected)
  if callback:
   if kind=='verify':callback(selected,gap)
   else:callback(selected)
  t=time.perf_counter();features=self.harvest(ids,past,width)if capture else None;record={'serial':self.serial,'kind':kind,'logical':list(ids),'past':list(past),'width':list(width),'CAPI_maxhost_s':max(v['wall']for v in p),'feature_harvest_local_s':time.perf_counter()-t,'relaxed_gap':gap};self.calls.append(record);self.serial+=1;return selected,features,record
 def commit(self,ids,decisions):
  a=np.asarray(ids,np.int32);b=np.asarray([d.accepted for d in decisions],np.uintp);c=np.asarray([int(not d.terminal)for d in decisions],np.int32);t=time.perf_counter();self.ck(self.lib.modern_kimi_batch_commit_r9122(self.ctx,len(a),*[self.ptr(x)for x in [a,b,c]]));return time.perf_counter()-t
 def replay(self,partial,state):
  ids=[x[0]for x in partial];past=[x[1]for x in partial];width=[x[2]for x in partial];T=sum(width);flags=np.asarray([-1],np.int32)
  if state:
   t=time.perf_counter();self.ck(self.lib.modern_kimi_batch_replay_r9122(self.ctx,1,None,None,self.ptr(flags)));assert flags[0]==0;p=self.control.collect('state-batch-replay:'+str(self.serial),{'wall':time.perf_counter()-t});return None,{'CAPI_maxhost_s':max(x['wall']for x in p),'state_only':True}
  self.arm(ids,past,width,True);logits=np.full((T,self.vocab),0x7fc0,np.uint16);out=np.full(T,-1,np.int64);t=time.perf_counter();self.ck(self.lib.modern_kimi_batch_replay_r9122(self.ctx,0,self.ptr(logits),self.ptr(out),self.ptr(flags)));_,features,record=self.finish('full-replay',ids,past,width,True,logits,out,flags,time.perf_counter()-t,None);return features,record
