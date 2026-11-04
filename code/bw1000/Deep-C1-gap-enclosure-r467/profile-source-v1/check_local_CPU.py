"""Changed production continuation/profile/parent control, recording Native leaves."""
from pathlib import Path
from types import SimpleNamespace
from enum import Enum
import ctypes as C,hashlib,importlib,json,sys,tempfile,threading,time
D=Path(__file__).resolve().parent;G=D.parents[3];P=G/'code/bw1000/Deep-HC-fullpre-public-pair-r449/source-v1'
sys.path[:0]=[str(D),str(P)]
from public_deep_model.fullpre_control import QuiescentMode,LEGACY,FULL
from public_deep_model.native_decode import generate
from profile_control import run_loaded
from entry import schema
class Reason(Enum):EOS_TOKEN='EOS';LENGTH='cap';CANCELED='canceled'
class Fn:
 def __init__(self,fn):self.fn=fn;self.argtypes=None;self.restype=None
 def __call__(self,*args):return self.fn(*args)
class Lib:
 def __init__(self,ctx,fail=None):
  self.ctx=ctx;self.owner=0;self.inner=0;self.fail=fail;self.pre=0;self.t1=0;self.mode=0
  for n,f in dict(deep_owner_arm_r8900=self.arm_owner,deep_attention_inner_arm_r9193=self.arm_inner,deep_owner_read_r8900=self.owner_read,deep_attention_inner_read_r9193=self.inner_read,
   deep_hc_fullpre_enable_r377=self.enable,deep_hc_fullpre_enabled_r377=lambda:self.mode,deep_hc_fullpre_clear_r377=lambda:0,deep_hc_fullpre_stats_r377=self.fullstats,
   deep_hc_model_epilogue_set_r366=lambda x:0,deep_hc_model_epilogue_get_r366=lambda:0,deep_hc_model_epilogue_counters_r366=self.legacy,
   modern_native_engine_reset=lambda p:0,modern_native_engine_destroy=lambda p:0,modern_native_engine_forward=lambda *a:0).items():setattr(self,n,Fn(f))
  for n in ['modern_native_engine_reset','modern_native_engine_destroy']:getattr(self,n).argtypes=[C.c_void_p];getattr(self,n).restype=C.c_int
  self.modern_native_engine_forward.argtypes=[C.c_void_p];self.modern_native_engine_forward.restype=C.c_int
 def enable(self,x):self.mode=x;return 0
 def fullstats(self,p,n):
  for i in range(n):p[i]=0
  return 0
 def legacy(self,p,n):
  values=dict(created=86,ready=86,pre=self.pre,eligible_bf16_t1=self.t1,completed=0,other_t=self.pre-self.t1,f32_t1=0)
  for d in range(8):
   for i,k in enumerate(LEGACY):p[d*len(LEGACY)+i]=values[k]
  return 0
 def arm_owner(self,w):
  if self.owner:return 1
  self.owner=w;return 0
 def arm_inner(self,w):
  if self.owner:return 1
  if self.inner and w:return 2
  self.inner=w;return 0
 def owner_read(self):
  assert self.owner==1
  if self.fail=='ownerread':return b'{}'
  q=dict(schema=1,window=1,rows=[])
  for d in range(8):
   for l in range(43):q['rows'].append(dict(device=d,layer=l,owner=d+1,stream=d+11,markers=32,event_create_during_device=d,event_query_during_device=d,spans=[dict(stage=k,start_ms=0.,end_ms=1.,stream_ms=1.,host_wall_ms=1.,thread_cpu_ms=.1)for k in range(16)]))
  self.owner=0;return json.dumps(q).encode()
 def inner_read(self):
  assert self.inner==1 and self.owner==0
  q=dict(schema=1,profile='attention-inner-r9193',GPU_handles_retained=0,scope='RECORDING NOT DEVICE',rows=[])
  for d in range(8):
   for l in range(43):q['rows'].append(dict(device=d,layer=l,kind=0 if l<2 else 1 if l%2 else 2,owner=d+1,stream=d+11,thread=d+21,tokens=1,requests=1,events_created=7,events_queried=7,events_destroyed=7,create_during_device=d,query_during_device=d,destroy_during_device=d,whole_stream_ms=6.,original_final_drain_host_ms=.1,segments=[dict(segment=k,start_ms=float(k),end_ms=float(k+1),stream_ms=1.,host_wall_ms=1.,thread_cpu_ms=.1)for k in range(6)]))
  self.inner=0;return json.dumps(q).encode()
class Context:
 def __init__(self,output_tokens=98,fail=None):
  self.thread=threading.get_ident();self.out=output_tokens;self.generated=0;self.attempts=0;self.resets=0;self.cfg=dict(num_hidden_layers=43,num_nextn_predict_layers=1,compress_ratios=[0,0]+[4 if i%2==0 else 128 for i in range(2,43)]+[0],vocab_size=129280)
  self.binding=SimpleNamespace(lib=Lib(self,fail),check=self.check);self.forward=self.call
 def check_thread(self):assert self.thread==threading.get_ident()
 def check(self,rc):assert rc==0
 def reset(self):self.check_thread();self.resets+=1;self.generated=0
 def floor(self):return[[20*1024**3,64*1024**3]for _ in range(8)]
 def call(self,ids,positions,rows,past,total,offsets):
  self.check_thread();self.attempts+=1;assert rows==[0]and positions==list(range(past[0],total[0]));assert offsets==[0,len(ids)]
  self.binding.lib.pre+=86
  if len(ids)==1:self.binding.lib.t1+=86
  final=total[0]>=8489
  if final:self.generated+=1
  token=1 if self.generated==self.out else 42
  return 0,[[0]],[token]
def main(private_input,config):
 b=Path(private_input).read_bytes();assert hashlib.sha256(b).hexdigest()=='1717324f35a2f7e987928e4a87b7c878a8115f0910f3af56fe4e611cd62c769f';prompt=json.loads(b)['prompt_token_ids'];results=[]
 for name,out,fail in [('full5',98,None),('earlyEOS_no_replacement',8,None),('ownerread_failstop',98,'ownerread')]:
  with tempfile.TemporaryDirectory()as temp:
   ctx=Context(out,fail);control=QuiescentMode(ctx);report={};err=None
   try:run_loaded(ctx,prompt,Path(temp)/'events',report,lambda:None,generate=generate,reasons=Reason,control=control,finite=lambda x:None,deadline=time.time()+300)
   except BaseException as e:err=repr(e)
   if fail:
    assert err and len(report['requests'])==4 and report['requests'][3].get('profile_failures')and control.poisoned
   else:
    assert err is None and len(report['requests'])==5 and ctx.resets==10
    assert report['profile_coverage_complete']==(out==98)
    assert len(report['requests'][3]['profile_windows'])==(2 if out==98 else 0)
   results.append(dict(case=name,passed=True,recording_Native=True,error_expected=bool(fail),actual_requests=len(report['requests']),recording_forward_calls=ctx.attempts,resets=ctx.resets,profile_coverage_complete=report.get('profile_coverage_complete')))
 c=json.loads(Path(config).read_bytes());schema(c);bad=dict(c,rows=4)
 try:schema(bad)
 except AssertionError:pass
 else:raise AssertionError('rows4 was accepted by rows1 profile')
 return dict(passed=True,cases=results,actual_parent_generate=True,actual_parent_QuiescentMode=True,actual_profile_control=True,actual_existing_profile_consumer=True,actual_input_1378_bytes_verified=True,actual_entry_schema=True,Native_leaf_recording=True,Native_dlopen=False,Runtime=False,GPU=False,Model_READY=False,new_jobs=0)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=main(a.input,a.config);Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
