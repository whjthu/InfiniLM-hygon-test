"""Execute the actual new protocol's nested forward consumer with a CPU-only
fake CAPI. Contract test only, no GPU/model output or performance evidence.
"""
import ast,pathlib,ctypes,json,hashlib,time,os
import numpy as np
from kimi_interleaved_prefill_decode_r8669zq import run_interleaved
r=pathlib.Path(__file__).resolve().parent;s=(r/'kimi_Model_events_protocol_r8669zq.py').read_text();tree=ast.parse(s);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run');actual=next(x for x in run.body if isinstance(x,ast.FunctionDef)and x.name=='forward');code=ast.unparse(actual)
wrap='def bind(env):\n '+"lib,ctx,control,g,report,node,tail,emitter,vocab,custom,moe,policy,candidate,counts,custom_counts=[env[k] for k in ['lib','ctx','control','g','report','node','tail','emitter','vocab','custom','moe','policy','candidate','counts','custom_counts']]"+'\n seq=0;last_local_ready=0.\n '+code.replace('\n','\n ')+'\n return forward,lambda:last_local_ready\n'
namespace={'np':np,'ctypes':ctypes,'time':time,'hashlib':hashlib,'pipeline':True};exec(compile(wrap,'<actual-protocol-forward>','exec'),namespace)
class F:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*args):return self.fn(*args)
class Control:
 def collect(self,label,p):return [p.copy()for _ in range(4)]
results=[]
for fault in [False,True]:
 gold=[[1000+i*128+j for j in range(128)]for i in range(16)];generated=[0]*16;custom_values=[[0,0]for _ in range(8)];seen=[]
 class Lib:pass
 lib=Lib();lib.modern_kimi_vocab_head_get_r8341=F(lambda:0)
 def native(ctx,N,B,flags,*ptrs):
  rows,past,total,offset,ids,positions,logits,out,outflags=ptrs
  arr=lambda p,n,t:np.ctypeslib.as_array(ctypes.cast(p,ctypes.POINTER(t)),shape=(n,))
  rr=arr(rows,B,ctypes.c_int32);pp=arr(past,B,ctypes.c_int32);tt=arr(total,B,ctypes.c_int32);oo=arr(offset,B+1,ctypes.c_int32);ii=arr(ids,N,ctypes.c_int64);pos=arr(positions,N,ctypes.c_int64);ys=arr(out,B,ctypes.c_int64)
  assert rr.tolist()==sorted(set(rr.tolist()))and oo[0]==0 and oo[-1]==N and flags in[3,7]
  for j,row in enumerate(rr):
   i=int(row)-1;n=generated[i];begin,end=map(int,oo[j:j+2]);assert int(pp[j])==(0 if n==0 else 512+n-1);assert int(tt[j])==int(pp[j])+end-begin;assert pos[begin:end].tolist()==list(range(int(pp[j]),int(tt[j])))
   if n==0:assert end-begin==512 and ii[begin:end].tolist()==[i]*512
   else:assert end-begin==1 and ii[begin]==gold[i][n-1]
   ys[j]=gold[i][n];generated[i]+=1
  if logits:
   ll=arr(logits,B*4096,ctypes.c_uint16).reshape(B,4096);ll.fill(0)
   for j,v in enumerate(ys):ll[j,int(v)]=0x3f80
  arr(outflags,1,ctypes.c_int32)[0]=4 if fault else((1 if flags==3 else 0)+2)
  if N==16:
   for v in custom_values:v[0]+=47;v[1]+=16*47
  seen.append({'rows':rr.tolist(),'past':pp.tolist(),'N':N,'flags':flags});return 0
 lib.modern_kimi_engine_forward_r2475=F(native)
 custom=F(lambda device,out,n:None);report={'counter_dispatch':[],'local_forward_samples':[]};env={'lib':lib,'ctx':1,'control':Control(),'g':{'memory':lambda:[(5*1024**3,64*1024**3)]*8},'report':report,'node':2,'tail':True,'emitter':2,'vocab':4096,'custom':custom,'moe':47,'policy':1,'candidate':True,'counts':lambda:[[0]*4 for _ in range(8)],'custom_counts':lambda:[v.copy()for v in custom_values]};namespace['check']=lambda code:code==0 or (_ for _ in()).throw(ValueError(code));forward,ready=namespace['bind'](env);state={}
 try:
  run_interleaved(inputs=[[i]*512 for i in range(16)],eos={999999},forward=forward,clock=time.perf_counter,state=state,label='actualdriverCPU',limit=128,token_ready_clock=ready)
  assert not fault and state['generated_ids']==gold and all(n==128 for n in generated);assert len(report['counter_dispatch'])==len(seen)==142;assert sum(v['N']==16 for v in seen)>0
  results.append({'actual_protocol_forward':True,'full128_IDs_match':True,'actual_input_offsets_positions_globalrows_checked_every_CAPI':True,'actual_I192_hit_fallback_assertions_executed':True,'tail_logits_CPUargmax_and_flags_checked':True,'calls':len(seen),'initial_decode_N':seen[1]['N']})
 except AssertionError:
  assert fault;results.append({'injected_bad_tail_output_flags':True,'actual_driver_rejected':True,'completed':state.get('completed',False)})
p=r/'Kimi-Model-events-driver-CPU-contract-r8669zq.json';p.write_text(json.dumps({'CPU_only':True,'GPU':False,'actual_protocol_SHA256':hashlib.sha256((r/'kimi_Model_events_protocol_r8669zq.py').read_bytes()).hexdigest(),'tests':results,'scope':'Actualnestedforwardfromsealedcandidateprotocol+actualinterleavedconsumer; fakeCAPIvalidatesCPUmetadata/feedback/numerics-shapecontract, notModelgold/performance.'},indent=2)+'\n',newline='\n');print(json.dumps(results))
