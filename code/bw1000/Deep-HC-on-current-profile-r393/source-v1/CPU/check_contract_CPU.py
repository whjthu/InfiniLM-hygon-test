"""Actual sparse wrapper/HC consumer with fake C ABI leaves, no Native or weights."""
from pathlib import Path
import sys,tempfile,json,types
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'owner'))
from profile_contract import SparseObserver
from chunk_contract import descriptor_for_index,HC_MODES
from hc_model_contract import HCModel
from CPU_fake_hc import fake_lib
from CPU_profile_fakes import library,CFG
def combined():
 lib=fake_lib(lambda ctx:0,'success');p=library(lambda ctx:0)
 for k,v in vars(p).items():
  if k!='modern_native_engine_reset':setattr(lib,k,v)
 return lib
def check(v):
 if v:raise RuntimeError('Fake nonzeroCAPI')
checks=[]
with tempfile.TemporaryDirectory()as temporary:
 root=Path(temporary);lib=combined();hc=HCModel(lib,43);obs=SparseObserver(lib,check,CFG,root)
 b=dict(rows=1,capacity=32768,prefill_chunks_supported=[1024,2048],hc_epilogue_modes_supported=[0,1]);calls=[]
 for i in range(6):
  hc.select(1);meta=dict(admitted_index=i,prefill_binding=descriptor_for_index(b,i),backend_token_monotonic=[],Native_forward_host_summary={'decode_calls':0})
  for t,past in [(min(1024,8489-p),p)for p in range(0,8489,1024)]+[(1,8489+j)for j in range(97)]:
   before=hc.snapshot()
   def forward(*args):lib.advance(len(args[0]));calls.append((i,past));return (0,None,[7])
   obs.forward(forward,([7]*t,list(range(past,past+t)),[0],[past],[past+t],[0,t]),meta,lambda:None)
   hc.verify_forward(before,hc.snapshot(),t,1)
   if past>=8489:meta['Native_forward_host_summary']['decode_calls']+=1
   meta['backend_token_monotonic']=[1.0]if past+t>=8489 else[]
  obs.complete(meta)
  assert [v['geometry']['decode_call_1based']for v in meta.get('owner_profiles',[])]==([8,87]if i in [3,4]else[])
 assert len(calls)==636 and lib.CPU_state=={'owner':0,'inner':0,'arm1':8}and HC_MODES==(1,)*6
 checks.append(dict(case='actual_sparse_wrapper_HC1_636forwards_4frames',passed=True,owner_rows_each=344,inner_rows_each=344,all86HC_each8owner=True))
 for scenario in ['arm_error','read_error','disarm_error']:
  where=root/scenario;where.mkdir();lib=combined();obs=SparseObserver(lib,check,CFG,where);count=[0]
  if scenario=='arm_error':
   original=lib.deep_owner_arm_r8900;lib.deep_owner_arm_r8900=lambda v:1 if v else original(v)
  else:
   def badread():raise RuntimeError('Fake read throw')
   lib.deep_attention_inner_read_r9193=badread
   if scenario=='disarm_error':lib.deep_owner_arm_r8900=lambda v:1 if v==0 else 0
  meta=dict(admitted_index=3,prefill_binding=descriptor_for_index(b,3),backend_token_monotonic=[1.0],Native_forward_host_summary={'decode_calls':7})
  def forward(*args):count[0]+=1;return(0,None,[7])
  try:obs.forward(forward,([7],[8496],[0],[8496],[8497],[0,1]),meta,lambda:None)
  except BaseException as exc:
   assert meta['profile_failures']and meta['profile_failures'][-1]['wrapper_exception_windows_closed']==(scenario!='disarm_error')
   assert count[0]==(0 if scenario=='arm_error'else 1)
  else:raise AssertionError('Profile fault accepted')
  checks.append(dict(case=scenario,passed=True,forward_calls=count[0],exception_stops_caller=True))
print(json.dumps(dict(passed=True,checks=checks,GPU=False,Slurm=False,Native_dlopen=False,weights=False,new_compile=False,actual_new_wrapper_and_HC_consumer=True,fake_CABI_leaves_only=True)))
