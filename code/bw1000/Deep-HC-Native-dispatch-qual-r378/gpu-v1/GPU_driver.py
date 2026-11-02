"""a314 actual dispatch/lifetime qualification; 32 calls, no performance experiment."""
import pathlib,json,hashlib,os,time,ctypes,threading,traceback,numpy as np
from fixture import load
from affinity_protocol import rank_plan
from protocol import run_owners
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
P=ctypes.c_void_p;I=ctypes.c_int;U=ctypes.c_uint64
def configure(lib):
 lib.deep_hc_bridge_error_r378.restype=ctypes.c_char_p
 lib.deep_hc_bridge_create_r378.argtypes=[I,ctypes.c_double,ctypes.c_double,I,P,P,P];lib.deep_hc_bridge_create_r378.restype=P
 lib.deep_hc_bridge_forward_r378.argtypes=[P,ctypes.POINTER(P),P]
 lib.deep_hc_bridge_state_r378.argtypes=[P,ctypes.POINTER(P)]
 lib.deep_hc_bridge_destroy_r378.argtypes=[P,ctypes.POINTER(I)]
 lib.deep_hc_fullpre_enable_r377.argtypes=[I];lib.deep_hc_fullpre_stats_r377.argtypes=[ctypes.POINTER(U),I]
 lib.deep_hc_model_epilogue_set_r366.argtypes=[I];lib.deep_hc_model_epilogue_counters_r366.argtypes=[ctypes.POINTER(U),I]
 lib.deep_hc_canonical_calls_r7130.argtypes=[I,I];lib.deep_hc_canonical_calls_r7130.restype=U
def counters(lib):
 new=(U*80)();old=(U*56)();assert lib.deep_hc_fullpre_stats_r377(new,80)==lib.deep_hc_model_epilogue_counters_r366(old,56)==0
 return dict(new=np.array(new,dtype=np.uint64).reshape(8,10).astype(object).tolist(),old=np.array(old,dtype=np.uint64).reshape(8,7).astype(object).tolist(),canonical=[int(lib.deep_hc_canonical_calls_r7130(r,2))for r in range(8)])
class Owner:
 def __init__(self,rank,prepared,lib,torch):self.rank=rank;self.data,self.proof=prepared;self.lib=lib;self.torch=torch;self.ctx=None;self.q=dict(rank=rank,fixture=self.proof,calls=[])
 def check(self,rc):
  if rc:raise RuntimeError(self.lib.deep_hc_bridge_error_r378().decode())
 def setup(self):
  r=self.rank;t=self.torch;d=self.data;initial,assigned=rank_plan(r);self.initial=initial;self.assigned=assigned;t.cuda.set_device(r);props=t.cuda.get_device_properties(r);assert 'gfx936'in props.gcnArchName
  self.q.update(TID=threading.get_native_id(),initial_allowed=initial,assigned=assigned,device=r,hardware=dict(name=props.name,arch=props.gcnArchName),memory_before=list(t.cuda.mem_get_info()))
  self.ctx=self.lib.deep_hc_bridge_create_r378(r,d['rms_eps'],d['hc_eps'],d['iterations'],*[P(a.ctypes.data)for a in d['weights']]);assert self.ctx,self.lib.deep_hc_bridge_error_r378().decode()
  assert sorted(os.sched_getaffinity(0))in [initial,assigned]
  self.back=[t.empty(16384+64,device=f'cuda:{r}',dtype=t.bfloat16),t.empty(4096+64,device=f'cuda:{r}',dtype=t.bfloat16),t.empty(4+16,device=f'cuda:{r}',dtype=t.float32),t.empty(16+16,device=f'cuda:{r}',dtype=t.float32)]
  self.x=self.back[0][:16384].view(1,4,4096);self.x.copy_(t.from_numpy(d['input_bits']).view(t.bfloat16).to(f'cuda:{r}'));self.outputs=[self.back[1][:4096].view(1,4096),self.back[2][:4].view(1,4),self.back[3][:16].view(1,4,4)]
  self.tails=[self.back[0][16384:],self.back[1][4096:],self.back[2][4:],self.back[3][16:]]
  for a in self.tails:a.fill_(17.)
  self.ptrs=(P*4)(*[P(a.data_ptr())for a in [self.x,*self.outputs]]);self.q['state_initial']=self.state()
 def state(self):
  arrays=[np.empty_like(a)for a in self.data['weights']];ptrs=(P*3)(*[P(a.ctypes.data)for a in arrays]);self.check(self.lib.deep_hc_bridge_state_r378(self.ctx,ptrs));assert all(np.array_equal(a,b)for a,b in zip(arrays,self.data['weights']));return [hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
 def call(self,name):
  t=self.torch
  for a in self.outputs:a.fill_(float('nan'))
  self.check(self.lib.deep_hc_bridge_forward_r378(self.ctx,self.ptrs,P(t.cuda.current_stream().cuda_stream)))
  assert sorted(os.sched_getaffinity(0))==self.assigned
  actual=[self.outputs[0].cpu().view(t.uint16).numpy().reshape(4096).copy(),self.outputs[1].cpu().numpy().reshape(4).copy(),self.outputs[2].cpu().numpy().reshape(4,4).copy()];metrics=[]
  for key,a in zip(['collapsed_BF16','post','comb'],actual):
   b=self.data['expected'][key];av=(a.astype(np.uint32)<<16).view(np.float32)if a.dtype==np.uint16 else a;bv=(b.astype(np.uint32)<<16).view(np.float32)if b.dtype==np.uint16 else b;assert np.isfinite(av).all()and np.isfinite(bv).all();delta=av.astype(np.float64)-bv.astype(np.float64);nrmse=float(np.linalg.norm(delta)/(np.linalg.norm(bv.astype(np.float64))+1e-30));assert nrmse<(.005 if key=='collapsed_BF16'else 1e-4),(name,key,nrmse);metrics.append(dict(name=key,NRMSE=nrmse,max_abs=float(np.max(np.abs(delta))),exact_diagnostic=bool(np.array_equal(a,b))))
  assert all(bool(t.all(a==17.))for a in self.tails)and np.array_equal(self.x.cpu().view(t.uint16).numpy(),self.data['input_bits']);assert self.state()==self.q['state_initial']
  q=dict(phase=name,numerics=metrics,output_poison_overwritten_finite=True,canaries=True,state_unchanged=True,final_allowed=sorted(os.sched_getaffinity(0)));self.q['calls'].append(q);return q
 def close(self):
  if self.ctx:
   expired=I();rc=self.lib.deep_hc_bridge_destroy_r378(self.ctx,ctypes.byref(expired));self.ctx=None;self.q.update(normal_close=rc==0 and expired.value==1,Runtime_expired=expired.value,destroy_code=rc)
  else:self.q.update(normal_close=True,context_created=False)
  return self.q
def main():
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/f'Deep378-gpu-v1-{step}.json';assert not out.exists();q=dict(completed=False,passed=False,actual_step=step,Model_loaded=False,performance_experiment=False,planned_calls=32,actual_calls=0,processes=1,owner_threads=8,ports_required=[]);started=time.perf_counter()
 try:
  mf=S/'GPU_MANIFEST.json';assert sha(mf)==os.environ['DEEP378_GPU_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
  cp=R/'candidates/Deep-HC-Native-dispatch-qual-r378-cpu-v1/CPU-build.json';assert sha(cp)==m['CPU757_receipt_SHA256'];b=json.loads(cp.read_bytes());assert b['passed']and b['actual_step']=='910407.757'and not b['Native_dlopen']and not b['GPU_initialized'];assert sha(b['bridge_library'])==b['bridge_SHA256']==m['bridge_SHA256'];assert sha(b['Native_path'])==b['Native_SHA256']==m['Native_SHA256']
  for p,h in b['Native50_objects_SHA256'].items():assert sha(p)==h
  for p,v in b['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
  assert sha(b['existing_HIP_provider']['path'])==b['existing_HIP_provider']['SHA256']
  import torch
  torch.set_num_threads(1);assert torch.cuda.device_count()==8;initial=sorted(os.sched_getaffinity(0));assert len(initial)>=32;q['parent_initial_allowed']=initial
  prepared=[load(r)for r in range(8)];original=R/'candidates/Deep-HC-epilogue-r358-cpu-r2/CPU-fixtures.json';bp=original.parent/'CPU-build.json';assert sha(bp)=='0a963eafd1941104823a2567b9dfbffb900372be6a6479779104780a372170b1';oldbridge=json.loads(bp.read_bytes());f=json.loads(original.read_bytes());assert sha(original)==oldbridge['CPU_fixture_receipt_SHA256']
  for rank,(_,p)in enumerate(prepared):assert p['weights']==f['fixtures'][rank]['weights']and p['input_SHA256']==f['fixtures'][rank]['input_SHA256']
  lib=ctypes.CDLL(b['bridge_library']);configure(lib);assert lib.deep_hc_fullpre_enabled_r377()==0 and lib.deep_hc_model_epilogue_get_r366()==0 and lib.deep_hc_canonical_get_r7130()==0
  assert lib.deep_hc_model_epilogue_set_r366(2)==1 and lib.deep_hc_fullpre_enable_r377(2)==1;assert lib.deep_hc_fullpre_enabled_r377()==0 and lib.deep_hc_model_epilogue_get_r366()==0
  paths=sorted({l.split()[-1]for l in pathlib.Path('/proc/self/maps').read_text().splitlines()if l.split()and l.split()[-1].startswith('/')})
  assert str(pathlib.Path(b['Native_path']).resolve())in paths and str(pathlib.Path(b['bridge_library']).resolve())in paths
  assert not any('libDeep_HC' in pathlib.Path(p).name and pathlib.Path(p).name not in [pathlib.Path(b['Native_path']).name,pathlib.Path(b['bridge_library']).name]for p in paths)
  p724=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-build.json';assert sha(p724)=='e3012f3c27c58f3c2c18f237906cd0120a4232620873cbea9b90e5012fd61152';parent976=json.loads(p724.read_bytes())
  forbidden=[str(pathlib.Path(p).resolve())for p in [oldbridge['Native_path'],parent976['library']]];assert all(p not in paths for p in forbidden);q['old_afc6_976f_Native_not_mapped']=forbidden
  expected_providers={p:v['SHA256']for p,v in b['actual_DTK_bindings'].items()};expected_providers[b['existing_HIP_provider']['path']]=b['existing_HIP_provider']['SHA256']
  q['provider_maps']={p:dict(resolved=str(pathlib.Path(p).resolve()),SHA256=sha(p),mapped=str(pathlib.Path(p).resolve())in paths)for p in [b['Native_path'],b['bridge_library'],*expected_providers]};assert all(v['mapped']for p,v in q['provider_maps'].items()if 'libinfiniccl.so'not in p)
  def control(index,name):
   if index==0:assert lib.deep_hc_fullpre_enabled_r377()==0 and lib.deep_hc_model_epilogue_get_r366()==0
   elif index==1:assert lib.deep_hc_model_epilogue_set_r366(1)==0
   elif index==2:assert lib.deep_hc_fullpre_enable_r377(1)==0
   else:
    assert lib.deep_hc_fullpre_clear_r377()==1;assert lib.deep_hc_fullpre_enable_r377(0)==0;assert lib.deep_hc_fullpre_clear_r377()==0;assert lib.deep_hc_model_epilogue_set_r366(0)==0
   before=counters(lib);assert all(not any(row)for row in before['new']);return before
  def observe(index,name,before,calls):
   assert all(v and v['phase']==name for v in calls);after=counters(lib);old_delta=(np.array(after['old'],dtype=np.int64)-np.array(before['old'],dtype=np.int64)).tolist();new_delta=(np.array(after['new'],dtype=np.int64)-np.array(before['new'],dtype=np.int64)).tolist()
   assert old_delta==[[0,0,1,1,1 if index==1 else 0,0,0]for _ in range(8)],old_delta
   assert new_delta==[([1,1,1,1,0,0,0,0,0,0]if index==2 else [0]*10)for _ in range(8)],new_delta
   assert [a-b for a,b in zip(after['canonical'],before['canonical'])]==[1]*8
   q['actual_calls']+=8;return dict(phase=name,actual_calls=8,counters_before=before,counters_after=after,old_delta=old_delta,new_delta=new_delta,numerics= calls)
  result=run_owners(lambda rank:Owner(rank,prepared[rank],lib,torch),control,observe);q['protocol']=result;assert result['passed'],result['errors'];assert q['actual_calls']==32
  masks=[set(v['calls'][0]['final_allowed'])for v in result['closes']];assert all(len(v)==4 for v in masks)and all(not masks[i]&masks[j]for i in range(8)for j in range(i));assert len({v['TID']for v in result['closes']})==8 and all(v['Runtime_expired']==1 for v in result['closes'])
  assert sorted(os.sched_getaffinity(0))==initial;assert lib.deep_hc_fullpre_enabled_r377()==0 and lib.deep_hc_model_epilogue_get_r366()==0 and all(not any(row)for row in counters(lib)['new']);q.update(passed=True,bridge_SHA256=b['bridge_SHA256'],Native_SHA256=b['Native_SHA256'],source_manifest_SHA256=sha(mf),default_OFF_restored=True,no_ABBA=True,no_speedup_claim=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q.update(completed=True,whole_driver_wall_s=time.perf_counter()-started);out.write_text(json.dumps(q,indent=2)+'\n');assert q['passed'],q.get('error')
if __name__=='__main__':main()
