"""Actual Python payload/dispatch glue, recording only its Native leaf. No GPU."""
from pathlib import Path
import ast,collections,copy,json,sys,time
from plan import records,row_order
from GPU_run import build_payload,dispatch
from CPU_build import dependency_paths
D=Path(__file__).resolve().parent
def main():
 cases=[]
 def passed(name):cases.append(dict(name=name,passed=True))
 plan=list(records());assert len(plan)==102 and sum(not p['warm']for p in plan)==96
 assert all(collections.Counter(r['strategy']for r in plan if not r['warm']and r['fixture']==fixture and r['block']==block)=={0:4,1:4,2:4}for fixture in range(2)for block in range(4));passed('balanced_fixed_three_arm_two_fixture_plan')
 a,b=row_order(0),row_order(1);assert len(a)==len(b)==512 and a!=b and collections.Counter(a)==collections.Counter(b)=={i:32 for i in range(16)};passed('same_global_rows_distinct_frozen_order')
 weights=[dict(name='gate.weight',pointer=123456,shape=[896,7168],dtype=1)]
 p=build_payload({'model_type':'kimi_k3'},weights,[1111,2222],0,'12.1.2.1',48091,'/refactor/results/recording391',time.time()+60);calls=[]
 def invoke(raw):calls.append(json.loads(raw));return 0
 answer=dispatch(p,invoke,lambda:json.dumps(dict(completed=True,all8normal_before_communicators=True)).encode());assert answer['completed']and calls==[p];passed('actual_dispatch_encodes_complete_payload_once_recording_leaf')
 for name,change in [('badnode',{'node':2}),('badpointers',{'input_pointers':[0,22]}),('expired',{'deadline':time.time()-1}),('relativeoutput',{'raw':'relative'}),('duplicateweight',{'weights':weights*2}),('badweightdtype',{'weights':[weights[0]|{'dtype':9}]})]:
  kw=dict(config={},weights=weights,input_pointers=[111,222],node=1,host='12.1.2.2',port=48091,raw='/refactor/results/recording391',deadline=time.time()+60);kw.update(change)
  try:build_payload(**kw)
  except(ValueError,KeyError):passed(name+'_before_Native_leaf')
  else:raise AssertionError(name)
 for rc,body in [(1,dict(completed=False,error='recording Native failure')),(0,dict(completed=True,all8normal_before_communicators=False))]:
  hit=[]
  try:dispatch(p,lambda b:(hit.append(b)or rc),lambda:json.dumps(body))
  except RuntimeError:assert len(hit)==1;passed('failure_no_retry_'+str(rc))
  else:raise AssertionError('failure swallowed')
 ldd=(D/'original782-ldd-r.log').read_text();paths=dependency_paths(ldd);assert '/refactor/candidates/Kimi-append512-r247/libKimi_append512_r247.so'in paths and '/refactor/candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so'in paths;assert any(p.endswith('libinfiniops.so')for p in paths);passed('actual782_direct_and_arrow_ldd_paths');
 source=(D/'adapter.cpp').read_text();assert source.count('module->forward(r,')==2 and 'modern_kimi_set_grouped_prefill(1)'in source and 'modern_kimi_set_tiled_w13_r1861(0)'in source
 assert 'held.clear();std::vector<rt::Tensor>().swap(held);r.syncStream();double wall'in source
 assert 'output->narrow({{1,start,64}})->copy_from(r,y)'in source and 'r.syncStream();double wall'in source and 'hd[0]==uint64_t(arm==0)'in source
 assert 'std::make_unique<infinilm::layers::moe::TopKRouter>'in source and 'weights.at("gate."+k)'in source;passed('production_owner_TLS_and_public_observer_source_boundary')
 for name in ['GPU_run.py','plan.py','check_CPU.py','CPU_build.py','invoke_CPU_once.py']:ast.parse((D/name).read_bytes(),filename=name)
 assert 'torch'not in sys.modules;passed('CPU_import_no_torch_Native_dlopen_or_Runtime')
 out=dict(completed=True,passed=True,CPU_only=True,GPU=0,Native_dlopen=False,Runtime=False,weight_payload=False,recording_leaf=True,cases=cases,actual_CAPI_payload_scope='Realbuild_payload+dispatch through recording CAPI leaf only; no Native numerical/path/owner claim',planned_warm=6,planned_formal=96)
 print(json.dumps(out));return out
if __name__=='__main__':main()
