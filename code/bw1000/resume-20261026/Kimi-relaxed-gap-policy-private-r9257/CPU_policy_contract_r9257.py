import sys,pathlib,itertools,math,json,hashlib
r=pathlib.Path(__file__).resolve().parents[1]
base=r/'Kimi-adaptive-Model-AB-private-r9225/runtime-A'
sys.path.insert(0,str(base));sys.path.insert(0,str(pathlib.Path(__file__).parent))
from batch_policy_r9128 import greedy,ReplayStateModel
from relaxed_gap_policy_r9257 import decide
checks=0
for K in range(3):
 for props in itertools.product(range(4),repeat=K):
  for top in itertools.product(range(4),repeat=K+1):
   for gaps in itertools.product([0.,.125,1.,float('nan')],repeat=K):
    for remaining in range(1,K+2):
     baseline=greedy(7,props,top,{3},remaining)
     off,meta=decide(7,props,top,None,0,{3},{2},remaining)
     assert off==baseline and not meta['enabled']
     d,m=decide(7,props,top,gaps,.25,{3},{2},remaining)
     assert d.commit_inputs==(7,)+d.emitted[:-1]
     state=ReplayStateModel([9,8]);state.verify((7,)+props);state.commit(d)
     assert state.shadow is None
     if not d.terminal:assert state.active==(9,8)+d.commit_inputs
     for i in m.get('relaxed_positions',[]):
      assert props[i]!=top[i] and props[i]not in{2,3}and top[i]not in{2,3}
      assert math.isfinite(gaps[i])and 0<=gaps[i]<=.25
     assert all(i < len(d.emitted) for i in m.get('relaxed_positions', []))
     assert not any(x==3 for x in d.emitted[:-1])
     assert len(d.emitted)<=remaining
     checks+=1
# Two near-greedy ordinary tokens can be accepted while EOS/control stays strict.
d,m=decide(7,[3,4],[5,6,8],[.125,.125],.25,{2},{1},8)
assert d.accepted==2 and d.emitted==(3,4,8)and d.commit_inputs==(7,3,4)and m['relaxed']==2
for bad in [True,-1,float('nan'),float('inf')]:
 try:decide(7,[],[0],[],bad)
 except ValueError:pass
 else:raise AssertionError('invalid epsilon accepted')
p=pathlib.Path(__file__).parent/'relaxed_gap_policy_r9257.py'
result={'passed':True,'scope':'CPU accepted-prefix/rollback/EOS/cap contract only; no actual logits, model runtime, quality or speed qualification','exhaustive_cases':checks,'policy_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'OFF_equal_original_greedy':True,'GPU':False,'native_changed':False,'threshold_choice_for_Model':None}
(pathlib.Path(__file__).parent/'CPU-policy-contract-r9257.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))
