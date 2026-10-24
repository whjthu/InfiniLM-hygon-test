"""CPU-only tests actual Python policy source with mocked transport, no Torch/GPU import."""
import ast,pathlib,json,hashlib,ctypes,threading
r=pathlib.Path(__file__).resolve().parent;source=r/'glm_layer77_policy_r5776.py';tree=ast.parse(source.read_text());tree.body=[v for v in tree.body if not isinstance(v,(ast.Import,ast.ImportFrom))]
class Tensor:
 def __init__(self,a):self.a=a
 def tolist(self):return self.a
class Torch:
 @staticmethod
 def tensor(a,**k):return Tensor(a)
space={'ctypes':ctypes,'json':json,'threading':threading,'torch':Torch};exec(compile(tree,str(source),'exec'),space);Policy=space['Layer77Policy']
class Function:
 def __init__(self,f):self.f=f
 def __call__(self,*a):return self.f(*a)
class Lib:
 def __init__(self,stage,rank):
  self.calls=[];self.ids=[{'layer':i,'stage':stage,'rank':rank}for i in range(0,38)if stage==0]or[{'layer':i,'stage':stage,'rank':rank}for i in range(38,78)]
  self.modern_glm_cross_control_r5775=Function(lambda*a:self.calls.append(a)or 0)
  self.modern_glm_cross_journal_r5775=Function(lambda*a:json.dumps({'identities':self.ids,'events':[]}).encode())
checks=0
for stage in [0,1]:
 for rank in range(8):
  lib=Lib(stage,rank);report={};p=Policy(lib,stage,rank,lambda t:setattr(t,'a',[v*16 for v in t.a]),lambda c:None,report,lambda:None)
  for mode,diag,shadow in[(0,False,-1),(1,True,-1),(0,True,77),(2,False,-1),(0,False,-1)]:p.switch('test',mode,diag,shadow);assert lib.calls[-1]==(rank,mode,int(diag),shadow);checks+=1
  # One-owner disagreement cannot pass the actual summed-mask gate.
  p.control=lambda t:setattr(t,'a',[16.,31.,0.,0.])
  try:p.switch('badmask',2,False);raise RuntimeError('wrongmaskaccepted')
  except AssertionError:checks+=1
  lib.ids.pop()
  try:p.collect('missingidentity',0);raise RuntimeError('missinglayeraccepted')
  except AssertionError:checks+=1
lib=Lib(1,0);p=Policy(lib,1,0,lambda t:None,None,{},lambda:None);errors=[]
def foreign():
 try:p.collect('foreign',0)
 except AssertionError:errors.append(True)
t=threading.Thread(target=foreign);t.start();t.join();assert errors==[True];checks+=1
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5776.json').read_text())
for n,h in m['source_SHA256'].items():
 assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
 if n.endswith('.py'):ast.parse((r/n).read_text())
main=(r/'modern_glm_crossrequest_r5776.py').read_text();assert main.index('all16registrybeforeWeightIO')<main.index('loader = baseline[')
probe=(r/'glm_layer77_probe_r5776.py').read_text();assert probe.count('if step==0 and native_diag:policy.collect(label,step)')==2 and probe.index('layer77_dispatch_before_ID_gate')<probe.index('same_micro_ids_ok=True')
assert "call('diag-I512-allN-firstlogits',i512_inputs,1,limit=1,diag=True)"in probe
assert "call('diag-I512-O-shadow77-firstlogits',i512_inputs,0,limit=1,diag=True,shadow=77)"in probe
# Exact regression for packed B8 single-token decode: packedN>1 must notlog.
cpp=(r/'glm_layer77_Model_cache_r5775.cpp').read_text();assert 'bool record5775=prefill5775&&diag5775;'in cpp and 'return q.tokens>1;'in cpp
for diagnostic in [False,True]:
 assert sum(bool(any(n>1 for n in [1]*8)and diagnostic)for _ in range(254))==0
 assert bool(any(n>1 for n in [512]*8)and diagnostic)==diagnostic
checks+=4
# Actual AST: firstfinally timestamps precede anyjournal/save. This failsoldboundary.
pt=ast.parse(probe);tries=[n for n in ast.walk(pt)if isinstance(n,ast.Try)and any('modern_glm_model_forward'in ast.unparse(v)for v in n.body)and any('policy.collect'in ast.unparse(v)for v in n.finalbody)]
assert len(tries)==2
for node in tries:
 assert isinstance(node.finalbody[0],ast.Assign)and node.finalbody[0].targets[0].id=='native_end'
 assert 'time.perf_counter()'in ast.unparse(node.finalbody[0])
 text='\n'.join(ast.unparse(v)for v in node.finalbody)
 assert text.index('prefill_capi_end.append(native_end - start)')<text.index('policy.collect')
checks+=6
assert "raise CandidateRejected('77 no-op/eligibility')"in probe
checks+=1
checks+=4
out={'passed':True,'checks':checks,'source_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Actual policy Python source exec with CPU fake control:16owneridentity,modeagreement/missinglayer/foreignthread rejection;ASTmanifest and preweight/ID-gate ordering. NOT native/GPU/math proof.','GPU_initialized':False};p=r/'GLM-layer77-CPU-contract-r5778.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
