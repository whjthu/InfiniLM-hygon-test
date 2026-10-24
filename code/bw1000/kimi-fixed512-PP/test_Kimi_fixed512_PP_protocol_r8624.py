"""Actual8607 Python control flow with explicit synthetic Engine fixtures; no GPU/performance evidence."""
import pathlib,json,types,sys,tempfile,hashlib,os,copy,importlib.util,time
r=pathlib.Path(__file__).resolve().parent
# Stub only the native Engine adapter; the actual consumer and sequence diff run.
MODE='INFINILM_KIMI_PREFILL512_R8562';ELIGIBLE='INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562'
class Fn:
 def __init__(self,fun):self.fun=fun
 def __call__(self,*a):return self.fun(*a)
class Lib:
 def __init__(self):
  self.modern_kimi_vocab_head_get_r8341=Fn(lambda:0)
  self.modern_kimi_engine_reset_r2475=Fn(lambda ctx:0)
  def count(d,v,n):
   for i in range(n):v[i]=0
   return 0
  self.modern_kimi_prefill512_counts_r8562=Fn(count);self.modern_kimi_CCL512_counts_r8603=Fn(count)
class Control:
 def collect(self,k,v):
  rows=[copy.deepcopy(v)for _ in range(4)]
  if 'oracle'in v:rows[3]['oracle']=dict(finite=True,native_equals_CPU=True)
  return rows
context={}
class Adapter:
 CPU_fixture=True
 def __init__(self,*a,full_logit_reference=False,**kw):self.stage=kw['stage'];self.state={};self.reference_observation=dict(finite=True,native_equals_CPU=True)
 def forward(self,job):
  out=[]
  for row,inp in zip(job['rows'],job['inputs']):
   if job['kind']=='prefill':
    key=json.dumps(inp);self.state[row]=[list(context['lookup'][key]),0,key]
   tokens,ordinal,key=self.state[row];token=tokens[ordinal]
   if context['kind']=='candidate-negative'and context['reports'][self.stage].get('phase')=='N-fixed512-PP-qualification'and ordinal==33 and len(json.loads(key))==512:token+=1
   if context['kind']=='baseline-negative'and os.environ[MODE]=='0'and ordinal==33 and len(json.loads(key))==512:token+=1
   out.append(token);self.state[row][1]+=1
  return out if self.stage==3 else None
 def close(self):pass
stub=types.ModuleType('kimi_prefill512_scope_r8563');stub.ScopedProvider=Adapter;stub.MODE=MODE;stub.ELIGIBLE=ELIGIBLE;sys.modules[stub.__name__]=stub
spec=importlib.util.spec_from_file_location('actual_protocol','kimi_fixed512_PP_protocol_r8624.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
gold=json.loads((r/'Kimi618-final-token-evidence-r8570.json').read_text());frozen=json.loads((r/'r4917-Kimi-two-geometry-frozen-gold.json').read_text());natural=next(v['actual_outputs']for v in gold['static']if v['label']=='O-reference-M1');eos=frozen['eos_token_ids'][0]

import threading
class Group:
 def __init__(self):self.lock=threading.Condition();self.data={}
 def collect(self,stage,index,label,payload):
  with self.lock:
   row=self.data.setdefault(index,dict(label=label,parts={}));assert row['label']==label,(index,row['label'],label)
   row['parts'][stage]=copy.deepcopy(payload);self.lock.notify_all()
   assert self.lock.wait_for(lambda:len(row['parts'])==4,timeout=20),(label,'CPU rendezvous')
   return[copy.deepcopy(row['parts'][i])for i in range(4)]
class ActualControl:
 def __init__(self,g,stage):self.g=g;self.stage=stage;self.i=0
 def set_timeout(self,n):pass
 def collect(self,label,payload):
  result=self.g.collect(self.stage,self.i,label,payload);self.i+=1;return result
results=[]
for kind in ['candidate-negative','full-pass']:
 with tempfile.TemporaryDirectory(prefix='Kimi8624-')as td:
  root=pathlib.Path(td);(root/'incoming').mkdir();(root/'results').mkdir()
  (root/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').write_text(json.dumps(frozen));(root/'incoming/Kimi618-final-token-evidence-r8570.json').write_bytes((r/'Kimi618-final-token-evidence-r8570.json').read_bytes())
  chat=dict(eos_token_ids=frozen['eos_token_ids'],cases={'sum':dict(input_ids=[5]),'multiply':dict(input_ids=[6])})
  known=dict(cases=[dict(task='sum',grouped=True,generated_ids=[[42,eos]]),dict(task='multiply',grouped=True,generated_ids=[[43,eos]])])
  (root/'results/kimi-chat-cases-r20.json').write_text(json.dumps(chat));(root/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').write_text(json.dumps(known))
  reports=[{}for _ in range(4)];context.update(kind=kind,reports=reports,lookup={json.dumps(frozen['natural_input_ids'][i]):natural[i]for i in range(2)});context['lookup'].update({json.dumps([5]):[42,eos],json.dumps([6]):[43,eos]})
  os.environ.update(INFINILM_DENSE_PREFILL_F32_R2475='1',INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562='0',INFINILM_KIMI_PREFILL512_R8562='0',INFINILM_KIMI_CCL512_R8603='0')
  group=Group();errors=[]
  def run_owner(node):
   g=dict(lib=Lib(),ctx=1,control=ActualControl(group,node),report=reports[node],node=node,r=root,tag=kind,tp_nodes=1,pp=4,check=lambda rc:None,memory=lambda:[(64*1024**3,64*1024**3)]*8,save=lambda:None)
   try:m.run(g)
   except BaseException as e:errors.append((node,repr(e)))
  workers=[threading.Thread(target=run_owner,args=(i,))for i in range(4)]
  for w in workers:w.start()
  for w in workers:w.join(90);assert not w.is_alive(),'CPU control deadlock'
  assert not errors,errors
  assert all(v['historical618_exact_O8']and v['serving_protocol_complete']for v in reports)
  if kind=='candidate-negative':assert all(not v['candidate_qualified']and not v.get('paired_timing')for v in reports)
  else:
   assert all(v['candidate_qualified']and len(v['paired_timing'])==8 for v in reports)
   assert all([v['prefill_collect_each']for v in report['paired_timing']]==[True,False,False,True]*2 for report in reports)
   assert all(v['CCL512']==v['projection512']==0 for report in reports for v in report['paired_timing'])
  assert all(v['exact']for report in reports for v in report['QA'])
  assert all(all(x==0 for x in counts)for report in reports for counts in report['final_CCL512_counts'])
  results.append(dict(fixture=kind,all4owners_complete=True,exact618_anchor=True,all4QA_exact=True,candidate_qualified=reports[0]['candidate_qualified'],ordinary_cases=len(reports[0].get('paired_timing',[])),native_math_flags_unchanged=True))
out=dict(CPU_only=True,GPU=False,native_numeric_evidence=False,scope='Actual8624 fullprotocol with4 CPU owners/synthetic Engine; complete128/EOS/matchedgold/candidatefail/ONNOtwice/restoreQA/defaultOFF control-flow proof only, no TPS measurement.',source_SHA256=hashlib.sha256((r/'kimi_fixed512_PP_protocol_r8624.py').read_bytes()).hexdigest(),helper_SHA256=hashlib.sha256((r/'kimi_fixed512_prefill_r8624.py').read_bytes()).hexdigest(),fixtures=results)
p=r/'Kimi-fixed512-PP-CPU-contract-r8624.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),newline='\n');print(json.dumps(out))
