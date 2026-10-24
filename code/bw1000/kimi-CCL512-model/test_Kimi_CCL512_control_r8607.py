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
   for i in range(n):v[i]=1
   return 0
  self.modern_kimi_prefill512_counts_r8562=Fn(count);self.modern_kimi_CCL512_counts_r8603=Fn(count)
class Control:
 def collect(self,k,v):
  rows=[copy.deepcopy(v)for _ in range(4)]
  if 'oracle'in v:rows[3]['oracle']=dict(finite=True,native_equals_CPU=True)
  return rows
context={}
class Adapter:
 def __init__(self,*a,full_logit_reference=False,**kw):self.state={};self.reference_observation=dict(finite=True,native_equals_CPU=True)
 def forward(self,job):
  out=[]
  for row,inp in zip(job['rows'],job['inputs']):
   if job['kind']=='prefill':
    key=json.dumps(inp);self.state[row]=[list(context['lookup'][key]),0,key]
   tokens,ordinal,key=self.state[row];token=tokens[ordinal]
   if context['kind']=='candidate-negative'and os.environ[MODE]=='1'and ordinal==33 and len(json.loads(key))==512:token+=1
   if context['kind']=='baseline-negative'and os.environ[MODE]=='0'and ordinal==33 and len(json.loads(key))==512:token+=1
   out.append(token);self.state[row][1]+=1
  return out
 def close(self):pass
stub=types.ModuleType('kimi_prefill512_scope_r8563');stub.ScopedProvider=Adapter;stub.MODE=MODE;stub.ELIGIBLE=ELIGIBLE;sys.modules[stub.__name__]=stub
spec=importlib.util.spec_from_file_location('actual_protocol','kimi_CCL512_protocol_r8607.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
gold=json.loads((r/'Kimi618-final-token-evidence-r8570.json').read_text());frozen=json.loads((r/'r4917-Kimi-two-geometry-frozen-gold.json').read_text());natural=next(v['actual_outputs']for v in gold['static']if v['label']=='O-reference-M1');eos=frozen['eos_token_ids'][0]
results=[]
for kind in ['candidate-negative','full-pass','baseline-negative']:
 with tempfile.TemporaryDirectory(prefix='Kimi8607-')as td:
  root=pathlib.Path(td);(root/'incoming').mkdir();(root/'results').mkdir()
  (root/'incoming/r4917-Kimi-two-geometry-frozen-gold.json').write_text(json.dumps(frozen));(root/'incoming/Kimi618-final-token-evidence-r8570.json').write_bytes((r/'Kimi618-final-token-evidence-r8570.json').read_bytes())
  chat=dict(eos_token_ids=frozen['eos_token_ids'],cases={'sum':dict(input_ids=[5]),'multiply':dict(input_ids=[6])})
  known=dict(cases=[dict(task='sum',grouped=True,generated_ids=[[42,eos]]),dict(task='multiply',grouped=True,generated_ids=[[43,eos]])])
  (root/'results/kimi-chat-cases-r20.json').write_text(json.dumps(chat));(root/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').write_text(json.dumps(known))
  context.update(kind=kind,lookup={json.dumps(frozen['natural_input_ids'][i]):natural[i]for i in range(2)})
  context['lookup'].update({json.dumps([5]):[42,eos],json.dumps([6]):[43,eos]})
  os.environ.update(INFINILM_DENSE_PREFILL_F32_R2475='1',INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562='0',INFINILM_KIMI_PREFILL512_R8562='0',INFINILM_KIMI_CCL512_R8603='0')
  report={};g=dict(lib=Lib(),ctx=1,control=Control(),report=report,node=0,r=root,tag=kind,tp_nodes=1,pp=4,check=lambda rc:None,memory=lambda:[(64*1024**3,64*1024**3)]*8,save=lambda:None)
  error=None
  try:m.run(g)
  except AssertionError as exc:error=str(exc)
  if kind=='candidate-negative':
   assert not error and not report['candidate_qualified']and report['static_final_IDs_equal']is False and not report.get('paired_timing')and report['serving_protocol_complete']
   assert len(report['model_comparisons'][3]['actual_outputs'])==8
  elif kind=='full-pass':
   assert not error and report['candidate_qualified']and len(report['paired_timing'])==8
   assert [v['CCL512']for v in report['paired_timing']]==[0,1,1,0]*2
   assert all(v['decode_generated_tokens_including_EOS']==1016 for v in report['paired_timing'])
  else:assert error and not report['historical618_exact_O8']and len(report['model_comparisons'])==3
  # Numeric actuals must be durable even when later classification rejects.
  assert all(pathlib.Path(v['path']).exists()for v in report['model_comparisons'])
  results.append(dict(fixture=kind,assertion=error,completed=report.get('serving_protocol_complete',False),qualified=report.get('candidate_qualified'),timing_cases=len(report.get('paired_timing',[])),actual_files=len(report['model_comparisons'])))
out=dict(CPU_only=True,GPU=False,native_numeric_evidence=False,scope='Synthetic native adapter only, actual8607 protocol control flow/complete128/EOS/gold rejection/buffered ONNO twice exercised. No timing numbers are performance measurements.',source_SHA256=hashlib.sha256((r/'kimi_CCL512_protocol_r8607.py').read_bytes()).hexdigest(),fixtures=results)
p=r/'Kimi-CCL512-CPU-control-r8607.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),newline='\n');print(json.dumps(out))
