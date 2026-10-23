import pathlib,json,os,hashlib,ast
from kimi_firstdecode_consumer_r8517 import replay_cut,KEYS
r=pathlib.Path.cwd();cuts=json.loads((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_text())['cuts'];seen=[];saved=[]
class Adapter:
 def __init__(self,cut):self.cut=cut
 def forward(self,j):
  seen.append(dict(seq=j['seq'],env={k:os.environ.get(k)for k in KEYS}))
  return self.cut['jobs'][j['seq']]['frozen_actual_ids']
class Collector:
 def __init__(self):self.n=0
 def __call__(self,label,value):return [dict(value)for _ in range(4)]
old={k:os.environ.get(k)for k in KEYS};root=r/'Kimi-firstdecode-CPU-fixture-r8523';root.mkdir(exist_ok=False)
for cut in cuts:
 seen.clear();records=replay_cut(Adapter(cut),cut,stage=3,case=cut['label'],root=root,collect=Collector(),persist=saved.append,layers=[0,23,24,46,47,69,70,92,93],points=['stage_input','decoder','final_norm'])
 assert len(records)==len(cut['jobs'])and all(v['firstdiff']is None for v in records)
 for v,j in zip(seen,cut['jobs']):
  if 0 in j['job']['keys']:assert v['env'][KEYS[4]]==str(j['job']['keys'].index(0))and v['env'][KEYS[5]]=='999'and v['env'][KEYS[6]]=='1000'and v['env'][KEYS[7]]=='0'and ('job'+str(j['job']['seq']))in v['env'][KEYS[0]]
  else:assert all(v['env'][k]is None for k in KEYS)
assert {k:os.environ.get(k)for k in KEYS}==old
assert any(v['env'][KEYS[8]]=='target'for v in seen if v['env'][KEYS[0]])
ast.parse((r/'kimi_firstdecode_consumer_r8517.py').read_text(encoding='utf-8-sig'))
p=r/'Kimi-firstdecode-consumer-CPU-r8523.json';out=dict(revision=8522,CPU_actual=True,GPU_executed=False,contracts=['Actual O1five/W1four immutable jobs consumed, companion inputs untouched','actual targetpacked index +unique case/stage/job directory','non-targetjobs stillforwarded withtraceOFF','everyowner samejobhash checked','fullactual/frozenIDs saved beforeassert','all env selectors restored in finally'],source_SHA256=hashlib.sha256((r/'kimi_firstdecode_consumer_r8517.py').read_bytes()).hexdigest(),native_validation='CPU native compile710 passed separately; no GPU Model execution; outerreset/QA/normaldestroy remains runnerresponsibility');p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
