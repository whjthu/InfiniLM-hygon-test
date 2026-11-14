import hashlib,json,pathlib,subprocess,time
import sys
if len(sys.argv) != 3:
 raise SystemExit("provide frozen v2 and v1 reader paths")
a,b=(pathlib.Path(path).read_bytes() for path in sys.argv[1:])
assert hashlib.sha256(a).hexdigest()=='f054e9986adcd4439f8900ae327a0d586a066ea400722f097357bf50eaf8c8cf'
outer={'__name__':'original_v2'}
exec(compile(a,'v2_reader.py','exec'),outer)
assert hashlib.sha256(b).hexdigest()==outer['READER_SHA256'] and b.count(outer['V1_CLAIM'].encode())==1
reader={'__name__':'original_v1'}
exec(compile(b.replace(outer['V1_CLAIM'].encode(),outer['CLAIM'].encode(),1),'v1_reader.py','exec'),reader)
reader.update(RUN=outer['RUN'],LABEL=outer['LABEL'])
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
label=outer['LABEL']
rows=subprocess.check_output(['sacct','-n','-P','-j',reader['PARENT'],'--format='+reader['COLUMNS']],text=True,timeout=12)
observed=time.time()
rows=[x.split('|') for x in rows.splitlines()]
indexed=reader['accounting'](rows)
assert indexed[label+'-single-drain-gate'][0:3]==['1103606.81',label+'-single-drain-gate','RUNNING']
src=root/(label+'-dispatch')/'src'
binding,binding_sha=reader['read'](src/'INPUT_BINDING.json')
assert binding_sha=='81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e'
plan,plan_sha=reader['read'](pathlib.Path(binding['private_plan']['path']))
assert plan_sha==binding['private_plan']['SHA256']
scorer=(src/'scoring.py').read_bytes()
assert hashlib.sha256(scorer).hexdigest()==reader['SCORER_SHA256']
namespace={}
exec(compile(scorer,'frozen scorer','exec'),namespace)
cases={r['ordinal']:r['case'] for r in plan['requests']}
def grade(ordinal,text,natural,cap):
 return namespace['grade'](cases[ordinal],text,natural_EOS=natural,cap_hit=cap,unexpected_tool=False)
pins=binding['private_plan']['requests']
expected={pin['ordinal']:pin for pin in pins}
eos=set(binding['eos_token_ids']);special=set(binding['tokenizer']['special_ids'])
def read_arm(arm,step):
 indexed_by_rank=[]
 costs_by_rank=[]
 sha=[]
 for rank in range(16):
  path=root/(label+'-'+arm)/f'owner-{rank:02d}.json'
  with path.open('rb') as f: raw=f.read(96*1024**2+1)
  assert len(raw)<=96*1024**2
  v=json.loads(raw)
  assert v['global_rank']==rank and v['actual_GPU']==step and v['Native_arm']==arm and not v.get('error')
  rows={}
  costs={}
  for r in v['requests']:
   if r['diagnostic'] or not(r['completed'] and r['reset_after']):continue
   key=r['ordinal'];assert key in expected and key not in rows
   actual=reader['audit_request'](dict(r, score=r.get('score',dict(correct=r['natural_EOS'],answer_matches_gold=True,parse_failure=False,unexpected_tool_intent=False))),expected[key],eos,special,grade if rank==0 else None)
   rows[key]=actual
   expected_calls=len(r['generated_token_ids'])-1
   cost=r['pipeline_decode']
   profile_names=['incoming_host_s','layers_host_s','outgoing_host_s','incoming_stream_s','layers_stream_s','outgoing_stream_s']
   assert expected_calls>0 and len(r['decode_CAPI_s'])==cost['count']==expected_calls
   assert r['pipeline_profile_names']==profile_names and len(cost['sum_s'])==6
   import math
   assert all(math.isfinite(x) and x>=0 for x in [*r['decode_CAPI_s'],*cost['sum_s']])
   costs[key]=dict(calls=expected_calls, per_call_s=[sum(r['decode_CAPI_s'])/expected_calls,
                     *[x/expected_calls for x in cost['sum_s']]])
  costs_by_rank.append(costs)
  indexed_by_rank.append(rows);sha.append(hashlib.sha256(raw).hexdigest())
 common=set.intersection(*(set(r) for r in indexed_by_rank))
 verified={k:indexed_by_rank[0][k] for k in common
           if len({r[k]['output_hash'] for r in indexed_by_rank})==1}
 assert len(verified)==len(common)
 return verified,[len(r) for r in indexed_by_rank],sha,costs_by_rank
a,ar,asha,ac=read_arm('baseline','1103606.78')
b,br,bsha,bc=read_arm('single-drain','1103606.81')
assert len(a)==32
import statistics
assert len(b)>=19
keys=[pin['ordinal'] for pin in pins if pin['ordinal'] in b][:19]
assert len(keys)==19 and all(a[k]['output_hash']==b[k]['output_hash'] for k in keys)
metric_names=['forward_enclosure','incoming_host','layers_host','outgoing_host','incoming_stream','layers_stream','outgoing_stream']
def describe(rows):
 return dict(min=min(rows),median=statistics.median(rows),max=max(rows))
details={}
for i,name in enumerate(metric_names):
 l=[rank[k]['per_call_s'][i]*1000 for rank in ac for k in keys]
 r=[rank[k]['per_call_s'][i]*1000 for rank in bc for k in keys]
 details[name]=dict(baseline_ms=describe(l),candidate_ms=describe(r),
  candidate_minus_baseline_ms=describe([b-a for a,b in zip(l,r)]),
  ratio=describe([b/a for a,b in zip(l,r) if a>0]) if any(l) else None)
print(json.dumps(dict(observed=observed,candidate_accounting=indexed[label+'-single-drain-gate'],
 paired_requests=len(keys),paired_rank_request_cells=len(keys)*16,ordinals=keys,
 matched_output_hashes=len(keys),baseline_forwards=sum(rank[k]['calls'] for rank in ac for k in keys),
 candidate_forwards=sum(rank[k]['calls'] for rank in bc for k in keys),
 profile=details,baseline_SHA256=asha,candidate_SHA256=bsha),allow_nan=False))
