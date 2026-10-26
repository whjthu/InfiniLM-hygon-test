import pathlib,json,hashlib,collections
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
run='GLM-arrival-r9284b-76cebdf004bd413f'
owners=[]; aligned=collections.defaultdict(list)
for stage in (0,1):
 for rank in range(8):
  p=r/'results'/f'glm-arrival-r9284b-{run}-stage{stage}-rank{rank}.json'
  assert p.stat().st_size<25*1024*1024
  raw=p.read_bytes();q=json.loads(raw)
  assert q['actual_producer']=='910376.755' and q['library_sha256']=='c6506898f7d718e7257253b51bf720ff8fc84ed733d9b7306edd989308d37514'
  assert q['completed'] and q['normal_reset_close'] and q['benchmark_complete']
  groups={}
  for a in q['episodes']:
   for c in a['native_calls']:
    e=q['native_events'][c['event_index']];B=len(c['rows']);T=e['input_tokens'];key=f"{a['label']}/{c['op']}/T{T}/B{B}"
    v=groups.setdefault(key,dict(calls=0,HOST_CAPI_seconds=0.,HOST_pipeline_sum=[0.,0.,0.]))
    v['calls']+=1;v['HOST_CAPI_seconds']+=e['native_seconds'];v['HOST_pipeline_sum']=[x+y for x,y in zip(v['HOST_pipeline_sum'],e['existing_pipeline_profile'])]
    aligned[(a['label'],c['seq'],c['op'],T,B)].append(dict(stage=stage,rank=rank,HOST_CAPI_seconds=e['native_seconds'],HOST_pipeline=e['existing_pipeline_profile']))
  owners.append(dict(stage=stage,rank=rank,path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),groups=groups))
summary=[]
for (label,seq,op,T,B),v in aligned.items():
 assert len(v)==16
 slow=max(v,key=lambda x:x['HOST_CAPI_seconds'])
 summary.append(dict(label=label,seq=seq,op=op,T=T,B=B,maximum_duration_owner=slow,
    stages=[dict(stage=s,min_HOST_CAPI_seconds=min(x['HOST_CAPI_seconds'] for x in v if x['stage']==s),max_HOST_CAPI_seconds=max(x['HOST_CAPI_seconds'] for x in v if x['stage']==s)) for s in (0,1)]))
print(json.dumps(dict(schema=1,scope='Exact completed755result scalar metadata only; no rawtokens/weights/activations/environment. HOST maximumduration owner is not a synchronized global critical path.',owners=owners,aligned_call_summary=summary),separators=(',',':')))
