import pathlib,json,hashlib,statistics
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');launch=json.loads((r/'GLM-M15-Model-r129-launch.json').read_bytes());run=launch['run'];out={'scope':'Existing764 actual owner0 perstage HOST CAPI shape attribution; no newGPU/profile; PP stage walls overlap and MUST NOT add across stages','stages':[]}
for stage in [0,1]:
 p=r/'results'/('glm-arrival-r129-'+run+'-stage'+str(stage)+'-rank0.json');q=json.loads(p.read_bytes());assert q['completed'];row={'stage':stage,'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'episodes':[]}
 for a in q['episodes']:
  groups={}
  for c in a['native_calls']:
   e=q['native_events'][c['event_index']];key=(c['op'],e['input_tokens'],len(c['rows']));groups.setdefault(key,[]).append(e['native_seconds'])
  row['episodes'].append({'label':a['label'],'on':a['m15_oproj'],'shapes':[{'op':op,'T':t,'B':b,'calls':len(v),'HOST_CAPI_sum_s':sum(v),'HOST_CAPI_median_ms':1000*statistics.median(v)}for(op,t,b),v in sorted(groups.items())]})
 out['stages'].append(row)
print(json.dumps(out))
