from pathlib import Path
import ast
D=Path(__file__).resolve().parent;L=D.parents[3];source=(D/'audit_cost.py').read_text();tree=ast.parse(source)
functions='\n\n'.join(ast.get_source_segment(source,n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['percentile','emitter_metrics'])
script="from pathlib import Path\nimport json,hashlib,datetime\n"+functions+"\n"+'''R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='GLM-C2M1-r355-a';p=R/'results'/('GLM-r355-'+run+'-stage3-rank0.json');raw=p.read_bytes();q=json.loads(raw)
rows=[]
for pair in q['pairs']:
 sums={}
 for row in range(2):
  calls=[c for c in pair['calls']if c['row']==row]
  sums[str(row)]={'prefill_calls':sum(c['prefill']for c in calls),'prefill_CAPI_s':sum(c['Native_CAPI_s']for c in calls if c['prefill']),'decode_calls':sum(not c['prefill']for c in calls),'decode_CAPI_s':sum(c['Native_CAPI_s']for c in calls if not c['prefill'])}
 rows.append({'index':pair['index'],'enabled':pair['enabled'],'warmup':pair['warmup'],'complete_schedule_wall_s':pair['complete_schedule_wall_s'],'decode_schedule_wall_s':pair['decode_schedule_wall_s'],'emitter_metrics':emitter_metrics(pair),'local_tail_CAPI_enclosures':sums,'scope_warning':'CAPI spans contain recv/stream/collective wait and do not isolate GPU work; warmups are not a paired performance conclusion'})
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'source_report':str(p),'source_SHA256':hashlib.sha256(raw).hexdigest(),'completed_pairs':rows,'completed':q['completed']}))
'''
(L/'remote355-pair-cost.py').write_text(script,encoding='utf8')
