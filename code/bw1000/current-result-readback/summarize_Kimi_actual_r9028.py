import pathlib,json,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run='TP16PP2-I1928669zh-8e319d252bb323d5'
d={'hardware':'BW1000','level':'ACTUAL_TARGET_CROSSOVER_AND_OFFLINE_DRAFT','status':'TERMINAL_READBACK','owners':[],'scope':'No qualified speculative serving gain; raw activations remain remote'}
for n in range(4):
 p=r/'results'/f'kimi-crossover-r9026-{run}-node{n}.json';q=json.loads(p.read_text());z={k:q.get(k) for k in ['node','step','completed','error','shutdown_error','all8_owned_workers_shutdown_completed','loading_seconds','library_sha256','crossover_complete','crossover_all_IDs_equal','performance_claim','allposition_finalIDs_EOS_pass','untested','crossover_arms']};z.update(report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),calls=len(q['crossover_calls']))
 if n==2:
  z['head_observations']=[{'arm':c['arm'],'head':o['head']} for c in q['crossover_calls'] for o in c['owners'] if o['node']==2 and o['head']]
  z['actual_forward_diagnostic_seconds_by_arm']={a:sum(c['diagnostic_maxhost_seconds'] for c in q['crossover_calls'] if c['arm']==a and c['phase']=='decode') for a in 'ABCDE'}
 d['owners'].append(z)
for name in ['Kimi-crossover-TP16PP2-r9026-watch.json','dflash2-actual-A-r9027-watch.json']:
 d[name]=json.loads((r/name).read_text())
p=r/'results/dflash2-actual-A-r9027-910374.997.json';d['actual_draft']=json.loads(p.read_text());d['actual_draft_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'results'/f'Kimi-crossover-TP16PP2-r9026-audit-{run}.json';d['diagnostic_audit']=json.loads(p.read_text());d['diagnostic_audit_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
# Retrieve only timings/metadata from original untraced675; never activation vectors.
oldwatch=json.loads((r/'Kimi-TP16PP2-I192-r9007-launch.json').read_text());p=r/'results'/f"kimi-pair-Model-r9007-{oldwatch['run']}-node2.json";old=json.loads(p.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()=='b7a6fad8d9e8b8c293c0efdc5a4f9619652df099bb9d71e59701a80426f4ceb6'
d['old675_untraced_report_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();d['old675_phase_calls']=[v for v in old.get('paired_phase_calls',[]) if v.get('phase')=='verify']
print(json.dumps(d))
