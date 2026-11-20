import datetime,hashlib,json,pathlib,statistics,subprocess,sys
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/'fleet-rccl-8rank-socket-shca-64-576-v1-9d72b8e5d68047838eaf13ffcec0b9be'
req='9d72b8e5d68047838eaf13ffcec0b9be'
manifest=(root/'SOURCE_FILES.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest()=='47081656795b4b0289fe1e82ffca1eb4e69ce0784a82bbac18a32c579f6c3168'
pins=json.loads(manifest)['files']
for name in ('audit.py','PLACEMENT.json','CLAIM.json','PROVIDERS.json'):
 raw=(root/name).read_bytes()
 assert pins[name]=={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}
sys.path.insert(0,str(root))
from audit import verify
plans=json.loads((root/'PLACEMENT.json').read_bytes())
terminal={}
cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
for j in ('1087656','1103606'):
 original=subprocess.check_output(['sacct','-nP','-j',j,'--format='+cols],text=True,timeout=9)
 terminal[j]=[v.split('|') for v in original.splitlines() if req in v]
status={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request':req}
for stage in ('sample','wide'):
 earned=root/f'{stage}-review.json'
 if not earned.is_file():
  status[stage]={'state':'LIVE_OR_UNQUALIFIED','slurm':{k:[r for r in v if f'-{stage}-' in r[1]] for k,v in terminal.items()}}
  continue
 record=earned.read_bytes()
 reported=json.loads(record)
 actual={}
 for j,pair in plans[stage].items():
  r=verify(root/f'{stage}-{j}',pair,req)
  assert r==reported[j]['result']
  rows=[r['independent_pre'],*[arm['row'] for _,arm in sorted(r['accounting'].items())],r['independent_post']]
  assert [x for x in terminal[j] if f'-{stage}-' in x[1]]==rows
  actual[j]=r
 pairs=[v for r in actual.values() for v in r['pairs']]
 summary={'all_planned_pairs':sum(len(x) for x in plans[stage].values()),'qualified_pairs':len(pairs)}
 if pairs:
  fields=('socket_ms','shca_ms','socket_over_shca')
  summary['medians']={k:statistics.median(v[k] for v in pairs) for k in fields}
  summary['minima']={k:min(v[k] for v in pairs) for k in fields}
  summary['maxima']={k:max(v[k] for v in pairs) for k in fields}
 status[stage]=dict(state='TERMINAL_READ_AND_RECONCILED',review_SHA256=hashlib.sha256(record).hexdigest(),
  parents=actual,summary=summary,verified_independent_post=True)
print(json.dumps(status,allow_nan=False))
