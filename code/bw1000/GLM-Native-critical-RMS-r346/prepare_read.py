import pathlib,json,hashlib
p=pathlib.Path(__file__).parent;repo=p.parents[2];a=json.loads((repo/'results/bw1000/2026-11-02/GLM308823-root-audited-W4-scoped-Model-gain-PASS-clean.json').read_text());pins={x['path']:x['SHA256'] for x in a['original_evidence']['owner_files']}
assert len(pins)==32
s='''import pathlib,json,hashlib,statistics,os
PINS=PIN_VALUE
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out={'read_only':True,'GPU_initialized':False,'owners':[],'source_paths':{}}
for name,expected in PINS.items():
 p=pathlib.Path(name);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
 q=json.loads(raw);rows=q.get('service_requests',q.get('requests',[]));o={'path':name,'SHA256':expected,'stage':q.get('stage'),'rank':q.get('rank'),'Native_SHA256':q.get('library_sha256'),'requests':[],'profile_top_keys':{k:(len(v) if isinstance(v,(list,dict)) else v) for k,v in q.items() if any(s in k.lower() for s in ['profile','trace','timing'])},'library_paths':q.get('actual_LM_native_libraries',{})}
 for row in rows:
  calls=row.get('native_calls',[]);pref=[x for x in calls if x.get('prefill')];dec=[x for x in calls if not x.get('prefill')]
  z={k:row.get(k) for k in ['pair_index','prefill_control_window','paired_warmup','Native_forward_host_summary']};z.update(native_call_count=len(calls),prefill_calls=pref,call_keys=sorted({k for x in calls for k in x}),decode_summary=None)
  if dec:
   vals=sorted(x['Native_CAPI_s'] for x in dec);z['decode_summary']={'n':len(vals),'sum_s':sum(vals),'min_s':vals[0],'p50_s':statistics.median(vals),'p95_s':vals[int(.95*(len(vals)-1))],'p99_s':vals[int(.99*(len(vals)-1))],'max_s':vals[-1]}
  o['requests'].append(z)
 out['owners'].append(o)
for rel in ['candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM','candidates/latest-family-native-r269/GLM','install/hygon-glm-5f91ad7-r3672/include','candidates/GLM-latest-native-r269','candidates/GLM-prefill-W2-W4-owner-r307']:
 p=r/rel;out['source_paths'][str(p)]={'exists':p.exists(),'symlink':p.is_symlink(),'target':os.readlink(p) if p.is_symlink() else None,'children':[x.name for x in p.iterdir()][:30] if p.is_dir() else []}
print(json.dumps(out))
'''.replace('PIN_VALUE',repr(pins));(p/'remote_read_owners.py').write_text(s)
print(json.dumps({'owner_count':len(pins),'audit_SHA256':hashlib.sha256((repo/'results/bw1000/2026-11-02/GLM308823-root-audited-W4-scoped-Model-gain-PASS-clean.json').read_bytes()).hexdigest()}))
