from pathlib import Path
import base64,zlib,json,hashlib,ast
r=Path(__file__).resolve().parent;b=r/'Kimi-crossover-Model-private-r9026'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obs=r/'Kimi-target-taps-private-r9022/fixture-GPU-loader-fixed-observed-a.json';t=json.loads(obs.read_bytes())['terminal'];assert t['qualified_fixture'] and t['post_pass'] and t['data']['normal_close'] and t['Slurm_row'][0]=='910374.988'
mp=b/'Kimi-crossover-manifest-r9026.json';m=json.loads(mp.read_bytes())
m.update(status='ONE_SOURCE_READY_DIAGNOSTIC_MODEL_GO',root_final_GO='Root authorizes exactly one immutable Math0 TP16PP2 B1 fixedA-E target diagnostic load after actualfixture988; no auto retry or speedclaim',runtime_fixture={'handle':'910374.988','post':'910374.989','receipt':'results/Kimi-taps-fixture-r9023-910374.988.json','local_observation_SHA256':sha(obs),'scope':'One owner weightless actualRuntime3diagAPI/callbackstate/tapbytes;not32ownerModelaudit'},GPU='NOT_SUBMITTED')
mp.write_bytes((json.dumps(m,indent=2)+'\n').encode())
files={n:(b/n).read_text() for n in m['runtime_files']};files[mp.name]=mp.read_text()
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
remote='''import pathlib,json,zlib,base64,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
assert not (r/'Kimi-crossover-TP16PP2-r9026-launch.json').exists(),'already launched; do not change sources'
for n,s in files.items():
 p=i/n
 if p.exists() and p.read_bytes()!=s.encode():
  assert n in {'Kimi_crossover_Model_r9026_inside.sh','launch_Kimi_crossover_Model_r9026.py','audit_Kimi_crossover_Model_r9026.py'},n
  backup=i/'Kimi-crossover-r9026-prelaunch-backups';backup.mkdir(exist_ok=True)
  prior=p.read_bytes();q=backup/(n+'.'+hashlib.sha256(prior).hexdigest())
  if q.exists():assert q.read_bytes()==prior
  else:q.write_bytes(prior)
 p.write_bytes(s.encode())
m=json.loads((i/'Kimi-crossover-manifest-r9026.json').read_text())
for n,h in m['source_sha256'].items():assert hashlib.sha256((i/n).read_bytes()).hexdigest()==h,n
f=json.loads((r/m['runtime_fixture']['receipt']).read_text());assert f['completed'] and f['fixture_only'] and f['normal_close'] and f['actual_metadata_pass'] and f['tap_bytes_pass'] and f['read_unchanged_write_changed']
for n in m['runtime_files']:
 if n.endswith('.py'):compile((i/n).read_text(),str(i/n),'exec')
 if n.endswith('.sh'):subprocess.run(['bash','-n',str(i/n)],check=True,capture_output=True)
print(json.dumps({'uploaded_and_syntax_pass':True,'Model_submitted':False,'manifest_SHA256':hashlib.sha256((i/'Kimi-crossover-manifest-r9026.json').read_bytes()).hexdigest(),'fixture':'910374.988'}))
'''.replace('PAYLOAD',repr(payload))
ast.parse(remote);(r/'upload_Kimi_crossover_Model_r9026.py').write_bytes(remote.encode());print(json.dumps({'stage_script':'upload_Kimi_crossover_Model_r9026.py','Model_GO':True,'manifest_SHA256':sha(mp)}))
