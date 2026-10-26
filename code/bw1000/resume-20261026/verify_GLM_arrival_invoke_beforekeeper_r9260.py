import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'incoming/GLM-arrival-Model-r9260';cp=json.loads((p/'CPU-result.json').read_bytes());q={'receipts':{},'known_runtime_steps':[]}
for n in ['a02r4n12','a02r4n13']:
 b=(p/('CPU-result-'+n+'.json')).read_bytes();q['receipts'][n]={'rawSHA256':hashlib.sha256(b).hexdigest(),'matches_qualified_CPU_semantics':json.loads(b)==cp['node_receipts'][n]};assert q['receipts'][n]['matches_qualified_CPU_semantics']
rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%100,State%40,ExitCode'],text=True,timeout=30).splitlines()];q['known_runtime_steps']=[v for v in rows if v[1]in ['GLM-arrival-Model-r9260','GLM-arrival-keeper-r9260']];assert not q['known_runtime_steps'];assert not (r/'GLM-arrival-Model-r9260-launch.json').exists();print(json.dumps(q))
