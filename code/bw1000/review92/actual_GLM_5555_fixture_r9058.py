import pathlib,subprocess,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=json.loads((i/'GLM-5555-fixture-manifest-r9058.json').read_bytes())
for n,h in m['sources_SHA256'].items():
 p=i/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if n.endswith('.py'):compile(p.read_bytes(),n,'exec')
assert subprocess.run(['bash','-n',str(i/'GLM_5555_fixture_r9058_inside.sh')]).returncode==0
exec(compile((i/'launch_GLM_5555_fixture_r9058.py').read_bytes(),'launch_GLM_5555_fixture_r9058.py','exec'))
