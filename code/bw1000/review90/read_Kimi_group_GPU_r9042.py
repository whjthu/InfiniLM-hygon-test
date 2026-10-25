import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');tag='Kimi-layer0-real-r9042';p=r/'results'/(tag+'-terminal.json');q={'sacct':[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910375','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if 'r9042'in x],'watch':json.loads(p.read_text())if p.exists()else None,'reports':[],'logs':{str(p):p.read_text()[-5000:]for p in(r/'logs').glob(tag+'-*')if p.is_file()}}
for p in(r/'results').glob(tag+'-910375.*-node*.json'):q['reports'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())})
print(json.dumps(q))
