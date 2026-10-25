import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');tag='Kimi-norm-real-r9037';p=r/'results'/(tag+'-terminal.json');q={'sacct':[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if 'r9037'in x],'watch':json.loads(p.read_text())if p.exists()else None}
for p in(r/'results').glob(tag+'-910374.*.json'):q['original_receipt']={'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())}
print(json.dumps(q))
