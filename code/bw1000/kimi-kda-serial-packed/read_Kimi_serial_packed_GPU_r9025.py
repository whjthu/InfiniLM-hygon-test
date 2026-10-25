import pathlib,json,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');tag='Kimi-KDA-only-serial-packed-r9025';p=r/'results'/(tag+'-terminal.json');q={'sacct':[x for x in subprocess.check_output(['sacct','-j','910374','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if 'r9025'in x],'watch':json.loads(p.read_text())if p.exists()else None,'logs':{}}
for p in(r/'logs').glob(tag+'-*'):
 if p.is_file():q['logs'][str(p)]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'tail':p.read_text()[-6000:]}
for p in(r/'results').glob(tag+'-910374.*.json'):
 q['original_receipt']={'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(q))
