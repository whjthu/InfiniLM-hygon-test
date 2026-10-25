import pathlib,json,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'sacct':[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910407','-n','-P','-o','JobIDRaw,JobName%80,State,ExitCode,Elapsed'],text=True).splitlines()if 'norm-CPU-r9037'in x],'logs':{str(p):p.read_text()[-6000:]for p in(r/'logs').glob('Kimi-norm-CPU-r9037*')}}
for p in(r/'results').glob('Kimi-norm-CPU-r9037-*.json'):q['receipt']={'path':str(p),'data':json.loads(p.read_text())}
print(json.dumps(q))
