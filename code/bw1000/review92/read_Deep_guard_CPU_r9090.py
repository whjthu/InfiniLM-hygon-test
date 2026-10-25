import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q={};q['accounting']=subprocess.check_output(['sacct','-j','910407','-S','2026-10-25','-n','-P','-o','JobID,JobName%60,State,ExitCode'],text=True).splitlines()[-12:]
p=r/'results/deep-guard-CPU-r9090.json'
if p.exists():q['receipt']=json.loads(p.read_bytes());import hashlib;q['receipt_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
q['logs']={p.name:p.read_text()[-5000:]for p in (r/'logs').glob('Deep-guard-CPU-r9090-*.log')}
p=r/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/InfiniCore';q['core_files']=[str(x.relative_to(p))for x in p.glob('src/**/*') if x.is_file()and 'operator'in x.name.lower()][:50]
print(json.dumps(q))
