import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/Kimi-A8-ABI-CPU-r8993.json';out={'CPU_only':True,'receipt_exists':p.exists()}
if p.exists():out['receipt']=json.loads(p.read_text());out['receipt_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
out['accounting']=subprocess.check_output(['sacct','-S','2026-10-25','-j','910406','-n','-P','-o','JobID,JobName%80,State%30,ExitCode,NodeList%80'],text=True)
out['logs']={str(p):p.read_text(errors='replace')[-5000:]for p in(r/'logs').glob('Kimi-A8-ABI-CPU-r8993*')if p.is_file()}
print(json.dumps(out))
