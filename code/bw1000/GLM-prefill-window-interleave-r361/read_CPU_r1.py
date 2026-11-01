from pathlib import Path
import subprocess,json,hashlib
R=Path.home()/'hygon-inference-refactor-20261014';raw=subprocess.check_output(['sacct','-S','2026-10-24','-j','910407','-n','-P','-o','JobIDRaw,JobName%120,State,ExitCode,Elapsed'],text=True,timeout=25)
q={'sacct':'\n'.join(x for x in raw.splitlines()if 'GLM-r361-whole-CPU-r1'in x),'logs':{str(p):p.read_text(errors='replace')[-7000:]for p in(R/'logs').glob('GLM-r361-whole-CPU-r1*.log')},'receipts':[]}
for d in(R/'results').glob('GLM361-whole-*'):
 p=d/'CPU-result.json'
 if p.exists():q['receipts'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_bytes())})
print(json.dumps(q))
