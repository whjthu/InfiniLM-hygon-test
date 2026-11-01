import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
raw=subprocess.check_output(['sacct','-S','2026-10-24','-j','910407','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed'],text=True)
q={'sacct':'\n'.join(s for s in raw.splitlines()if 'GLM349-whole-glue' in s),'logs':{str(p):p.read_text(errors='replace')[-8000:] for p in(r/'logs').glob('GLM349-whole-glue-v*.log')},'receipts':[]}
for d in(r/'results').glob('GLM349-whole-glue-*'):
 p=d/'CPU-result.json'
 if p.exists():q['receipts'].append({'path':str(p),'SHA256':__import__('hashlib').sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_bytes())})
print(json.dumps(q))
