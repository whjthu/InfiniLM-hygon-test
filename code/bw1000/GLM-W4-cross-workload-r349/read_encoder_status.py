import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';raw=subprocess.check_output(['sacct','-S','2026-10-24','-j','910407','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed'],text=True);q={'sacct':'\n'.join(s for s in raw.splitlines()if'GLM349-official-encoder'in s),'logs':{str(p):p.read_text(errors='replace')[-6000:]for p in(r/'logs').glob('GLM349-encoder-*.log')},'metadata':[]}
for d in(r/'results').glob('GLM349-official-encoding-*'):
 p=d/'metadata.json'
 if p.exists():q['metadata'].append({'path':str(p),'data':json.loads(p.read_text())})
print(json.dumps(q))
