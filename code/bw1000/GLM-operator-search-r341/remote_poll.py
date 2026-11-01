import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';b=r/'candidates/GLM-operator-search-r341'
q={'CPU_RESULT':json.loads((b/'CPU_RESULT.json').read_text()) if (b/'CPU_RESULT.json').exists() else None}
q['sacct']=subprocess.run(['sacct','-S','2026-11-01','-j','910407,910371,910374','-n','-P','-o','JobID,JobName%70,State,ExitCode,NodeList'],capture_output=True,text=True).stdout
q['sacct']='\n'.join(s for s in q['sacct'].splitlines() if 'GLM341' in s)
q['logs']={str(f):f.read_text(errors='replace')[-5000:] for f in (r/'logs').glob('GLM341-CPU*.log')}
q['results']={str(f):json.loads(f.read_text()) for f in (r/'results').glob('GLM341-*.json')}
print(json.dumps(q))
