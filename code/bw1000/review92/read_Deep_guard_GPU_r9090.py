import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q={}
for n in ['deep-guard-r9090-launch.json','deep-guard-r9090-watch.json']:
 p=r/n
 if p.exists():q[n]=json.loads(p.read_bytes())
for p in (r/'results').glob('deep-guard-r9090-910374.*.json'):q[p.name]=json.loads(p.read_bytes())
q['accounting']=[x for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobID,JobName%70,State,ExitCode'],text=True).splitlines()if 'deep-guard'in x]
p=r/'candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/third_party/InfiniCore';q['Core_shallow']=[str(x.relative_to(p))for x in p.glob('*')]
print(json.dumps(q))
