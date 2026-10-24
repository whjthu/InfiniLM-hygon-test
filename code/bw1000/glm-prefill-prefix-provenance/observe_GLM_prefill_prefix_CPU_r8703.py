import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d={'CPU_only':True,'no_replay':True};rows=subprocess.check_output(['sacct','-j','910407','--noheader','--parsable2','--format=JobIDRaw,JobName%70,State,ExitCode,Elapsed,NodeList','-S','2026-10-24'],text=True,timeout=30);d['rows']=[v.split('|')for v in rows.splitlines()if 'GLM-prefill-prefix-CPU-r8703'in v]
p=r/'candidates/glm-prefill-prefix-r8703/CPU-build.json'
if p.is_file():
 q=json.loads(p.read_text());d['build']={k:v for k,v in q.items()if k!='commands'}
d['logs']={p.name:p.read_text(errors='replace')[-7000:] for p in (r/'logs').glob('GLM-prefill-prefix-CPU-r8703-*.log')}
print(json.dumps(d))
