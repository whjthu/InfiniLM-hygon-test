import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';b=r/'candidates/Kimi-W2-INT8-register-v2-r113c';p=b/'CPU-build.json';out={'squeue':subprocess.check_output(['squeue','--steps','-j','910375','-h','-o','%i|%j|%N'],text=True),'CPU_build':json.loads(p.read_bytes())if p.exists()else None,'CPU_build_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None}
if out['CPU_build']:out['sacct']=subprocess.check_output(['sacct','-j',out['CPU_build']['actual_step'],'--format=JobID,State,ExitCode,Elapsed,NodeList','-P','-n'],text=True)
else:out['logs']={p.name:p.read_text()[-2000:]for p in b.glob('*.log')}
p=b/'matrix-ISA.s'
if p.exists():out['matrix_ISA']=p.read_text()
print(json.dumps(out))
