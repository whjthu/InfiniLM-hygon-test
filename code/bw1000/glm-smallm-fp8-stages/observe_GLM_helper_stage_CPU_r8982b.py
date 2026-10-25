import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'candidates/glm-helper-stage-trace-r8982b/CPU-build.json';q={'exists':p.exists(),'GPU':False,'CPU_only':True}
if p.exists():q.update(receipt=json.loads(p.read_text()),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),build_log=(p.parent/'build.log').read_text())
q['Slurm']=[x for x in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobIDRaw,JobName%70,State,ExitCode,Elapsed,NodeList%50'],text=True).splitlines()if 'helper-stage-trace-CPU-r8982b'in x];print(json.dumps(q))
