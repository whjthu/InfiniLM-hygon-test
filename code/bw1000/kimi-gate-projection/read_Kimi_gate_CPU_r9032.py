import pathlib,json,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');print(json.dumps({'receipts':{str(p):json.loads(p.read_text())for p in(r/'results').glob('Kimi-gate-CPU-r9032-*.json')},'sacct':[x for x in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed'],text=True).splitlines()if 'r9032'in x]}))
