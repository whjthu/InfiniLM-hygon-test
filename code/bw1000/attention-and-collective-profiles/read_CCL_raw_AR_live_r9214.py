import pathlib,json,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'results/CCL-raw-AR-control-r9214.json';q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'control':json.loads(p.read_bytes())if p.exists()else None};q['sacct']=[]
for job in ['910406','910373']:q['sacct']+=[s for s in subprocess.check_output(['sacct','-S','2026-10-26','-j',job,'-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if'ccl-raw-ar-'in s.lower()]
q['logs']={p.name:p.read_text(errors='replace')[-3500:]for p in(r/'logs').glob('CCL-raw-AR-r9214*.log')};print(json.dumps(q))
