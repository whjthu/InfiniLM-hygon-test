import pathlib,json,subprocess,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'results/CCL-channel-control-r9217.json';q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'control':json.loads(p.read_bytes())if p.exists()else None,'sacct':[]}
for job in ['910406','910373']:q['sacct']+=[s for s in subprocess.check_output(['sacct','-S','2026-10-26','-j',job,'-n','-P','-o','JobIDRaw,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if'ccl-channel-'in s.lower()]
q['logs']={p.name:p.read_text(errors='replace')[-2500:]for p in(r/'logs').glob('CCL-channel-r9217-*.log')};print(json.dumps(q))
