import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';rows=[v for v in subprocess.check_output(['sacct','-S','2026-10-25','-j','910407','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%80'],text=True,timeout=30).splitlines()if'Deep-Cast-frontend-CPU-r9060'in v];q=dict(accounting=rows)
for row in rows:
 p=r/'logs'/('Deep-Cast-frontend-CPU-r9060-'+row.split('|')[0]+'.log')
 if p.exists():
  s=p.read_text(errors='replace');q.update(logtail=s[-7000:],log_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),preflight_rows=[json.loads(v)for v in s.splitlines()if v.startswith('{')])
print(json.dumps(q))
