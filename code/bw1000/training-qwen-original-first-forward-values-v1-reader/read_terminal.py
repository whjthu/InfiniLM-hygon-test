import datetime,hashlib,json,subprocess
from pathlib import Path
l='training-qwen-original-first-forward-values-v1'
rq='70bc6bfcd7c641d0aefaa04980ac96d7'
r=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/(l+'-'+rq)
cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
cmd=subprocess.run(['sacct','-n','-P','-j','1103606','--format='+cols],capture_output=True,text=True,timeout=8,check=True)
rows=[x for ln in cmd.stdout.splitlines() if (x:=ln.split('|'))[1]==l]
assert len(rows)<=1
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=rq,columns=cols.split(','),accounting=rows,namespace=r.is_dir(),pins={},data={})
for n in ('intent.json','terminal.json','result.json'):
 p=r/n
 if p.is_file():
  b=p.read_bytes()
  assert 0<len(b)<=131072
  out['pins'][n]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
  out['data'][n]=json.loads(b)
if rows and rows[0][2] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'):
 p=r/'srun.log'
 if p.is_file():
  b=p.read_bytes()
  assert len(b)<=65536
  out['logged_error_class_and_guard']=[x for x in b.decode(errors='replace').splitlines() if x.startswith(('AssertionError','ValueError','RuntimeError','FileNotFoundError','KeyError','TypeError'))][-3:]
print(json.dumps(out,allow_nan=False))
