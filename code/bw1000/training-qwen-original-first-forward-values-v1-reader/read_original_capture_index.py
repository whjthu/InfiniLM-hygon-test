import datetime,hashlib,json,stat,subprocess
from pathlib import Path
r=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
labels=('training-qwen-first-resumed-forward-v1','training-qwen-resumed-decoder-internals-v1')
p=subprocess.run(['sacct','-n','-P','-j','1087656.429,1087656.433','--format=JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList'],capture_output=True,text=True,timeout=8,check=True)
rows=[x.split('|') for x in p.stdout.splitlines() if x]
assert {x[0] for x in rows}=={'1087656.429','1087656.433'} and all(x[2:4]==['COMPLETED','0:0'] for x in rows)
report=[]
names=['initial.pt',*[f'forward-{i:02d}-{kind}.pt' for i in range(1,9) for kind in ('input','output')],
       *[f'update-{i:02d}-{kind}.pt' for i in (1,2) for kind in ('preclip','postclip','before-optimizer','after-optimizer')]]
assert len(set(names))==25
for rank in range(8):
 receipts=[]
 for label in labels:
  q=r/label/'continuous/capture'/f'rank-{rank:02d}'/'receipt.json'
  st=q.lstat()
  assert stat.S_ISREG(st.st_mode) and 0<st.st_size<=65536
  b=q.read_bytes();d=json.loads(b)
  assert d['rank']==rank and d['completed_global_step']==2 and d['optimizer_updates']==d['clip_calls']==2
  files={x['name']:{k:x[k] for k in ('bytes','SHA256')} for x in d['files']}
  assert set(files)==set(names) and len(files)==len(d['files'])
  receipts.append((dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest()),files))
 a,b=(q[1] for q in receipts)
 equal=[x for x in names if a[x]==b[x]]
 different=[x for x in names if a[x]!=b[x]]
 report.append(dict(rank=rank,receipt_pins=[q[0] for q in receipts],matched=equal,different=different,
                    first_mismatch=different[0] if different else None))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels=labels,rows=rows,
                      rank_reports=report,compared_files=len(names)*len(report),archive_bytes_read=0)))
