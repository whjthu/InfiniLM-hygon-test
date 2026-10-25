import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';a=subprocess.check_output(['sacct','-S','2026-10-25','-j','910407','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%80'],text=True,timeout=30);rows=[v for v in a.splitlines()if'Deep-distinct-tokenizer-CPU-r9070b'in v];q=dict(accounting=rows)
p=r/'results/Deep-distinct32-frozen-r9070b.json'
if p.exists():q.update(frozen_inputs=json.loads(p.read_bytes()),frozen_inputs_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),frozen_inputs_bytes=p.stat().st_size)
for row in rows:
 p=r/'logs'/('Deep-distinct-tokenizer-CPU-r9070b-'+row.split('|')[0]+'.log')
 if p.exists():q['logtail']=p.read_text(errors='replace')[-5000:]
q['actual7169sources']=[str(p)for p in (r/'candidates/deep-combined-r7169').glob('**/*.cpp')][:24]
q['receipts7169']=[str(p)for p in (r/'results').glob('*7169*.json')][:16]
print(json.dumps(q))
