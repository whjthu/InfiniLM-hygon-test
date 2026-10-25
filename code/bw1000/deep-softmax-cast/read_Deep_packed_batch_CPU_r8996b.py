import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out={'CPU_only':True};a=subprocess.check_output(['sacct','-S','2026-10-24','-j','910407,910406','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%80'],text=True,timeout=30);out['accounting']=[s for s in a.splitlines()if'Deep-packed-batch-CPU-r8996b'in s]
p=r/'results/deep-packed-batch-CPU-r8996b.json'
if p.exists():
 q=json.loads(p.read_bytes());out['receipt_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();out['CPU']={k:q[k]for k in ['producer','completed','CPU_only','CPU_seconds','scope']};out['library']=q['libraries']['full'];out['required_symbols']=q['actual_Model_consumer_required_symbols'];out['new_units']=[{k:u[k]for k in ['source','source_SHA256','object_SHA256','fresh_M_equals_MD']}for u in q['new_compiled_units']]
for line in out['accounting']:
 tag=line.split('|')[0];p=r/'logs'/('Deep-packed-batch-CPU-r8996b-'+tag+'.log')
 if p.exists():out['log_tail']=p.read_text(errors='replace')[-7000:]
print(json.dumps(out))
