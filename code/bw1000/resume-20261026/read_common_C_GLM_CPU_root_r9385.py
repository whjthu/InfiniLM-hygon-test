import pathlib,json,hashlib,subprocess,datetime,re
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'contracts':[]}
for d in sorted((r/'results').glob('bridge-C-timeout-contract-r145-*')):
 for p in sorted(d.glob('*.json')):
  assert p.stat().st_size<1000000;out['contracts'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_bytes())})
p=r/'candidates/GLM-agent-service-r143b/CPU-prepare.json';out['GLM_CPU']={'exists':p.exists()}
if p.exists():out['GLM_CPU'].update(receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_bytes()))
rows=subprocess.check_output(['sacct','-S','2026-10-27','-j','910407','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True,timeout=30).splitlines();out['Slurm']=[s.split('|')for s in rows if 'bridge-C-timeout-CPU-r145'in s or'GLM-agent-service-CPU-r143b'in s]
for row in out['Slurm']:
 if row[2]=='FAILED':
  p=r/'logs'/(row[1]+'-'+row[0]+'.log')
  if p.exists():out.setdefault('errors',[]).append({'step':row[0],'lines':[s[:1200]for s in p.read_text(errors='replace').splitlines()if re.search('Error|Assertion|Traceback|No module',s)][-12:]})
print(json.dumps(out))
