from pathlib import Path
import hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-minicpm-first-update-recorder-CPU-v1'
def load(p):
 assert p.stat().st_size<1000000
 raw=p.read_bytes();return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
out,outpin=load(W/'outcome.json');assert out==EXPECTED and out['passed'] is False and out['returncode']==1
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656.129','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if x.strip()]
assert rows==out['accounting'] and rows[0][2:4]==['FAILED','1:0'] and rows[0][5]=='f11r4n09'
intent,intpin=load(W/'intent.json');assert intent['claim']=='c6138f5b4f9bb2328902c51338252c655a78bb21' and intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']==out['epoch']+240
for name,h in intent['files'].items():
 p=W/'source'/name;assert p.stat().st_size<1000000 and hashlib.sha256(p.read_bytes()).hexdigest()==h
assert len(intent['files'])==13
for name in ('CPU-result.json','owned-node.json'):
 value,pin=load(W/'evidence'/name);assert pin['SHA256']==out[name]['SHA256'] and value==out[name]['value']
assert out['CPU-result.json']['value']['owner']==out['owned-node.json']['value']
assert hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest()==out['log_SHA256']
metadata=[]
for folder in ('usr/local/lib/python3.10/site-packages','usr/local/lib/python3.10/dist-packages','usr/lib/python3.10/site-packages'):
 base=R/'containers/dtk26-sandbox-r1'/folder
 for part in ('torch','triton','triton/_C'):
  root=base/part
  if not root.is_dir():continue
  targets=[root/'__init__.py']
  if part=='torch':targets+=list(root.glob('_C.*.so'))
  if part=='triton/_C':targets+=list(root.glob('libtriton*.so'))
  for path in targets:
   if path.is_file():metadata.append(dict(package=part,file=path.name,bytes=path.stat().st_size))
print(json.dumps(dict(independent_failed_terminal_verified=True,accounting=rows,outcome=outpin,intent=intpin,source_files_verified=13,CPU_result=out['CPU-result.json'],runtime_file_metadata=metadata,log_SHA256=out['log_SHA256'],GPU_cards=0,Model_GO=False)))
