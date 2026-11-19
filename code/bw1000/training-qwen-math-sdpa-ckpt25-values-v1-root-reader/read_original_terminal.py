import datetime,hashlib,json,subprocess
from pathlib import Path
r=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-math-sdpa-ckpt25-values-v1')
def read(where):
 if not where.is_file():return None,None
 assert where.stat().st_size<=65536
 data=where.read_bytes()
 try:j=json.loads(data)
 except json.JSONDecodeError:return None,None
 return j,{'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest()}
def query(*a):return subprocess.check_output(a,timeout=10,text=True).splitlines()
outcome,pin=read(r/'outcome.json')
intent,intent_pin=read(r/'intent.json')
cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%170'
rows=[line.split('|') for line in query('sacct','-j','1103606','-n','-P','--format='+cols) if line.split('|')[1]==r.name]
live=query('squeue','-s','-h','-j','1103606','-o','%i')
code=None
if (r/'CPU.log').is_file():
 f=r/'CPU.log'
 with f.open('rb') as s:
  s.seek(max(0,f.stat().st_size-8000));t=s.read(8000).decode(errors='replace')
 code=[k for k in ('AssertionError','FileNotFoundError','ValueError','RuntimeError','ImportError','Killed','TIMEOUT') if k in t]
ready=bool(outcome and len(rows)==1 and rows[0][2] in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED') and rows[0][0] not in live)
if ready:
 assert outcome['request']=='384ecc576e115e57a36d4520a9f67df9'
 if outcome.get('passed'):
  assert rows[0][2:4]==['COMPLETED','0:0'] and 'gres/dcu=' not in rows[0][6] and outcome['compute']['step']==rows[0][0]
  assert outcome['compute']['GPU_cards']==0 and outcome['compute']['adapter']['tensors']==504 and outcome['compute']['optimizer_step_all_25']
  actual,actual_pin=read(r/'evidence/compute.json')
  assert actual==outcome['compute']
 else:actual_pin=None
else:actual_pin=None
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_request='384ecc576e115e57a36d4520a9f67df9',
    account=rows,accounting_columns=cols.split(','),terminal=ready,outcome_pin=pin,cpu_payload_pin=actual_pin,intent_pin=intent_pin,
    CPU_error_classes=code,result=outcome),allow_nan=False))
