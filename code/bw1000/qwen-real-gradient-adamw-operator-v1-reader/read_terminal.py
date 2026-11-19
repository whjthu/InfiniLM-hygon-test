import datetime,hashlib,json,statistics,subprocess
from pathlib import Path
label='qwen-real-gradient-adamw-operator-v1'
req='dd90c349e50a46ff8f86c0987a4abb51'
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/(label+'-'+req)
def read(name):
 p=root/name
 if not p.is_file():return None,None
 b=p.read_bytes()
 assert 0<len(b)<=262144
 return json.loads(b),dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
fmt='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
p=subprocess.run(['sacct','-n','-P','-j','1087656','--format='+fmt],capture_output=True,timeout=8,check=True)
rows=[x for line in p.stdout.decode().splitlines() if (x:=line.split('|'))[1] in (label,label+'-pre',label+'-post',label+'-failure-post')]
assert len({r[1] for r in rows})==len(rows)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=req,accounting_columns=fmt.split(','),accounting=rows,
         namespace=root.is_dir(),data={},pins={})
for name in ('intent.json','terminal.json','device.json','result.json','rank-00-result.json','rank-06-result.json'):
 d,pin=read(name)
 if d:
  out['pins'][name]=pin
  if name.startswith('rank-'):
   blocks=d.get('blocks',[])
   if blocks:
    assert [b['mode'] for b in blocks]==['foreach','fused','fused','foreach']
    for block in blocks:assert len(block['wall_seconds'])==len(block['event_ms'])==block['timed']==20 and min(block['wall_seconds'])>0
    d['fused_over_foreach']={k:statistics.median(b[k] for b in blocks if b['mode']=='fused')/statistics.median(b[k] for b in blocks if b['mode']=='foreach')
       for k in ('event_median_ms','wall_median_seconds')}
   out['data'][name]=d
  elif name=='terminal.json':
   out['data'][name]={k:d.get(k) for k in ('request','gpu_passed','clean_post','step','error_class')}
  else:out['data'][name]=d
print(json.dumps(out,allow_nan=False))
