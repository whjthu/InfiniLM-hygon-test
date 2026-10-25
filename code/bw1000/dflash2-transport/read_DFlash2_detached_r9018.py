import hashlib,json,pathlib,socket
b=pathlib.Path.home()/'hygon-inference-refactor-20261014'
receipt=b/'results/DFlash2-CDN-detached-r9018.json';log=b/'results/DFlash2-CDN-detached-r9018.log'
folder=b/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769'
d={'observer_node':socket.gethostname(),'expected_worker_node':'zz-login01','PID':1373747,'receipt_exists':receipt.exists(),'log_exists':log.exists()}
if socket.gethostname()=='zz-login01':d['process_exists']=pathlib.Path('/proc/1373747').exists()
if log.exists():
 lines=log.read_text().splitlines();d['progress_line_count']=len(lines)
 if lines:
  last=json.loads(lines[-1]);d['latest']=last if 'terminal' not in last else {'terminal_present':True}
if receipt.exists():
 raw=receipt.read_bytes();full=json.loads(raw);d['receipt_SHA256']=hashlib.sha256(raw).hexdigest();d['terminal']={k:v for k,v in full.items() if k!='segments'};d['completed_segment_count']=len(full.get('segments',[]))
for name in ['model.safetensors.r9004.partial','model.safetensors']:
 p=folder/name;d[name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
print(json.dumps(d))
