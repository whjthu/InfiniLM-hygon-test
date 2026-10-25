import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-A8-ABI-real-r8994';o={}
for suffix in ['launch','watch']:
 p=r/(label+'-'+suffix+'.json')
 if p.exists():o[suffix]=json.loads(p.read_bytes());o[suffix+'_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
w=o.get('watch',{});h=w.get('actual_handle')
if h:
 p=r/'results'/(label+'-'+h+'.json')
 if p.exists():raw=p.read_bytes();o['result']=json.loads(raw);o['result_SHA256']=hashlib.sha256(raw).hexdigest()
 p=r/'logs'/(label+'-'+h+'.log')
 if p.exists():o['numeric_tail']=p.read_text(errors='replace')[-3000:]
for suffix in ['controller','watcher','srun']:
 p=r/'logs'/(label+'-'+suffix+'.log')
 if p.exists():o[suffix+'_tail']=p.read_text(errors='replace')[-1800:]
print(json.dumps(o))
