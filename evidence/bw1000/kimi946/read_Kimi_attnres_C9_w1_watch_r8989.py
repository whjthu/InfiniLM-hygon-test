import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='kimi-attnres-C9-w1-r8989';out={'CPU_only':True,'watch_path':str(r/(label+'-watch.json'))}
for kind in ['launch','watch']:
 p=r/(label+'-'+kind+'.json')
 if p.exists():out[kind]=json.loads(p.read_text());out[kind+'_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
w=out.get('watch',{});handle=w.get('actual_handle')
if handle:
 p=r/'results'/(label+'-'+handle+'.json');out['result_exists']=p.exists()
 if p.exists():out['result']=json.loads(p.read_text());out['result_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
 p=r/'logs'/(label+'-'+handle+'.log')
 if p.exists():out['log_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest();out['log_tail']=p.read_text(errors='replace')[-5000:]
if not handle:
 p=r/'logs'/(label+'-controller.log')
 if p.exists():out['controller_log_tail']=p.read_text(errors='replace')[-3000:]
print(json.dumps(out))
