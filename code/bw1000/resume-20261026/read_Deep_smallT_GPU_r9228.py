import pathlib,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-smallT-packed-r9228';q={}
for n in [label+'-launch.json',label+'-watch.json']:
 p=r/n;q[n]=json.loads(p.read_text())if p.exists()else None
w=q.get(label+'-watch.json')or{};h=w.get('actual_handle')
if h:
 p=r/'results'/(label+'-'+h+'.json');q['report']=json.loads(p.read_text())if p.exists()else None;p=r/'logs'/(label+'-'+h+'.log');q['log_tail']=p.read_text(errors='replace')[-4000:]if p.exists()else None
print(json.dumps(q))
