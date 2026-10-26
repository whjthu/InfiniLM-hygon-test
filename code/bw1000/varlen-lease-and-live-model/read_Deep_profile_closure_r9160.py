import pathlib,json,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/deep-packed-varlen-CPU-r9101b.json';d=json.loads(p.read_bytes());objs=list(d['link']['full']['objects']);o={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':str(p),'completed':d['completed'],'library':d['libraries']['full'],'compile_prefix':d['new_compiled_units'][0]['compile_argv'],'objects':[],'candidate_sources':[]}
seen=set()
for n in objs:
 q=pathlib.Path(n);o['objects'].append({'object':n,'SHA256':d['link']['full']['objects'][n]})
 for folder in [q.parent,q.parent/'src',q.parent/'incoming']:
  if not folder.is_dir():continue
  for s in folder.glob('*.cpp'):
   if s in seen:continue
   seen.add(s)
   if any(x in s.name for x in ['routed_text_model','native_engine','probe','moe','attention']):o['candidate_sources'].append({'path':str(s),'bytes':s.stat().st_size})
print(json.dumps(o))
