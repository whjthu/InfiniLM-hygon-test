import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP';out={'read_only':True,'GPU':0,'directory':str(d),'selection_rule':'Inspect original numeric file order and full request input shape/tool name only; never rank by answer correctness or measured speed. Keep frozen request002 anchor. Select earliest actual short tool request and earliest distinct long named-tool request after anchor if present.','filenames':[p.name for p in d.iterdir()],'requests':[]}
for p in sorted(d.glob('request-*.request.json')):
 raw=p.read_bytes();q=json.loads(raw);i=int(p.name.split('-')[1].split('.')[0]);z={'index':i,'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'tool_choice':q.get('tool_choice'),'tools':[(t.get('function')or{}).get('name') for t in q.get('tools',[])],'max_tokens':q.get('max_tokens'),'message_roles':[x.get('role') for x in q.get('messages',[])],'message_char_counts':[len(str(x.get('content',''))) for x in q.get('messages',[])]}
 if i in [0,1,2,3,4,5]:z['complete_body']=q
 out['requests'].append(z)
print(json.dumps(out,ensure_ascii=False))
