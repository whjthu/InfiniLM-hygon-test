import pathlib,json,hashlib,base64
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag='GLM-regular-service-r204-b34db8d1eb674914';d=r/'results'/('GLM-regular-live-r204-'+tag)/'HTTP';owner=r/'results'/('glm-agent-service-r204-'+tag+'-stage3-rank0.json');out={'read_only':True,'GPU':0,'selected_by_original_order_input_shape':[1,2,4],'owner_path':str(owner),'fixtures':[]}
if owner.exists():
 raw=owner.read_bytes();q=json.loads(raw);out['owner_SHA256']=hashlib.sha256(raw).hexdigest();out['owner_keys']=list(q);rows=q.get('service_requests',[])
else:rows=[]
for i in [1,2,4]:
 request=d/('request-%03d.request.json'%i);raw=request.read_bytes();body=json.loads(raw);wire=d/('request-%03d.wire'%i);reply=wire.read_bytes();decoded=d/('request-%03d.decoded.json'%i);parsed=json.loads(decoded.read_bytes());w=json.loads(reply);rid=w.get('id')
 z={'index':i,'request_path':str(request),'request_SHA256':hashlib.sha256(raw).hexdigest(),'raw_request_base64':base64.b64encode(raw).decode(),'wire_path':str(wire),'wire_SHA256':hashlib.sha256(reply).hexdigest(),'raw_reply_base64':base64.b64encode(reply).decode(),'wire':w,'decoded':parsed,'owner_rows_matching_id':[x for x in rows if x.get('request_id')==rid]}
 out['fixtures'].append(z)
print(json.dumps(out,ensure_ascii=False))
