import pathlib,json,hashlib,re,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':[],'network_evidence':[]}
expected=['69b7a9b78ad9a64200ac59af2252f2dda3414f8aa31f430090aec7e9bb632943','d3ae419818eed2d7aa722691b65eb8cbb23cfebd4cd28075d4f5ff95962f98ee']
for node in [0,1]:
 p=r/'results'/('CCL-raw-AR-r9214-node%d.json'%node);b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert h==expected[node];out['results'].append({'path':str(p),'SHA256':h,'data':json.loads(b)})
for p in sorted((r/'logs').glob('CCL-raw-AR-r9214-910373.3553-*.log')):
 text=p.read_text(errors='replace');lines=text.splitlines();keep=[l for l in lines if any(v in l for v in ['Using network','NET/IB','NET/Socket','GDRDMA','GPU Direct','GpuDirect','NCCL WARN','NCCL ERROR','CCL_RAW_AR_R9214_NORMAL'])];out['network_evidence'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':keep,'raw_linecount':len(lines),'used_IBext_v8':'Using network IBext_v8'in text,'using_Socket_data':'Using network Socket'in text,'explicit_GDRDMA':'GDRDMA'in text})
print(json.dumps(out))
