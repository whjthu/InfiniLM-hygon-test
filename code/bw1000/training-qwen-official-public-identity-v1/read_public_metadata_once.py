"""Fetch only ten public processor/config/index files at the verified revision."""
from pathlib import Path
import datetime,hashlib,json,time,urllib.request
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
identity=json.loads((D/'public-weight-identity.json').read_bytes());assert identity['passed']
model=json.loads((G/'coordination/training/qwen-original-model-files-v1.json').read_bytes())
W=D/'public-metadata';W.mkdir(exist_ok=False);start=time.monotonic();rows={}
for name,pin in model['files'].items():
 if name.endswith('.safetensors'):continue
 assert pin['bytes']<=8*1024**2
 url='https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/'+identity['revision']+'/'+name
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'project-model-identity-check'}),timeout=30) as response:
  assert response.geturl()==url;raw=response.read(pin['bytes']+1)
 assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
 with (W/name).open('xb') as f:f.write(raw)
 rows[name]=dict(**pin,public_url=url)
assert len(rows)==10
value=dict(passed=True,at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=identity['revision'],files=rows,metadata_bytes=sum(p['bytes'] for p in rows.values()),weight_payload_bytes=0,raw_training_tensor_bytes=0,wall_seconds=time.monotonic()-start)
with (D/'public-metadata-identity.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(passed=True,metadata_files=10,metadata_bytes=value['metadata_bytes'],weight_payload_bytes=0,raw_training_tensor_bytes=0)))
