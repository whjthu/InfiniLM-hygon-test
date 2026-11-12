"""Read only official tiny LFS pointers for the already-hashed original weights."""
from pathlib import Path
import datetime,hashlib,json,re,urllib.request
D=Path(__file__).resolve().parent
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
model=json.loads((G/'coordination/training/qwen-original-model-files-v1.json').read_bytes())
revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b';base='https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/'+revision+'/'
result=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='Qwen/Qwen3-VL-8B-Instruct',revision=revision,files={},public_weight_payload_bytes_downloaded=0)
for name,pin in model['files'].items():
 if not name.endswith('.safetensors'):continue
 request=urllib.request.Request(base+name,headers={'User-Agent':'project-model-identity-check'})
 with urllib.request.urlopen(request,timeout=30) as response:
  assert response.geturl()==base+name
  raw=response.read(4097);assert 0<len(raw)<=4096
 lines=raw.decode('ascii').splitlines();assert len(lines)==3 and lines[0]=='version https://git-lfs.github.com/spec/v1'
 oid=re.fullmatch('oid sha256:([0-9a-f]{64})',lines[1]);size=re.fullmatch('size ([0-9]+)',lines[2]);assert oid and size
 assert oid[1]==pin['SHA256'] and int(size[1])==pin['bytes']
 with (D/(name+'.pointer.txt')).open('xb') as f:f.write(raw)
 result['files'][name]=dict(public_pointer_url=base+name,pointer_bytes=len(raw),pointer_SHA256=hashlib.sha256(raw).hexdigest(),weight_SHA256=oid[1],weight_bytes=int(size[1]),matches_original_complete_SHA=True)
assert len(result['files'])==4
result.update(passed=True,public_weight_bytes=sum(p['weight_bytes'] for p in result['files'].values()),original_weights_rehashed=False,private_trainable_exports=0,GPU_calls=0)
with (D/'public-weight-identity.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(passed=True,official_pointer_files=4,matched_original_weight_bytes=result['public_weight_bytes'],public_weight_payload_bytes_downloaded=0,private_trainable_exports=0,GPU_calls=0)))
