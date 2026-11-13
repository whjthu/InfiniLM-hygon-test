"""Check the exact public 480p T2V assets without reading weight payloads."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

B = Path(__file__).resolve().parent
tree = ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
         and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
ns = {'__file__': __file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'existing-hygon-ssh','exec'),ns)
remote = r'''
import datetime, hashlib, json, urllib.error, urllib.request
from pathlib import Path

root='https://huggingface.co/tencent/HunyuanVideo-1.5/'
revision='efe2609403af902268e12e0e6d9a54ae73d1884a'
dirpath='/transformer/480p_t2v/'
weight='diffusion_pytorch_model.safetensors'
size=33306632192
sha='71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86'
config_sha='689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None

opener=urllib.request.build_opener(NoRedirect())
def observe(url, method, limit):
    request=urllib.request.Request(url,method=method,headers={'Accept-Encoding':'identity'})
    try:
        with opener.open(request,timeout=8) as r:
            return dict(code=r.status,length=r.headers.get('Content-Length')),r.read(limit+1) if limit else b''
    except urllib.error.HTTPError as e:
        return dict(code=e.code,linked_size=e.headers.get('X-Linked-Size'),
                    linked_etag=(e.headers.get('X-Linked-Etag') or '').strip('"')),b''
    except (OSError,ValueError) as e:
        return dict(error_type=type(e).__name__,reason_type=type(getattr(e,'reason',None)).__name__),b''

head,_=observe(root+'resolve/'+revision+dirpath+weight,'HEAD',0)
head['linked_metadata_matches']=head.get('linked_size')==str(size) and head.get('linked_etag')==sha
pointer,payload=observe(root+'raw/'+revision+dirpath+weight,'GET',1024)
pointer['matches_original']=len(payload)<=1024 and {
    'oid sha256:'+sha,'size '+str(size)
} <= set(payload.decode('utf-8',errors='replace').splitlines())
config,data=observe(root+'raw/'+revision+dirpath+'config.json','GET',8192)
config.update(body_bytes=len(data),SHA256=hashlib.sha256(data).hexdigest() if data else None,
              matches_original=len(data)==968 and hashlib.sha256(data).hexdigest()==config_sha)
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
dest=R/'training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5'
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    revision=revision, public_weight_head=head,public_pointer=pointer,public_config=config,
    target_model_root_present=dest.exists(),
    original_private_cache_present=(R/'training-hunyuanvideo-original-inputs-v1/sample.pt').is_file(),
    weight_payload_bytes_read=0,private_payload_bytes_transferred=0,new_slurm_steps=0)))
'''
r = subprocess.run(ns['ssh'],input=remote.encode(),capture_output=True,timeout=45)
receipt = dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if not r.returncode:
    receipt['observation']=json.loads(r.stdout)
with (B/'hunyuan-official-public-target-route-v1.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
