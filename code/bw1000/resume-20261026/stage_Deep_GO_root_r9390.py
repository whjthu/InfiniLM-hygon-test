import pathlib,json,hashlib,os,tempfile
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
g={'GPU_authorized': True, 'source_published': True, 'publication_branch': 'bw1000-experiments', 'publication_commit': '1d9e6e91cf17ac1933ffffe12311527efcd540bd', 'controller_manifest_SHA256': '1dc55cfca83a2607868e025fead58ade1914aa0583d0c971fd3a05300c7ef26e', 'CPU_receipt_SHA256': 'd9bc761eab923cbf6832de4312946cf0aeca74facb611bf6155382468f98ce4e', 'actual_CPU_step': '910374.1143', 'Common_CPU_HTTP_step': '910407.472', 'Common_CPU_HTTP_receipt_SHA256': '0f0b1837829872c7976affb5f93afa69c99cbab964bf65894c465dd71e8a8247'}
p=R/'candidates/Deep-agent-p0-r143c/CPU-preflight.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==g['CPU_receipt_SHA256']
o=json.loads(p.read_bytes());assert o['completed']and o['CPU_only']and o['static_only']and o['actual_step']==g['actual_CPU_step']
p=R/'results/bridge-C-timeout-contract-r145-910407.472/CPU-HTTP-timeout-contract-r145.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==g['Common_CPU_HTTP_receipt_SHA256']
p=R/'incoming/Deep-agent-P0-control-r143c/root-GO-after-CPU.json';data=(json.dumps(g,indent=2)+'\n').encode()
if p.exists():assert p.read_bytes()==data
else:
 fd,tmp=tempfile.mkstemp(dir=str(p.parent))
 try:
  with os.fdopen(fd,'wb')as f:f.write(data)
  os.link(tmp,str(p))
 finally:os.unlink(tmp)
print(json.dumps({'GO_staged':str(p),'CPU_step':g['actual_CPU_step']}))
