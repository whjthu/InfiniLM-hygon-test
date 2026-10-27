import pathlib,json,os,tempfile,hashlib
items=[('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM-agent-service-r143b/root-GO-r9388.json', {'GPU_authorized': True, 'source_published': True, 'publication_branch': 'bw1000-experiments', 'publication_commit': '1d9e6e91cf17ac1933ffffe12311527efcd540bd', 'GPU_manifest_SHA256': '39403de9502f544d613159331db67de1274d05b10852ac1729ff4267c2e7a21c', 'CPU_receipt_SHA256': '8da319aaf5376b1957882541a12430080501d4482cbb21a9e330bad1582d0721'}), ('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Kimi-agent-service-Model-r144c/root-GO-r9388.json', {'GPU_authorized': True, 'source_published': True, 'publication_branch': 'bw1000-experiments', 'publication_commit': '1d9e6e91cf17ac1933ffffe12311527efcd540bd', 'source_manifest_SHA256': '924dceef58c46007b77a8576eb0820617261435bdd1b9d1e22d2a9e7ab94648b'})]
for name,q in items:
 p=pathlib.Path(name);data=(json.dumps(q,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==data
 else:
  fd,tmp=tempfile.mkstemp(dir=str(p.parent))
  try:
   with os.fdopen(fd,'wb')as f:f.write(data)
   os.link(tmp,str(p))
  finally:os.unlink(tmp)
print(json.dumps({'GO_staged':[x[0]for x in items],'source_published':True}))
