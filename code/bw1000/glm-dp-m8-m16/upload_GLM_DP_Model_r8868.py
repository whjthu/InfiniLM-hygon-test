import pathlib,json,sys,subprocess,hashlib
r=pathlib.Path(__file__).resolve().parent;d=r/'GLM-DP-Mmatrix-private-r8868';m=json.loads((d/'GLM-CROSSREQUEST-Model-source-manifest-r8868.json').read_bytes())
names=list(dict.fromkeys([n for n in m['source_SHA256']if 'r8868'in n]+['GLM-CROSSREQUEST-Model-source-manifest-r8868.json','GLM-transient-Model-observer-manifest-r8868.json','dispatch_GLM_resident_exact_Model_r8868.py']))
for n in names:
 p=d/n
 if n in m['source_SHA256']:assert hashlib.sha256(p.read_bytes()).hexdigest()==m['source_SHA256'][n]
 if p.suffix=='.sh':assert b'\r\n'not in p.read_bytes()
subprocess.run([sys.executable,'-B','-X','utf8',str(r/'continue_refactor_remote.py'),'upload',*[str(d/n)for n in names]],check=True)
print(json.dumps({'uploaded_private_files':names,'shared_helpers_uploaded':False,'GPU':False}))
