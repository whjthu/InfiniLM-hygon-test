from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in D.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
for name in ['owner','controller']:
 d=D/name;files={str(p.relative_to(d)).replace('\\','/'):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='SOURCE_MANIFEST.json'}
 q={'schema':1,'Native_SHA256':'038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00','GPU_GO':False,'files_SHA256':files}
 (d/'SOURCE_MANIFEST.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
files={str(p.relative_to(D)).replace('\\','/'):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name not in ['CPU_PACKAGE_MANIFEST.json','CHECKPOINT.json','controller-CPU-local.json']}
(D/'CPU_PACKAGE_MANIFEST.json').write_text(json.dumps({'schema':1,'GPU':0,'files_SHA256':files},indent=2)+'\n',encoding='utf8')
print(json.dumps({'owner_SHA256':sha(D/'owner/SOURCE_MANIFEST.json'),'controller_SHA256':sha(D/'controller/SOURCE_MANIFEST.json'),'CPU_package_SHA256':sha(D/'CPU_PACKAGE_MANIFEST.json')}))
