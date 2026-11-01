import pathlib,json,hashlib,base64
p=pathlib.Path(__file__).parent;d=p/'repair_v3';d.mkdir(exist_ok=True)
for f in (p/'repair_v2').iterdir():
 if f.name=='SOURCE_MANIFEST.json':continue
 s=f.read_text().replace('GLM-operator-search-r341/repair_v2','GLM-operator-search-r341/repair_v3').replace('GLM341c','GLM341d')
 s=s.replace("str(r/'logs/GLM341d-'+mode+'-%J-rank%t.log')","str(r/('logs/GLM341d-'+mode+'-%J-rank%t.log'))")
 (d/f.name).write_text(s,newline='\n')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();m=json.loads((p/'repair_v2/SOURCE_MANIFEST.json').read_text());m['supersedes_pre_GPU_type_error_manifest_SHA256']=sha(p/'repair_v2/SOURCE_MANIFEST.json');m['files_SHA256']={f.name:sha(f) for f in d.iterdir() if f.name!='SOURCE_MANIFEST.json'}
(d/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2))
files={f.name:base64.b64encode(f.read_bytes()).decode() for f in d.iterdir()}
s="""import pathlib,json,base64
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/GLM-operator-search-r341/repair_v3';assert not d.exists();d.mkdir()
files=FILES
for n,b in files.items():(d/n).write_bytes(base64.b64decode(b))
print(json.dumps({'staged':list(files),'GPU_submitted':False}))
""".replace('FILES',repr(files));(p/'remote_stage_v3.py').write_text(s);print(sha(d/'SOURCE_MANIFEST.json'))
