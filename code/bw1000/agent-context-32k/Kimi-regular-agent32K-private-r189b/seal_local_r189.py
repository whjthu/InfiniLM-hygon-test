"""Local/CPU static sealing only. Never import Native, torch, or start jobs."""
import ast,hashlib,json,pathlib,sys
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def seal():
    for p in D.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
    model=D/'runtime/model-manifest-r9183.json'; m=json.loads(model.read_bytes())
    prefix='/refactor/incoming/Kimi-agent-service-Model-r189/'
    m=json.loads(json.dumps(m).replace('r144d','r189'))
    for name in list(m['source_SHA256']):
        assert name.startswith(prefix),name
        p=D/name.removeprefix(prefix);assert p.is_file(),name;m['source_SHA256'][name]=sha(p)
    assert m['native_DSO_SHA256']=='ba6f9a01e459c5b57839677160cd916cba29312ee8355c939b898bc591ac7d81'
    assert m['topology']=={'tp_nodes':2,'pp':2,'logical_capacity':1},m['topology']
    model.write_text(json.dumps(m,indent=2)+'\n')
    p=D/'service-source-manifest-r189.json';s=json.loads(p.read_bytes());s['model_manifest_SHA256']=sha(model)
    s['overlay_SHA256']={q.relative_to(D/'overlay').as_posix():sha(q)for q in sorted((D/'overlay').rglob('*.py'))}
    s['source_SHA256']={q.relative_to(D).as_posix():sha(q)for q in sorted(D.rglob('*'))if q.is_file()and q!=p and '__pycache__'not in q.parts and q.name not in ['CPU-local-contract-r189.json']}
    s['GPU_authorized']=False;s['model_or_tools_qualified']=False;p.write_text(json.dumps(s,indent=2)+'\n')
    return sha(p)
if __name__=='__main__':print(json.dumps({'CPU_only':True,'source_manifest_SHA256':seal(),'GPU':False}))
