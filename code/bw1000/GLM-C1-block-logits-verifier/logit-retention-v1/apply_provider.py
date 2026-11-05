"""Exact parent copy plus five Python modules; never executes the provider."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
PARENT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM952-prefix-continuation/native-admission-v2/provider')
DEST=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/GLM952-prefix-logit-retention-v1/provider')
H=lambda b:hashlib.sha256(b).hexdigest()
def build():
    old=(PARENT/'SOURCE_MANIFEST.json').read_bytes();assert H(old)=='8c3de5d3437c027204df83ec7c98cc8b2a422e16e6fd7f4c44054dc810f7d818'
    before=json.loads(old);raw=(D/'PROVIDER_MANIFEST.json').read_bytes();after=json.loads(raw)
    modules=['accepted_prefix_continuation.py','accepted_prefix_owner.py','resident_continuation.py','stage_compare.py','retain_logits.py']
    changes={'public_glm_native/'+n for n in modules}
    assert{k for k,h in after['files_SHA256'].items()if before['files_SHA256'].get(k)!=h}==changes
    sources={}
    for name,h in before['files_SHA256'].items():
        b=(PARENT/name).read_bytes();assert H(b)==h,name;sources[name]=b
    for name in modules:sources['public_glm_native/'+name]=(D/name).read_bytes()
    assert set(sources)==set(after['files_SHA256'])
    for name,b in sources.items():assert H(b)==after['files_SHA256'][name],name
    assert not DEST.exists(),'Immutable new namespace only'
    DEST.mkdir(parents=True)
    for name,b in sources.items():
        p=DEST/name;assert p.resolve().is_relative_to(DEST.resolve());p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b)
    with(DEST/'SOURCE_MANIFEST.json').open('xb')as f:f.write(raw)
    return dict(passed=True,provider_manifest_SHA256=H(raw),parent_manifest_SHA256=H(old),members=len(sources),changed_modules=modules,Native_compile=0,GPU=0)
if __name__=='__main__':print(json.dumps(build()))
