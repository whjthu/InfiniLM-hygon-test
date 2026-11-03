"""Private static context snapshot; only the frozen Native SHA literal changes."""
import argparse
import hashlib
import json
from pathlib import Path

from binding_proof import OLD_MODEL,validate_pin
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
MODULES=['native_context.py','checkpoint_loader.py','forward.py','common_mapping_audit.py','model_affinity_check_r2628.py']


def stage(parent,parent_mf,pin,output):
    validate_pin(pin,require_ready=True)
    parent,output=Path(parent),Path(output)
    assert sha(parent/'SOURCE_MANIFEST.json')==parent_mf
    manifest=json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())
    assert not output.exists(),'No mutation of a previous context package'
    package=output/'diagnostic_deep_context';package.mkdir(parents=True)
    records={}
    for name in MODULES:
        key='public_deep_model/'+name;source=parent/key;raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest['files_SHA256'][key]
        if name=='checkpoint_loader.py':
            old=('NATIVE_SHA='+repr(OLD_MODEL)).encode()
            assert raw.count(old)==1
            raw=raw.replace(old,('NATIVE_SHA='+repr(pin['library_SHA256'])).encode())
        (package/name).write_bytes(raw)
        records[name]=dict(parent_SHA256=sha(source),private_SHA256=sha(package/name),
            delta='NATIVE_SHA literal only'if name=='checkpoint_loader.py'else'byte-identical')
        compile(raw,str(package/name),'exec')
    (package/'__init__.py').write_bytes(b'')
    proof=dict(parent_manifest_SHA256=parent_mf,modules=records,Native_SHA256=pin['library_SHA256'],
        Native_CPU_receipt_SHA256=pin['CPU_receipt_SHA256'],CPU_READY=True,Model_READY=False,device_executed=False)
    (output/'CONTEXT_DERIVATION.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['parent','parent-mf','pin','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(stage(a.parent,a.parent_mf,json.loads(Path(a.pin).read_bytes()),a.output)))
