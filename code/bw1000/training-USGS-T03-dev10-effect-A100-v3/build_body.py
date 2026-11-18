"""Create one source-only installer body after root publication/adoption; no dispatch."""
import base64
import hashlib
import json
from pathlib import Path
import re
import runpy
import time
D=Path(__file__).resolve().parent

def make_body(publication, request_id, epoch):
    result=runpy.run_path(str(D/'verify_package.py'),run_name='package_verification')['verify']()
    assert result['root_GO'] is True and result['GPU_GO'] is True
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{32}',request_id)
    assert 0 < epoch <= time.time() < epoch+740
    mf=(D/'SOURCE_FILES.json').read_bytes()
    names=[*json.loads(mf)['files'],'SOURCE_FILES.json']
    payload=dict(publication=publication,request_id=request_id,epoch=epoch,
        source_manifest_SHA256=hashlib.sha256(mf).hexdigest(),
        claim_SHA256=hashlib.sha256((D/'CLAIM.json').read_bytes()).hexdigest(),
        package={name:base64.b64encode((D/name).read_bytes()).decode() for name in names})
    packed=base64.b64encode(json.dumps(payload,separators=(',',':')).encode()).decode()
    assert len(packed)<4*1024**2 and time.time()<epoch+740
    return 'INSTALL_PAYLOAD_B64='+repr(packed)+'\n'+(D/'install_and_run.py').read_text()
