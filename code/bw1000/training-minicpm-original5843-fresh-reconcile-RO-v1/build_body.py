"""Prepare exact stdin source only; root owns ordinary publication and one held SSH."""
import hashlib,json,math,re,time
from pathlib import Path
D=Path(__file__).resolve().parent
def make_body(publication,request_id,epoch):
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{32}',request_id)
    assert math.isfinite(epoch) and 0<epoch<=time.time()<epoch+40
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    assert {p.name for p in D.iterdir() if p.is_file()}==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        raw=(D/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    claim=json.loads((D/'CLAIM.json').read_bytes());assert claim['root_GO'] is True and claim['READ_GO'] is True and claim['max_invocations']==1
    return ('ORIGIN='+repr(epoch)+'\nREQUEST='+repr(request_id)+'\nPUBLICATION='+repr(publication)+'\nCLAIM='+repr(claim)+'\n'+(D/'read_remote.py').read_text(encoding='utf8')).encode()
