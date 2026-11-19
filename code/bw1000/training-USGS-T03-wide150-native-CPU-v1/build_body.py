"""Assemble exact source + private local150 adapter for one ordinary SSH stdin."""
import base64,json
from pathlib import Path
from prepare_inputs import D,canonical,digest,need,pinned,prepare,verify_source
def make_body(publication,request,epoch,private,check):
    verify_source(check);plan=json.loads((D/'PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False,'CPU_ADOPTED')
    adapter=private/'adapter';db=prepare(adapter,plan['namespace']+'/data',check)
    rawmf=(D/'SOURCE_FILES.json').read_bytes();mf=json.loads(rawmf)['files']
    package={'SOURCE_FILES.json':base64.b64encode(rawmf).decode()}
    for name,pin in mf.items():package[name]=base64.b64encode(pinned(D/name,pin,2*1024**2,check)).decode()
    datanames=set(db['files'])|set(db['media'])|{'data-binding-private.json'};data={}
    for name in sorted(datanames):
        check();p=adapter/name
        if name=='data-binding-private.json':raw=canonical(db)
        else:raw=pinned(p,db['files'].get(name,db['media'].get(name)),2*1024**2,check)
        data[name]=base64.b64encode(raw).decode()
    payload=dict(case='T03.sft',request_id=request,publication=publication,epoch=epoch,source_manifest_SHA256=digest(rawmf),claim_SHA256=digest((D/'CLAIM.json').read_bytes()),package=package,data=data,data_binding_SHA256=digest(canonical(db)))
    raw=canonical(payload);need(len(raw)<=4*1024**2,'SOURCE_DATA_PAYLOAD4M')
    check();return ('INSTALL_PAYLOAD_B64='+repr(base64.b64encode(raw).decode())+'\n'+(D/'install_and_run.py').read_text()).encode()
