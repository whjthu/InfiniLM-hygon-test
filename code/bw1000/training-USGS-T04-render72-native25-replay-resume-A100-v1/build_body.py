"""Source-only ordinary SSH body; no raw data/state payload reaches PC."""
import base64,json
from common import D,canonical,sha,need,verify_source

def make_body(publication,request,epoch,private,check):
    verify_source();check();intent=json.loads((private/'intent.json').read_bytes());files={p.name:base64.b64encode(p.read_bytes()).decode() for p in D.iterdir() if p.is_file()};x=dict(context=intent,files=files);b=canonical(x);need(len(b)<=4*1024**2,'SMALL_SOURCE_ONLY_INSTALL');return ('INSTALL_B64='+repr(base64.b64encode(b).decode())+'\n'+(D/'install_and_run.py').read_text(encoding='utf-8')).encode()
