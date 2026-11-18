"""Only create root's future bounded stdin body; no SSH or dataset operations."""
import argparse
import hashlib
import json
from pathlib import Path
import re

D=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--request-id',required=True);p.add_argument('--publication',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert re.fullmatch('[0-9a-f]{32}',a.request_id) and re.fullmatch('[0-9a-f]{40}',a.publication)
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        raw=(D/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    binding=json.loads((D/'BINDING.json').read_bytes());assert binding['root_GO'] is True and binding['RO_GO'] is True
    context=dict(epoch=a.epoch,deadline=a.epoch+30,request_id=a.request_id,publication=a.publication)
    body=('BINDING='+repr(binding)+'\nCONTEXT='+repr(context)+'\n').encode()+(D/'read_headers.py').read_bytes()
    assert len(body)<=32768
    with a.output.open('xb') as f:f.write(body)
    print(json.dumps(dict(bytes=len(body),SHA256=hashlib.sha256(body).hexdigest(),SSH=False)))


if __name__=='__main__':main()
