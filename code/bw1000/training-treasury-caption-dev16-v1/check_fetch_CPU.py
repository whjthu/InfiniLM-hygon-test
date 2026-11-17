"""Header-first rejection tests; fake responses cannot read a body or network."""
import argparse
from email.message import Message
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from fetch_https import URL,EXPECTED_ETAG,BYTES,START,END,TOTAL,validate_response,fetch,remaining
from prepare_caption_dev import private_artifact_bytes


class Response:
    def __init__(self,status=206,url=URL,changes=None):
        self.status=status;self.url=url;self.read_calls=0;self.headers=Message()
        base={'Content-Range':[f'bytes {START}-{END}/{TOTAL}'],'Content-Length':[str(BYTES)],'ETag':[EXPECTED_ETAG]}
        if changes:base.update(changes)
        for key,values in base.items():
            for value in values:self.headers[key]=value
    def geturl(self):return self.url
    def read(self,*_):self.read_calls+=1;raise AssertionError('Body read during header rejection')
    def close(self):pass


def check(private):
    valid=Response();validate_response(valid);assert valid.read_calls==0
    cases=[('HTTP200',dict(status=200)),('redirect302',dict(status=302)),
        ('wrong_final_URL',dict(url='https://example.invalid/archive')),
        ('HTTP_final_URL',dict(url=URL.replace('https:','http:'))),
        ('missing_length',dict(changes={'Content-Length':[]})),
        ('whole_archive_length',dict(changes={'Content-Length':[str(TOTAL)]})),
        ('duplicate_length',dict(changes={'Content-Length':[str(BYTES),str(BYTES)]})),
        ('wrong_range',dict(changes={'Content-Range':[f'bytes 0-{END}/{TOTAL}']})),
        ('missing_range',dict(changes={'Content-Range':[]})),
        ('duplicate_range',dict(changes={'Content-Range':[f'bytes {START}-{END}/{TOTAL}']*2})),
        ('missing_ETag',dict(changes={'ETag':[]})),
        ('wrong_ETag',dict(changes={'ETag':['"wrong-object"']})),
        ('duplicate_ETag',dict(changes={'ETag':[EXPECTED_ETAG,EXPECTED_ETAG]})),
        ('gzip',dict(changes={'Content-Encoding':['gzip']})),
        ('chunked',dict(changes={'Transfer-Encoding':['chunked']}))]
    refused=[]
    for name,args in cases:
        response=Response(**args)
        try:validate_response(response)
        except ValueError:refused.append(name)
        else:raise AssertionError('Accepted '+name)
        assert response.read_calls==0
    response=Response(status=200);opencalls=[]
    class Opener:
        def open(self,request,timeout):
            opencalls.append((request.full_url,request.get_header('Range'),timeout));return response
    with tempfile.TemporaryDirectory(dir=private,prefix='header-only-') as directory:
        folder=Path(directory);assert folder.resolve().is_relative_to(private.resolve())
        with patch('urllib.request.build_opener',return_value=Opener()):
            try:fetch(folder,time.time()+10,time.monotonic()+10)
            except ValueError:pass
            else:raise AssertionError('HTTP200 accepted by fetch body')
        progress=json.loads((folder/'fetch-progress.json').read_bytes())
        assert progress['body_bytes_read']==progress['body_bytes_saved']==0 and response.read_calls==0
        assert len(opencalls)==1 and not list(folder.glob('*.partial')) and not list(folder.glob('*.bin'))
    try:remaining(time.time()-1,time.monotonic()+10)
    except ValueError:refused.append('expired_global_epoch')
    else:raise AssertionError('Expired epoch accepted')
    for name in ('roster.json','references.json','inputs.json','receipt.json'):
        assert len(private_artifact_bytes(name,{'small':'synthetic'}))<32768
        try:private_artifact_bytes(name,'x'*(262145 if name=='references.json' else 32769))
        except ValueError:refused.append('oversized_'+name)
        else:raise AssertionError('Oversized private artifact accepted')
    try:private_artifact_bytes('unbounded-unknown.json',{})
    except ValueError:refused.append('unknown_private_artifact')
    else:raise AssertionError('Unknown private artifact accepted')
    return dict(state='LOCAL_FAKE_HTTP_HEADER_GATES_PASS',valid_headers_without_body=1,rejected_cases=refused,
        actual_fetch_HTTP200_body_reads=0,stub_open_calls=1,real_HTTP_requests=0,real_dataset_bytes=0,
        real_images=0,actual_roster_selected=0,quality_metric_emitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();value=check(a.private)
    with a.output.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=value['state'],rejected_count=len(value['rejected_cases']),real_HTTP_requests=0)))
