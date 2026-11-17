"""Only the new suffix response-header boundaries; no network or prefix reads."""
import argparse
from email.message import Message
import json
from pathlib import Path

from chain import ETAG,TAIL_START,TAIL_END,TAIL_BYTES,PREFIX_BYTES,FULL_BYTES
from tail_https import URL,TOTAL,validate_headers


class Response:
    def __init__(self,status=206,url=URL,changes=None):
        self.status=status;self.url=url;self.headers=Message();self.body_reads=0
        fields={'Content-Range':[f'bytes {TAIL_START}-{TAIL_END}/{TOTAL}'],
                'Content-Length':[str(TAIL_BYTES)],'ETag':[ETAG]}
        if changes:fields.update(changes)
        for name,values in fields.items():
            for value in values:self.headers[name]=value
    def geturl(self):return self.url
    def read(self,*_):self.body_reads+=1;raise AssertionError('Body read before suffix header acceptance')


def check():
    valid=Response();validate_headers(valid)
    if valid.body_reads!=0:raise AssertionError('Validheader unexpectedly reads body')
    bad=[('HTTP200',dict(status=200)),
        ('old_full_range',dict(changes={'Content-Range':[f'bytes 159287046-{TAIL_END}/{TOTAL}'],'Content-Length':[str(FULL_BYTES)]})),
        ('suffix_start_off_by_one',dict(changes={'Content-Range':[f'bytes {TAIL_START-1}-{TAIL_END}/{TOTAL}']})),
        ('wrong_tail_length',dict(changes={'Content-Length':[str(TAIL_BYTES+1)]})),
        ('missing_ETag',dict(changes={'ETag':[]})),('wrong_ETag',dict(changes={'ETag':['"different-object"']})),
        ('duplicate_ETag',dict(changes={'ETag':[ETAG,ETAG]})),('redirect_final_URL',dict(url='https://example.invalid/archive')),
        ('chunked',dict(changes={'Transfer-Encoding':['chunked']})),('gzip',dict(changes={'Content-Encoding':['gzip']}))]
    refused=[]
    for name,args in bad:
        response=Response(**args)
        try:validate_headers(response)
        except ValueError:refused.append(name)
        else:raise AssertionError('Accepted '+name)
        if response.body_reads:raise AssertionError('Rejectedheader readbody')
    if TAIL_START!=159287046+PREFIX_BYTES or TAIL_END-TAIL_START+1!=TAIL_BYTES or PREFIX_BYTES+TAIL_BYTES!=FULL_BYTES:
        raise AssertionError('New tail arithmetic differs')
    return dict(state='LOCAL_SUFFIX_HEADER_ONLY_BOUNDARIES_PASS',valid_headers=1,rejected=refused,
        expected_tail_start=TAIL_START,expected_tail_end=TAIL_END,expected_tail_bytes=TAIL_BYTES,
        actual_HTTP_calls=0,actual_HTTP_prefix_bytes=0,original_partial_read=False,body_read_calls=0,
        memory_probe_or_selector_reexecuted=False,real_image_roster=0,planned_images=16)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=check()
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],rejected=len(result['rejected']),actual_HTTP_calls=0,original_partial_read=False)))
