"""Synthetic local JPEG and fake HTTP guards only; no actual dataset body or network."""
import argparse
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import time

from PIL import Image
import media_engine as engine


def main(output):
    require = engine.require
    require(not output.exists(), 'CPU receipt already exists')
    passed = []; rejected = []
    def bad(name, function):
        try:
            function()
        except (ValueError, OSError):
            rejected.append(name)
        else:
            raise AssertionError(name+' was not rejected')
    row = dict(coco_image_id=1, file_name='000000000001.jpg', width=2, height=3)
    url = engine.image_url(row)
    class Response:
        status = 200
        def __init__(self, pairs, body=b'', target=url):
            self.headers = Message(); self.body = body; self.offset = 0; self.read_calls = 0; self.target = target
            for k,v in pairs:
                self.headers[k] = v
        def geturl(self): return self.target
        def read(self, count):
            self.read_calls += 1; value = self.body[self.offset:self.offset+count]; self.offset += len(value); return value
        def close(self): pass
    base = [('Content-Length','100'),('Content-Type','image/jpeg'),('ETag','"synthetic"')]
    require(engine.validate_headers(Response(base),url)['Content_Length'] == 100,'Valid header rejected'); passed.append('exact_200_JPEG_headers')
    cases = {
        'missing_length':[p for p in base if p[0]!='Content-Length'],
        'duplicate_length':base+[('Content-Length','100')],
        'zero_length':[('Content-Length','0')]+base[1:],
        'over2MiB':[('Content-Length','2097153')]+base[1:],
        'nondecimal_length':[('Content-Length','1e6')]+base[1:],
        'missing_ETag':base[:2], 'duplicate_ETag':base+[('ETag','"other"')],
        'gzip':base+[('Content-Encoding','gzip')], 'chunked':base+[('Transfer-Encoding','chunked')],
        'unexpected_range':base+[('Content-Range','bytes 0-99/100')],
        'wrong_mime':[p for p in base if p[0]!='Content-Type']+[('Content-Type','text/html')],
    }
    for name,pairs in cases.items():
        response = Response(pairs); bad(name,lambda r=response:engine.validate_headers(r,url))
        require(response.read_calls == 0, 'Rejected header read body')
    response=Response(base); response.status=206
    bad('non200',lambda:engine.validate_headers(response,url)); require(response.read_calls==0,'Status rejection read body')
    response=Response(base,target='http://images.cocodataset.org/train2017/000000000001.jpg')
    bad('redirect_or_TLS_downgrade',lambda:engine.validate_headers(response,url)); require(response.read_calls==0,'URL rejection read body')
    bad('unsafe_name',lambda:engine.image_url(dict(coco_image_id=1,file_name='../image.jpg')))
    bad('bool_identity',lambda:engine.image_url(dict(coco_image_id=True,file_name='000000000001.jpg')))
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='synthetic-jpeg-') as td:
        path=Path(td); image=path/'synthetic.jpg'; Image.new('RGB',(2,3),(1,2,3)).save(image,format='JPEG')
        raw=image.read_bytes(); require(engine.jpeg_header(image,2,3)['full_pixel_decode'] is False,'Passive header mismatch')
        passed.append('passive_synthetic_JPEG_dimensions')
        bad('wrong_dimensions',lambda:engine.jpeg_header(image,3,2))
        broken=path/'broken.jpg'; broken.write_bytes(raw[:-2]); bad('missing_JPEG_EOI',lambda:engine.jpeg_header(broken,2,3))
        original_opener=engine.urllib.request.build_opener; original_socket=socket.socket
        class Opener:
            def __init__(self,response): self.response=response; self.calls=0
            def open(self,request,timeout):
                require(request.full_url==url and request.method=='GET' and 0<timeout<=15,'Actual fake argv/budget differs')
                self.calls+=1; return self.response
        def no_socket(*args,**kwargs): raise AssertionError('Actual socket prohibited in CPU check')
        try:
            socket.socket=no_socket
            response=Response([('Content-Length',str(len(raw))),*base[1:]],body=raw); opener=Opener(response)
            engine.urllib.request.build_opener=lambda *args:opener
            destination=path/'valid'; destination.mkdir(); progress=[]
            result=engine.acquire(row,destination,time.time()+60,time.monotonic()+60,lambda value:progress.append(dict(value)))
            require(result['actual_image_qualified'] is True and opener.calls==1 and result['bytes_read']==len(raw)
                    and result['SHA256']==hashlib.sha256(raw).hexdigest() and progress[-1]['state']=='ACTUAL_MEDIA_HEADER_AND_SHA_RECORDED','Fake acquisition failed')
            passed.append('real_acquire_code_fake_one_GET_and_full_SHA')
            response=Response(cases['over2MiB'],body=raw); opener=Opener(response)
            destination=path/'invalid'; destination.mkdir()
            result=engine.acquire(row,destination,time.time()+60,time.monotonic()+60,lambda value:None)
            require(result['actual_image_qualified'] is False and opener.calls==1 and response.read_calls==0
                    and not list(destination.iterdir()),'Overlimit fake read body/wrote file')
            rejected.append('acquire_oversize_beforebody_no_partial')
            opener=Opener(Response(base)); destination=path/'late'; destination.mkdir()
            result=engine.acquire(row,destination,time.time()+10,time.monotonic()+10,lambda value:None)
            require(result['HTTP_open_attempts']==opener.calls==0 and result['actual_image_qualified'] is False,'Late epoch began GET')
            rejected.append('late_epoch_before_open')
        finally:
            engine.urllib.request.build_opener=original_opener; socket.socket=original_socket
    records=[dict(SHA256='a'*64,actual_image_qualified=True),dict(SHA256='b'*64,actual_image_qualified=True),
             dict(SHA256='b'*64,actual_image_qualified=True),dict(SHA256='c'*64,actual_image_qualified=True)]
    old,duplicate=engine.content_dedup(records,{'a'*64})
    require((old,duplicate)==(1,2) and [r['actual_image_qualified'] for r in records]==[False,False,False,True], 'Dedup kept first duplicate or old content')
    passed.append('old64_SHA_and_all_duplicate_members_rejected')
    bad('expired_wall_and_monotonic_epoch',lambda:engine.remaining(time.time()-1,time.monotonic()-1))
    result=dict(passed=True,valid_cases=passed,rejected_cases=rejected,valid_count=len(passed),rejected_count=len(rejected),
        actual_HTTP_requests=0,real_dataset_images_read=0,synthetic_only=True,no_model_or_Torch=True,
        actual_Job_reexecuted=False,old_qualified_Job_exact_copy=True,full_real_JPEG_decode_qualified=False,
        full16_actual_media_stage_qualified=False)
    with output.open('x',encoding='utf8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('passed','valid_count','rejected_count','actual_HTTP_requests','real_dataset_images_read')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
