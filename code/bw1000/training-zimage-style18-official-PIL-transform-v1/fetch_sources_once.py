"""Bounded official source-only download; never install/import torchvision/Torch."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

D=Path(__file__).resolve().parent
G=D.parents[2]
OUT=G/'local-only/training-zimage-style18-official-PIL-transform-v1-source-original'
REQUEST='3bb488f111fa4c6d9dd558f15821e83e'
FILES={'functional.py':'torchvision/transforms/functional.py',
       '_functional_pil.py':'torchvision/transforms/_functional_pil.py','LICENSE':'LICENSE'}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    epoch=time.time();mono=time.monotonic();OUT.mkdir()
    result={'schema':'OFFICIAL_TORCHVISION_PIL_SOURCE_ACQUISITION_V1','request_id':REQUEST,'epoch':epoch,
            'whole_seconds_max':60,'planned_GETs':3,'actual_GETs':0,'download_bytes':0,'files':{},
            'passed':False,'Torch_import_or_install':False,'target_version_or_source_identity_qualified':False}
    save('intent.json',{'request_id':REQUEST,'epoch':epoch,'deadline':epoch+60,'URLs':FILES,'tag':'v0.24.0',
                        'max_each_bytes':98304,'max_total_bytes':196608,'redirect_retry_fallback':False})
    opener=urllib.request.build_opener(NoRedirect())
    try:
        for name,relative in FILES.items():
            left=60-(time.monotonic()-mono)
            if left<=0:raise TimeoutError('Single source epoch elapsed')
            url='https://raw.githubusercontent.com/pytorch/vision/v0.24.0/'+relative
            record={'URL':url,'status':None,'read_bytes':0,'natural_EOF':False};result['files'][name]=record
            result['actual_GETs']+=1
            with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=min(15,left)) as response:
                record.update(status=response.status,Content_Type=response.headers.get('Content-Type'),Content_Length=response.headers.get_all('Content-Length'),ETag=response.headers.get('ETag'))
                if response.status!=200 or response.url!=url or response.headers.get('Content-Encoding') not in (None,'identity'):
                    raise ValueError('Exact official source TLS200/identity required')
                length=response.headers.get_all('Content-Length')
                if length is not None and (len(length)!=1 or not length[0].isdigit() or int(length[0])>98304):
                    raise ValueError('Oversize/ambiguous source framing')
                with (OUT/name).open('xb') as f:
                    while True:
                        if time.monotonic()-mono>=60:raise TimeoutError('Single source epoch elapsed')
                        b=response.read1(min(65536,98305-record['read_bytes']))
                        if not b:record['natural_EOF']=True;break
                        record['read_bytes']+=len(b);result['download_bytes']+=len(b)
                        if record['read_bytes']>98304 or result['download_bytes']>196608:raise ValueError('Source byte cap exceeded')
                        f.write(b)
                    f.flush();os.fsync(f.fileno())
            body=(OUT/name).read_bytes()
            if length is not None and len(body)!=int(length[0]):raise ValueError('Natural EOF source length differs')
            record.update(bytes=len(body),SHA256=hashlib.sha256(body).hexdigest())
        license_body=(OUT/'LICENSE').read_text('utf8')
        if not all(x in license_body for x in ['BSD 3-Clause License','Redistribution and use','Soumith Chintala']):
            raise ValueError('Official source license statement differs')
        result['passed']=True
    except Exception as e:
        result.update(error_type=type(e).__name__,reason=str(e)[:512])
    result.update(wall_seconds=time.monotonic()-mono,state='KNOWN_TERMINAL')
    save('result.json',result)
    print(json.dumps({k:result[k] for k in ['passed','actual_GETs','download_bytes','state']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
